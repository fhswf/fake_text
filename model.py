import json
import math
from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm
import torch
import numpy as np
import fire
import attr
import torch

from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint


@attr.s(auto_attribs=True, frozen=True)
class HParams:
    n_vocab: int
    n_ctx: int
    n_embed: int
    n_hidden: int
    n_head: int
    n_layer: int
    gradient_checkpointing: bool


class Model(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()
        self.hparams = hparams
        self.wpe = nn.Embedding(hparams.n_ctx, hparams.n_embed)
        nn.init.normal_(self.wpe.weight, std=0.01)
        self.wte = nn.Embedding(hparams.n_vocab, hparams.n_embed)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(hparams) for _ in range(hparams.n_layer)])
        self.ln_f = Norm(self.hparams.n_hidden)
        if hparams.n_hidden != hparams.n_embed:
            self.in_proj = Conv1D(hparams.n_embed, hparams.n_hidden)
            self.out_proj = Conv1D(hparams.n_hidden, hparams.n_embed)
        else:
            self.in_proj = self.out_proj = None

    def forward(self, x, past=None):
        # Embedding
        past_length = 0 if past is None else past.shape[-2]
        batch_size, n_ctx = x.shape
        position = position_for(batch_size, n_ctx, past_length, x.device)
        h = self.wte(x) + self.wpe(position)
        assert h.shape == (batch_size, n_ctx, self.hparams.n_embed)
        if self.in_proj:
            h = self.in_proj(h)
        # Transformer
        presents = []
        for i, block in enumerate(self.blocks):
            if self.hparams.gradient_checkpointing:
                h, present = torch.utils.checkpoint.checkpoint(
                    block, h, past[:, i] if past is not None else None)
            else:
                h, present = block(
                    h, past=past[:, i] if past is not None else None)
            presents.append(present)
        h = self.ln_f(h)
        if self.out_proj:
            h = self.out_proj(h)
        # Output logits
        h_flat = h.reshape([batch_size * n_ctx, self.hparams.n_embed])
        logits = torch.matmul(h_flat, self.wte.weight.t())
        logits = logits.reshape([batch_size, n_ctx, self.hparams.n_vocab])
        return {
            'presents': torch.stack(tuple(presents), dim=1),
            'logits': logits,
        }


class Block(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()
        self.ln_1 = Norm(hparams.n_hidden)
        self.ln_2 = Norm(hparams.n_hidden)
        self.mlp = MLP(hparams.n_hidden, hparams.n_hidden * 4)
        self.attn = Attention(hparams)

    def forward(self, x, past):
        a, present = self.attn(self.ln_1(x), past=past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class Norm(nn.Module):
    """ Normalize to mean = 0, std = 1, then do a diagonal affine transform.
    """

    def __init__(self, n_features, *, dim=-1, epsilon=1e-5):
        super().__init__()
        self.n_features = n_features
        self.dim = dim
        self.epsilon = epsilon
        self.g = nn.Parameter(torch.ones(n_features))
        self.b = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        assert x.shape[-1] == self.n_features
        u = torch.mean(x, dim=self.dim, keepdim=True)
        xmu = x - u
        s = torch.mean(xmu * xmu, dim=self.dim, keepdim=True)
        return xmu * torch.rsqrt(s + self.epsilon) * self.g + self.b


class MLP(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.c_fc = Conv1D(n_features, n_hidden)
        self.c_proj = Conv1D(n_hidden, n_features)

    def forward(self, x):
        x = gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()
        assert hparams.n_hidden % hparams.n_head == 0
        self.hparams = hparams
        self.c_attn = Conv1D(hparams.n_hidden, hparams.n_hidden * 3)
        self.c_proj = Conv1D(hparams.n_hidden, hparams.n_hidden)

    def forward(self, x, past):
        assert len(x.shape) == 3  # [batch, sequence, features]
        assert x.shape[-1] == self.hparams.n_hidden
        if past is not None:
            # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]
            assert len(past.shape) == 5
            assert past.shape[-1] == self.hparams.n_hidden
        c = self.c_attn(x)
        q, k, v = map(self.split_heads, torch.split(c, x.shape[-1], dim=2))
        present = torch.stack([k, v], dim=1)
        if past is not None:
            pk, pv = past[:, 0], past[:, 1]
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
        a = self.multihead_attn(q, k, v)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

    def split_heads(self, x):
        """ From [batch, sequence, features] to
        [batch, heads, sequence, features].
        """
        return self.split_states(x, self.hparams.n_head).permute(0, 2, 1, 3)

    @staticmethod
    def split_states(x, n):
        """ Reshape the last dimension of x into [n, x.shape[-1]/n].
        """
        *start, m = x.shape
        return x.reshape(start + [n, m // n])

    def merge_heads(self, x):
        """ Reverse of split_heads.
        """
        return self.merge_states(x.permute(0, 2, 1, 3))

    @staticmethod
    def merge_states(x):
        """ Smash the last two dimensions of x into a single dimension.
        """
        *start, a, b = x.shape
        return x.reshape(start + [a * b])

    def mask_attn_weights(self, w):
        # w has shape [batch, heads, dst_sequence, src_sequence],
        # where information flows from src to dst.
        _, _, nd, ns = w.shape
        b = self.attention_mask(nd, ns, dtype=w.dtype, device=w.device)
        b = b.reshape((1, 1, nd, ns))
        w = w * b - 1e10 * (1 - b)
        return w

    @staticmethod
    def attention_mask(nd, ns, *, dtype, device=None):
        """ 1's in the lower triangle, counting from the lower right corner.
        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd),
        but doesn't produce garbage on TPUs.
        """
        i = torch.arange(0, nd).unsqueeze(1)
        j = torch.arange(ns)
        return (i >= j - ns + nd).to(dtype=dtype, device=device)

    def multihead_attn(self, q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = torch.matmul(q, k.permute(0, 1, 3, 2))
        w = w / math.sqrt(v.shape[-1])
        w = self.mask_attn_weights(w)
        w = F.softmax(w, dim=-1)
        a = torch.matmul(w, v)
        return a


class Conv1D(nn.Linear):
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)


def gelu(x, c=math.sqrt(2 / math.pi)):
    return 0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3))))


def position_for(batch_size, n_steps, past_length, device=None):
    return (torch.arange(past_length, n_steps + past_length, device=device)
            .unsqueeze(0).repeat(batch_size, 1))


class ModelWrapper:

    DEFAULT_MODEL = Path("/home/cgawron/gpt2-german/de345-root")
    UNK = '<unk>'
    END_OF_LINE = '<endofline>'
    END_OF_TEXT = '<endoftext>'

    def __init__(self, device, model: Model, sp_model: spm.SentencePieceProcessor):
        self.device = device
        self.model = model
        self.sp_model = sp_model
        self.EOS = self.tokenize('.')

    @classmethod
    def fixed_state_dict(cls, state_dict):
        if all(k.startswith('module.') for k in state_dict):
            # legacy multi-GPU format
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        return state_dict

    @classmethod
    def load(cls, root=DEFAULT_MODEL):
        device = torch.device("cuda")
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(str(root / 'sp.model'))
        hparams = json.loads((root / 'params.json').read_text())['hparams']
        hparams.setdefault('n_hidden', hparams['n_embed'])
        model = Model(HParams(**hparams))
        state = torch.load(root / 'model.pt', map_location='cuda:0')
        model.to(device)
        state_dict = cls.fixed_state_dict(state['state_dict'])
        model.load_state_dict(state_dict)

        tensor_list = list(state_dict.items())
        for layer_tensor_name, tensor in tensor_list:
            print("Layer %-42s: %9d elements" %
                  (layer_tensor_name, torch.numel(tensor)))
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total # params: %d" % pytorch_total_params)

        return cls(device, model, sp_model)

    def tokenize(self, s: str) -> List[str]:
        return self.sp_model.EncodeAsPieces(s)

    def token_to_id(self, token: str) -> int:
        return self.sp_model.PieceToId(token)

    def id_to_token(self, token_id: int) -> str:
        return self.sp_model.IdToPiece(int(token_id))

    def get_log_probs(self, tokens: List[str]) -> torch.Tensor:
        """ Return a tensor with shape (len(tokens), len(self.sp_model)),
        with log-probabilities for tokens after each token in tokens.
        If this is a start of the text, you may want to prepend END_OF_TEXT:
        model.get_log_probs([model.END_OF_TEXT] + tokens).
        Use model.tokenize to obtain tokens.
        """
        assert len(tokens) <= self.model.hparams.n_ctx  # TODO
        ids = [self.token_to_id(t) for t in tokens]
        ctx = torch.LongTensor(ids).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(ctx)['logits'].squeeze(0)
            return torch.log_softmax(logits, dim=1)

    def get_occurred_log_probs(
            self, tokens: List[str]) -> List[Tuple[float, str]]:
        """ Return a list of log probs of actually occurred tokens,
        starting from the second.
        """
        log_probs = self.get_log_probs(tokens)
        out = []
        for idx, token in enumerate(tokens[1:]):
            out.append((float(log_probs[idx, self.token_to_id(token)]), token))
        return out

    def get_next_top_k(
            self, tokens: List[str], top_k: int) -> List[Tuple[float, str]]:
        """ Return a list of top k tuples of log prob and token,
        for what would come after the last token.
        """
        next_log_probs = self.get_log_probs(tokens)[-1]
        return sorted([(float(next_log_probs[i]), self.id_to_token(i))
                       for i in next_log_probs.argsort()[-top_k:]],
                      reverse=True)

    def generate_tokens(self, tokens_prefix: List[str], tokens_to_generate: int, max_sentences: int, top_k: int) -> List[str]:

        tokens = list(tokens_prefix)
        sentence = 0
        for i in range(tokens_to_generate):

            # generate TOP_K potential next tokens
            ntk = self.get_next_top_k(tokens, top_k)

            # convert log probs to real probs
            logprobs = np.array(list(map(lambda a: a[0], ntk)))
            probs = np.exp(logprobs) / np.exp(logprobs).sum()

            # pick next token randomly according to probs distribution
            next_token_n = np.random.choice(top_k, p=probs)
            next_token = ntk[next_token_n][1]
            # print (next_token)

            tokens.append(next_token)
            if next_token == self.EOS:
                sentence += 1
                if sentence >= max_sentences:
                    break

        return self.sp_model.DecodePieces(tokens)
