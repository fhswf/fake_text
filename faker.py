from model import ModelWrapper


class Faker:

    def __init__(self):
        self.mw = ModelWrapper.load()

    def generate_text(self, text, tokens_to_generate=100, max_sentences=3, top_k=8):
        tokens = self.mw.tokenize(text)

        continuations = [self.mw.generate_tokens(
            tokens, tokens_to_generate, max_sentences, top_k) for i in range(4)]

        return continuations
