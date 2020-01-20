import io
import ssl
from faker import Faker
from flask import Flask
from flask_restful import Api, Resource, request
from flask_cors import CORS

app = Flask(__name__, static_url_path='/', static_folder='frontend')
CORS(app)
api = Api(app)


class FakeText(Resource):
    """REST end point for the fake text service."""

    def __init__(self):
        self.faker = Faker()

    def post(self):
        """Create fake text.

        Create four different continuations for text. 
        """

        json = request.get_json()
        text = json.get('text')

        answer = self.faker.generate_text(text)

        data = {"result": answer, "input": json}
        return data, 200


api.add_resource(FakeText, "/text")

if __name__ == "__main__":
    app.run(debug=True)
