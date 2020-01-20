import io
import ssl
import urllib.request
from faker import Faker
from flask import Flask, send_from_directory
from flask_restful import Api, Resource, reqparse, request
from flask_cors import CORS

app = Flask(__name__, static_url_path='/ui')
CORS(app)
api = Api(app)


@app.route('/<path:path>')
def send_static(path):
    """Serve static content.
    Static resources for the demo interface are served via the frontend directory.

    Parameters
    ----------
    path : str
        Path of requested rersource.
    """
    return send_from_directory('frontend', path)


class FakeText(Resource):
    """REST end point for the fake text service.
    """

    def __init__(self):
        self.faker = Faker()

    def post(self):
        """
        """

        json = request.get_json()
        text = json.get('text')

        answer = self.faker.generate_text(text)

        data = {"result": answer, "input": json}
        return data, 201


api.add_resource(FakeText, "/text")

if __name__ == "__main__":
    app.run(debug=True)
