from flask import Flask, request
from flask_restful import Api, Resource

from model import initialize_model
from generate import generate_text

app = Flask(__name__)
api = Api(app)

DEFAULT_NUM_TOKENS = 256

model_name = "openai-community/gpt2"
model, tokenizer = initialize_model(model_name)

class TextGeneration(Resource):
    def get(self):
        return {'message': "Hello, Welcome to /GENERATE"}

    def post(self):
        # Parse input JSON data
        input_data = request.get_json()
        num_tokens = input_data.get('num_tokens') if input_data.get('num_tokens') != None else DEFAULT_NUM_TOKENS
        message = input_data.get('message') 

        # Generate text
        generated_text = generate_text(model, tokenizer, message, num_tokens)

        return {'generated_text': generated_text}

# Add resource to API
api.add_resource(TextGeneration, '/generate')

if __name__ == '__main__':
    app.run(debug=True)