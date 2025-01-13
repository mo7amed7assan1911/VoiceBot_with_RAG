import os
import openai
from .base_provider import BaseModelProvider

class OpenAIProvider(BaseModelProvider):
    print("Using OpenAI as the model provider ... ")
    def __init__(self, model_name=None, max_tokens=None, temperature=None):
        super().__init__(model_name, temperature, max_tokens)
        openai.api_key = os.getenv('OPENAI_API_KEY')

    def get_response(self, prompt):
        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response['choices'][0]['text'].strip()
