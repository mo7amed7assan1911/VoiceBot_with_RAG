import torch
from transformers import pipeline
from .base_provider import BaseModelProvider


class HuggingFaceProvider(BaseModelProvider):
    print("Using HuggingFace as the model provider ... ")
    def __init__(self, model_name=None, max_tokens=None, temperature=None):
        super().__init__(model_name, max_tokens, temperature)
        
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("text2text-generation", model=self.model_name, device=device)
        
        
    def get_response(self, prompt):
        response = self.pipe(prompt, max_length=self.max_tokens, temperature=self.temperature)
        return response