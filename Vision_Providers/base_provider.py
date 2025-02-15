from abc import ABC, abstractmethod
import base64


class BaseModelProvider(ABC):
    def __init__(self, model_name, max_tokens=250, temperature=0.4):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    
    @abstractmethod
    def get_response(self, image, prompt):
        raise NotImplementedError("Subclass must implement abstract method")