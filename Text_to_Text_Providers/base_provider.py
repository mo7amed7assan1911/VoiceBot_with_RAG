from abc import ABC, abstractmethod
class BaseModelProvider(ABC):
    def __init__(self, model_name, max_tokens=250, temperature=0.4):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def get_response(self, prompt):
        raise NotImplementedError("Subclass must implement abstract method")
