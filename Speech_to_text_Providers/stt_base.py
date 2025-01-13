from abc import ABC, abstractmethod

class BaseSTTProvider(ABC):
    def __init__(self, modle_name):
        self.model_name = modle_name
        
    @abstractmethod
    def transcribe(self, audio_file_path, model_name):
        raise NotImplementedError("Subclass must implement abstract method")