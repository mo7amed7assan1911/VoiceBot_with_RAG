from abc import ABC, abstractmethod


class TTSBase(ABC):
    def __init__(self):
        pass        
        
    @abstractmethod
    def synthesis(self, text: str, output_path: str, **kwargs) -> bytes:
        pass