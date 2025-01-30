from abc import ABC, abstractmethod


class TTSBase(ABC):
    def __init__(self):
        pass        
        
    @abstractmethod
    def synthesis(self, text: str, **kwargs) -> bytes:
        pass