from abc import ABC, abstractmethod
import io

class BaseSTTProvider(ABC):
    def __init__(self, modle_name):
        self.model_name = modle_name
    
    
    def process_audio(self, audio_data):
        if isinstance(audio_data, bytes):  # If raw bytes from Streamlit
            audio_bytes = audio_data
        
        elif isinstance(audio_data, io.BytesIO):  # If it's a BytesIO object
            audio_bytes = audio_data.getvalue()  # Convert to raw bytes

        elif isinstance(audio_data, str):  # If a file path is given
            with open(audio_data, "rb") as file:
                audio_bytes = file.read()
        
        else:
            raise ValueError(f"Invalid audio data type: {type(audio_data)}. Supported types: bytes, str")

        return audio_bytes
    
    @abstractmethod
    def transcribe(self, audio_file, model_name):

        raise NotImplementedError("Subclass must implement abstract method")