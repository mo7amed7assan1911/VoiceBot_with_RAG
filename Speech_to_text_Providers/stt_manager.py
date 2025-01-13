from .stt_local import LocalSpeechToText
from .stt_groq import GroqSpeechToText
from .stt_hugging_face import HuggingFaceSpeechToText

class SpeechToTextManager:
    def __init__(self, mode, model_name):
        """
        Initialize the Speech-to-Text manager.

        Args:
            mode (str): The mode of the provider (e.g., 'local', 'groq', 'hugging_face').
            kwargs: Additional arguments for provider initialization.
        """
        self.model_name = model_name
        self.provider = self._load_provider(mode)

    def _load_provider(self, mode):
        if mode == "local":
            return LocalSpeechToText(model_name=self.model_name)
        
        elif mode == "groq":
            return GroqSpeechToText(modle_name=self.model_name)
        
        elif mode == "hugging_face":
            return HuggingFaceSpeechToText(model_name='openai/whisper-large')
        
        else:
            raise ValueError("Invalid mode. Supported modes: 'local', 'groq', 'hugging_face'")

    def transcribe(self, audio_data):
        """
        Transcribe audio using the selected provider.
        """
        return self.provider.transcribe(audio_data)
