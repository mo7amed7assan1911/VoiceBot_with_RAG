import os
from faster_whisper import WhisperModel
from .stt_base import BaseSTTProvider

class LocalSpeechToText(BaseSTTProvider):
    def __init__(self, model_name="large"):
        super().__init__(model_name)
        print("Your sound-to-text (STT) model running locally")
        self.whisper_model = WhisperModel(
            self.model_name, device="cpu", cpu_threads=os.cpu_count() // 2
        )

    def transcribe(self, audio_data):
        """
        Transcribe audio using the local Whisper model.
        """
        segments, _ = self.whisper_model.transcribe(audio_data, language="ar")
        return "".join(segment.text for segment in segments)
