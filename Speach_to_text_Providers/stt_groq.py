import os
from groq import Groq
from .stt_base import BaseSTTProvider

class GroqSpeechToText(BaseSTTProvider):
    def __init__(self, modle_name):
        super().__init__(modle_name)
        print("Your sound-to-text (STT) model running online")
        
        self.grok_client = Groq()

    def transcribe(self, audio_data):
        """
        Transcribe audio using the Groq API.
        """
        # with open(audio_data, "rb") as file:
        #     transcription = self.grok_client.audio.transcriptions.create(
        #         file=(audio_data, file.read()),
        #         model=self.modle_name,
        #         prompt="Specify context or spelling",
        #         response_format="json",
        #         temperature=0.0,
        #     )
            
        
        with open(audio_data, "rb") as file:
            # Create a transcription of the audio file
            transcription = self.grok_client.audio.transcriptions.create(
            file=(audio_data, file.read()), # Required audio file
            model="whisper-large-v3", # Required model to use for transcription
            prompt="Specify context or spelling",  # Optional
            response_format="json",  # Optional
            # language="ar",  # Optional
            temperature=0.0  # Optional
        )

        return transcription.text.strip()
