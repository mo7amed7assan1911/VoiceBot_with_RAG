import os
from elevenlabs import ElevenLabs, save, play
from .tts_base import TTSBase

class ElevenLabs_tts(TTSBase):
    def __init__(self):
        super().__init__()
        
        api_key = os.getenv("ELEVENLABS_API_KEY")
        self.model = ElevenLabs(api_key=api_key)
        

    def synthesis(self, text, output_path=None, **kwargs):
        """
        Synthesize text using the ElevenLabs model.
        """
        
        voice_id = kwargs.get('voice_id', 'IK7YYZcSpmlkjKrQxbSn')
        model = kwargs.get('model', 'eleven_multilingual_v2')
        
        audio = self.model.generate(
            text=text,
            voice=voice_id,
            model=model
        )
        
        play(audio)
        if output_path:
            save(audio, output_path)