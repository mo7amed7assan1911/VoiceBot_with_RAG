import os
from elevenlabs import ElevenLabs, save, play, stream
from .tts_base import TTSBase

class ElevenLabs_tts(TTSBase):
    def __init__(self):
        super().__init__()
        api_key = os.getenv('ELEVENLABS_API_KEY')
        print(f'Key: {api_key}')
        self.model = ElevenLabs(api_key=api_key)
        print('intialized the model')

    def synthesis(self, text, **kwargs):
        """
        Synthesize text using the ElevenLabs model.
        """
        
        output_path = kwargs.get('output_path', None)
        voice_id = kwargs.get('voice_id', 'IK7YYZcSpmlkjKrQxbSn')
        model = kwargs.get('model', 'eleven_multilingual_v2')
        streaming_mode = kwargs.get('streaming_mode', False)
        
        print(f"Streaming type is {type(streaming_mode)}")
        audio = self.model.generate(
            text=text,
            voice=voice_id,
            model=model,
            stream=True
        )
        
        # stream(audio)
        # play(audio)
        
        if output_path:
            save(audio, output_path)
            
        return audio