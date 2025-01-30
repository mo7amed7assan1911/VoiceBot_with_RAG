from .tts_elevenlabs import ElevenLabs_tts
# from .tts_gTTs import GTTS_tts

class TextToSpeachManager:
    def __init__(self, mode):
        
        if mode == 'gtts':
            # self.provider = GTTS_tts()
            pass
        elif mode == 'elevenlabs':
            self.provider = ElevenLabs_tts()
        else:
            raise ValueError("Invalid mode. Supported modes: 'gtts', 'elevenlabs'")
        
    def synthesis(self, text, **kwargs):
        """
        Synthesize text using the selected provider.
        """
        return self.provider.synthesis(text, **kwargs)
