from gtts import gTTS
from .tts_base import TTSBase


class GTTS_tts(TTSBase):
    def __init__(self):
        super().__init__()
        
    def synthesis(self, text, output_path=None, **kwargs):
        """
        Synthesize text using the gTTS model.
        """
        tts = gTTS(text=text, lang='ar', slow=False)
        
        if output_path:
            tts.save(output_path)
        
        else:
            tts.play()
            
        return tts