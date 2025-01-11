import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from .stt_base import BaseSTTProvider

class HuggingFaceSpeechToText(BaseSTTProvider):
    def __init__(self, model_name):
        super().__init__(model_name)
        print("Your sound-to-text (STT) model running with Hugging Face")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(self.model_name)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe(self, audio_data):
        """
        Transcribe audio using the Hugging Face model.
        """
        return self.pipe(audio_data)["text"].strip()
