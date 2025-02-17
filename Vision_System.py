from Speech_to_text_Providers.stt_manager import SpeechToTextManager

from dotenv import load_dotenv
load_dotenv()
import os

class image_and_query_to_text:
    def __init__(self, 
                translator_provider='groq',
                vision_provider='groq',
                translation_model_name='llama-3.3-70b-versatile',
                vision_model_name='llama-3.2-11b-vision-preview',
                max_tokens=250,
                temperature=0.4,
                streaming_mode=False,
                STT_PROVIDER_NAME='groq',
                STT_MODEL_NAME='whisper-large-v3'
                ):
                
        self.streaming_mode = streaming_mode
        
        self.stt_manager = SpeechToTextManager(mode=STT_PROVIDER_NAME, model_name=STT_MODEL_NAME)
        
        if translator_provider == "groq":
            from Text_to_Text_Providers.groq_provider import GroqProvider
            self.translation_model = GroqProvider(translation_model_name, max_tokens, temperature)

        if vision_provider == "groq":
            from Vision_Providers.groq_provider import GroqProvider
            self.vision_model = GroqProvider(vision_model_name, max_tokens, temperature)

    def transcribe_audio(self, audio_path):
        transcription = self.stt_manager.transcribe(audio_path)
        return transcription

    def describe_the_image(self, image_path, query, streaming_mode=False):
        
        if streaming_mode:
            response = self.vision_model.get_stream(image_path=image_path, user_prompt=query)
        else:
            response = self.vision_model.get_response(image_path=image_path, user_prompt=query)
            
        return response
    
    def translate_text(self, text, streaming_mode=False):
        
        system_message = "You are an AI model that answers user questions in Arabic based on an image description provided by a vision model. You will receive the image description in English along with the user's question. Your response should be clear, natural, and directly answer the question using the provided description. If the description lacks enough details, inform the user instead of making assumptions."
        
        if streaming_mode:
            response = self.translation_model.get_stream(prompt=text, system_message=system_message)
        else:
            response = self.translation_model.get_response(prompt=text, system_message=system_message)
            
        return response


    def process_user_image(self, image_path, query, streaming_mode=False):
        image_description = self.describe_the_image(image_path, query, streaming_mode=streaming_mode)
        
        final_query = f"""
        User Query: {query},
        Image Description: {image_description}
        """
        
        final_response = self.translate_text(final_query, streaming_mode=streaming_mode)
        
        return final_response
        
        
if __name__ == "__main__":
    vision_pipeline = image_and_query_to_text()
    
    # Process the user message
    image_path = 'images/System_Design.png'
    query = "what is the name of the vectore database in the image?"
    streaming_mode = False
    
    final_response = vision_pipeline.process_user_image(image_path, query, streaming_mode=streaming_mode)
    
    for chunk in final_response:
        print(chunk, flush=True, end='')