from dotenv import load_dotenv
load_dotenv()
import os

class image_and_query_to_text:
    def __init__(self, 
                llm_provider='groq',
                vision_provider='groq',
                translation_model_name='llama-3.3-70b-versatile',
                vision_model_name='llama-3.2-11b-vision-preview',
                max_tokens=250,
                temperature=0.4,
                streaming_mode=False):
        
        self.streaming_mode = streaming_mode
        
        # Initialize the model provider
        # if llm_provider == "groq":
        #     from Text_to_Text_Providers.groq_provider import GroqProvider
        #     self.text_model = GroqProvider(translation_model_name, max_tokens, temperature)

        if vision_provider == "groq":
            from Vision_Providers.groq_provider import GroqProvider
            self.vision_model = GroqProvider(vision_model_name, max_tokens, temperature)

    def describe_the_image(self, image_path, query, streaming_mode=False):
        
        if streaming_mode:
            response = self.vision_model.get_stream(image_path=image_path, user_prompt=query)
        else:
            response = self.vision_model.get_stream(image_path=image_path, user_prompt=query)
            
        return response

if __name__ == "__main__":
    vision_pipeline = image_and_query_to_text()
    
    # Process the user message
    image_path = 'images/System_Design.png'
    query = "Describe the image"
    streaming_mode = True
    response = vision_pipeline.describe_the_image(image_path, query, streaming_mode=streaming_mode)

    for txt in response:
        print(txt, flush=True, end="")
    # print(response)