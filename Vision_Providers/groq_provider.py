from .base_provider import BaseModelProvider
import os

class GroqProvider(BaseModelProvider):
    def __init__(self, model_name=None, max_tokens=None, temperature=None):
        print("Using groq as the model provider ... ")
        from groq import Groq
        super().__init__(model_name, max_tokens, temperature) # inhirit from the parent class
        
        # os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
        self.client = Groq()    

    def get_response(self, image_path, user_prompt="Describe the image"):

        base64_image = self.encode_image(image_path)
                
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        return chat_completion.choices[0].message.content
        
        
    def get_stream(self, image_path, user_prompt="Describe the image"):

        base64_image = self.encode_image(image_path)

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True
        )
        
        for chunk in chat_completion:
            yield chunk.choices[0].delta.content