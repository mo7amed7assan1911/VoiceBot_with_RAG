import os
from .base_provider import BaseModelProvider

class GroqProvider(BaseModelProvider):
    def __init__(self, model_name=None, max_tokens=None, temperature=None):
        print("Using groq as the model provider ... ")
        from groq import Groq
        super().__init__(model_name, max_tokens, temperature) # inhirit from the parent class
        
        # os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
        self.client = Groq()
    

    def get_response(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional conference assistant fluent in Arabic and English. Respond concisely and professionally ONLY IN ARABIC"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens, # comming from the parent class
            temperature=self.temperature, # comming from the parent class
        )
        
        return response.choices[0].message.content
    