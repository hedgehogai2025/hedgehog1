import openai
import logging
import os

logger = logging.getLogger(__name__)

class OpenAIHelper:
    """Helper class for OpenAI API interactions compatible with version 1.0+"""
    
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"
    
    def generate_content(self, prompt):
        """Generate content using OpenAI API with new interface"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specialized in cryptocurrency and artificial intelligence."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating AI content: {str(e)}")
            return None