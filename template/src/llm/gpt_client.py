import os
from typing import List, Dict, Any

import openai
import tiktoken

from .base import BaseLLMClient


class GPTClient(BaseLLMClient):
    """OpenAI GPT API client."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.tokenizer = tiktoken.encoding_for_model(
            self.model_name.replace("gpt-4o", "gpt-4")
        )
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion using GPT."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response using GPT."""
        system = kwargs.pop("system", None)
        
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=messages
        )
        return response.choices[0].message.content
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.tokenizer.encode(text))
