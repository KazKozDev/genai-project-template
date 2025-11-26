import os
from typing import List, Dict, Any

import anthropic

from .base import BaseLLMClient


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion using Claude."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response using Claude."""
        system = kwargs.pop("system", None)
        
        create_kwargs = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": messages,
        }
        
        if system:
            create_kwargs["system"] = system
        
        response = self.client.messages.create(**create_kwargs)
        return response.content[0].text
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Claude's tokenizer."""
        response = self.client.messages.count_tokens(
            model=self.model_name,
            messages=[{"role": "user", "content": text}]
        )
        return response.input_tokens
