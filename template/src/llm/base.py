from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.
    
    All LLM implementations should inherit from this class
    to ensure consistent interface across providers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.7)
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response for a chat conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        pass
    
    def validate_config(self) -> bool:
        """Validate that required configuration is present."""
        required = ["model_name"]
        return all(key in self.config for key in required)
