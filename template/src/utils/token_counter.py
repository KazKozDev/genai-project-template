from typing import Optional

import tiktoken


class TokenCounter:
    """Utility for counting and managing tokens."""
    
    def __init__(self, model: str = "cl100k_base"):
        """Initialize with a tiktoken encoding.
        
        Args:
            model: Encoding name or model name
        """
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoder = tiktoken.get_encoding(model)
    
    def count(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.encoder.encode(text))
    
    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit.
        
        Args:
            text: Input text
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens])
    
    def split_by_tokens(self, text: str, chunk_size: int) -> list:
        """Split text into chunks of specified token size.
        
        Args:
            text: Input text
            chunk_size: Tokens per chunk
            
        Returns:
            List of text chunks
        """
        tokens = self.encoder.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(self.encoder.decode(chunk_tokens))
        
        return chunks
    
    def estimate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int,
        input_cost_per_million: float,
        output_cost_per_million: float
    ) -> float:
        """Estimate API cost for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            input_cost_per_million: Cost per 1M input tokens
            output_cost_per_million: Cost per 1M output tokens
            
        Returns:
            Estimated cost in dollars
        """
        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million
        return input_cost + output_cost
