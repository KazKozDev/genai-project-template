"""Example: Basic completion with all utilities."""

import yaml
from pathlib import Path

from src.llm import ClaudeClient
from src.utils import RateLimiter, TokenCounter, ResponseCache
from src.handlers import retry_on_error


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    return yaml.safe_load(Path(path).read_text())


def main():
    # Load configurations
    model_config = load_config("config/model_config.yaml")
    prompts_config = load_config("config/prompts.yaml")
    
    # Initialize utilities
    rate_limiter = RateLimiter(
        requests_per_minute=model_config["rate_limits"]["requests_per_minute"],
        tokens_per_minute=model_config["rate_limits"]["tokens_per_minute"]
    )
    
    token_counter = TokenCounter()
    
    cache = ResponseCache(
        cache_dir=model_config["cache"]["directory"],
        ttl_seconds=model_config["cache"]["ttl_seconds"]
    )
    
    # Initialize client
    client = ClaudeClient(model_config["models"]["claude"])
    
    # Get prompt template
    prompt_template = prompts_config["templates"]["summarize"]
    
    # Example text to summarize
    text = """
    Artificial intelligence has transformed numerous industries over the past decade.
    From healthcare diagnostics to autonomous vehicles, AI systems are becoming
    increasingly sophisticated and capable. However, with these advances come
    important ethical considerations around bias, privacy, and job displacement.
    """
    
    # Format prompt
    prompt = prompt_template.format(text=text, length="2")
    
    # Check cache first
    cached_response = cache.get(prompt, client.model_name)
    if cached_response:
        print("Using cached response:")
        print(cached_response)
        return
    
    # Count tokens
    input_tokens = token_counter.count(prompt)
    print(f"Input tokens: {input_tokens}")
    
    # Apply rate limiting
    rate_limiter.acquire(tokens=input_tokens)
    
    # Make API call with retry
    @retry_on_error(max_retries=3, delay=1.0, backoff=2.0)
    def get_completion():
        return client.complete(prompt)
    
    response = get_completion()
    
    # Cache response
    cache.set(prompt, client.model_name, response)
    
    # Output
    print("Response:")
    print(response)
    
    # Show usage stats
    print(f"\nRate limiter stats: {rate_limiter.get_current_usage()}")
    print(f"Cache stats: {cache.stats()}")


if __name__ == "__main__":
    main()
