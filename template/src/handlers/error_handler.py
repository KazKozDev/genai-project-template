import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple, Optional

logger = logging.getLogger(__name__)


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """Decorator for retrying functions on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
        on_retry: Optional callback called on each retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        
                        if on_retry:
                            on_retry(attempt, e, current_delay)
                        
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed. "
                            f"Last error: {e}"
                        )
            
            raise last_exception
        return wrapper
    return decorator


class APIError(Exception):
    """Base exception for API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass


class TokenLimitError(APIError):
    """Raised when token limit is exceeded."""
    pass


def handle_api_error(error: Exception) -> None:
    """Log and categorize API errors.
    
    Args:
        error: The exception to handle
    """
    error_type = type(error).__name__
    
    if isinstance(error, RateLimitError):
        logger.warning(f"Rate limit exceeded: {error}")
    elif isinstance(error, AuthenticationError):
        logger.error(f"Authentication failed: {error}")
    elif isinstance(error, TokenLimitError):
        logger.warning(f"Token limit exceeded: {error}")
    else:
        logger.error(f"API error ({error_type}): {error}")
