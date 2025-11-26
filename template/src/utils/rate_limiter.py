import time
from collections import deque
from threading import Lock
from typing import Optional


class RateLimiter:
    """Thread-safe rate limiter for API calls.
    
    Implements a sliding window rate limiting algorithm.
    """
    
    def __init__(
        self, 
        requests_per_minute: int,
        tokens_per_minute: Optional[int] = None
    ):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.request_timestamps = deque()
        self.token_counts = deque()
        self.lock = Lock()
    
    def acquire(self, tokens: int = 0) -> None:
        """Wait until a request can be made within rate limits.
        
        Args:
            tokens: Number of tokens for this request (optional)
        """
        with self.lock:
            now = time.time()
            
            # Clean old timestamps
            while (self.request_timestamps and 
                   self.request_timestamps[0] < now - 60):
                self.request_timestamps.popleft()
                if self.token_counts:
                    self.token_counts.popleft()
            
            # Check request rate limit
            if len(self.request_timestamps) >= self.rpm:
                sleep_time = 60 - (now - self.request_timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
            
            # Check token rate limit
            if self.tpm and tokens > 0:
                current_tokens = sum(self.token_counts)
                if current_tokens + tokens > self.tpm:
                    sleep_time = 60 - (now - self.request_timestamps[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            
            self.request_timestamps.append(time.time())
            if tokens > 0:
                self.token_counts.append(tokens)
    
    def get_current_usage(self) -> dict:
        """Get current rate limit usage statistics."""
        with self.lock:
            now = time.time()
            
            # Clean old timestamps
            while (self.request_timestamps and 
                   self.request_timestamps[0] < now - 60):
                self.request_timestamps.popleft()
                if self.token_counts:
                    self.token_counts.popleft()
            
            return {
                "requests_used": len(self.request_timestamps),
                "requests_limit": self.rpm,
                "tokens_used": sum(self.token_counts) if self.token_counts else 0,
                "tokens_limit": self.tpm
            }
