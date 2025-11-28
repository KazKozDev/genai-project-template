# Agent System Prompt

> Copy everything below the line into your AI agent's system prompt or custom instructions.

---

## GenAI Project Structure Instructions

### Activation

Apply these rules when:
- Creating a new project that uses LLM/GenAI capabilities
- Refactoring existing GenAI code
- Adding LLM functionality to an existing project
- Reviewing GenAI project structure

### Directory Structure

Use this structure for all GenAI projects:

```
{project_name}/
├── config/
│   ├── __init__.py
│   ├── model_config.yaml
│   ├── prompts.yaml
│   └── logging_config.yaml
├── src/
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── claude_client.py
│   │   ├── gpt_client.py
│   │   └── utils.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── templates.py
│   │   ├── few_shot.py
│   │   └── chainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── rate_limiter.py
│   │   ├── token_counter.py
│   │   ├── cache.py
│   │   └── logger.py
│   └── handlers/
│       ├── __init__.py
│       └── error_handler.py
├── data/
│   ├── cache/
│   ├── prompts/
│   ├── outputs/
│   └── embeddings/
├── examples/
├── tests/
├── requirements.txt
├── README.md
└── Dockerfile
```

### Mandatory Rules

ALWAYS extract to config/:
- Model parameters (temperature, max_tokens, model_name)
- Prompt templates
- API endpoints
- Logging settings

NEVER hardcode:
- Prompt text in source files
- Model names in code
- API keys or secrets

### Configuration Templates

model_config.yaml:
```yaml
models:
  default: claude
  
  claude:
    model_name: claude-sonnet-4-5
    max_tokens: 4096
    temperature: 0.7
    
  gpt:
    model_name: gpt-5
    max_tokens: 4096
    temperature: 0.7

rate_limits:
  requests_per_minute: 50
  tokens_per_minute: 100000

cache:
  enabled: true
  ttl_seconds: 3600
```

prompts.yaml:
```yaml
system_prompts:
  default: |
    You are a helpful assistant.
    
  analyst: |
    You are a data analyst. 
    Provide structured, factual responses.

templates:
  summarize: |
    Summarize the following text:
    {text}
    
  extract: |
    Extract {entity_type} from:
    {text}
```

### Required Components

1. Base LLM Client (src/llm/base.py):
```python
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

class BaseLLMClient(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.7)
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str:
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass
```

2. Rate Limiter (src/utils/rate_limiter.py):
```python
import time
from collections import deque
from threading import Lock

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.timestamps = deque()
        self.lock = Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            while self.timestamps and self.timestamps[0] < now - 60:
                self.timestamps.popleft()
            
            if len(self.timestamps) >= self.rpm:
                sleep_time = 60 - (now - self.timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.timestamps.append(time.time())
```

3. Token Counter (src/utils/token_counter.py):
```python
import tiktoken
from typing import Optional

class TokenCounter:
    def __init__(self, model: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(model)
    
    def count(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def truncate(self, text: str, max_tokens: int) -> str:
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens])
```

4. Cache (src/utils/cache.py):
```python
import hashlib
import json
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timedelta

class ResponseCache:
    def __init__(self, cache_dir: str, ttl_seconds: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def _hash_key(self, prompt: str, model: str) -> str:
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, prompt: str, model: str) -> Optional[str]:
        key = self._hash_key(prompt, model)
        path = self.cache_dir / f"{key}.json"
        
        if not path.exists():
            return None
        
        data = json.loads(path.read_text())
        cached_time = datetime.fromisoformat(data["timestamp"])
        
        if datetime.now() - cached_time > self.ttl:
            path.unlink()
            return None
        
        return data["response"]
    
    def set(self, prompt: str, model: str, response: str):
        key = self._hash_key(prompt, model)
        path = self.cache_dir / f"{key}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "response": response
        }
        path.write_text(json.dumps(data))
```

5. Error Handler (src/handlers/error_handler.py):
```python
import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple

logger = logging.getLogger(__name__)

def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
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
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator
```

### Required Tests

Every GenAI project must include tests for core utilities. Create tests in `tests/` directory.

Structure:
```
tests/
├── __init__.py
├── test_rate_limiter.py
├── test_token_counter.py
├── test_cache.py
└── test_error_handler.py
```

test_rate_limiter.py:
```python
import time
import pytest
from src.utils import RateLimiter


def test_rate_limiter_allows_requests_under_limit():
    limiter = RateLimiter(requests_per_minute=10)
    for _ in range(5):
        limiter.acquire()
    assert limiter.get_current_usage()["requests_used"] == 5


def test_rate_limiter_blocks_when_limit_exceeded():
    limiter = RateLimiter(requests_per_minute=2)
    limiter.acquire()
    limiter.acquire()
    
    start = time.time()
    limiter.acquire()
    elapsed = time.time() - start
    
    assert elapsed >= 0.5  # Should have waited


def test_rate_limiter_resets_after_window():
    limiter = RateLimiter(requests_per_minute=60)
    limiter.acquire()
    assert limiter.get_current_usage()["requests_used"] == 1
```

test_token_counter.py:
```python
import pytest
from src.utils import TokenCounter


def test_count_empty_string():
    counter = TokenCounter()
    assert counter.count("") == 0


def test_count_simple_text():
    counter = TokenCounter()
    count = counter.count("Hello world")
    assert count > 0
    assert isinstance(count, int)


def test_truncate_under_limit():
    counter = TokenCounter()
    text = "Short text"
    result = counter.truncate(text, max_tokens=100)
    assert result == text


def test_truncate_over_limit():
    counter = TokenCounter()
    text = "This is a longer text that should be truncated"
    result = counter.truncate(text, max_tokens=5)
    assert counter.count(result) <= 5


def test_split_by_tokens():
    counter = TokenCounter()
    text = "One two three four five six seven eight nine ten"
    chunks = counter.split_by_tokens(text, chunk_size=3)
    assert len(chunks) > 1
    for chunk in chunks:
        assert counter.count(chunk) <= 3
```

test_cache.py:
```python
import pytest
import tempfile
import time
from pathlib import Path
from src.utils import ResponseCache


@pytest.fixture
def cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ResponseCache(cache_dir=tmpdir, ttl_seconds=2)


def test_cache_set_and_get(cache):
    cache.set("prompt", "model", "response")
    result = cache.get("prompt", "model")
    assert result == "response"


def test_cache_miss(cache):
    result = cache.get("nonexistent", "model")
    assert result is None


def test_cache_expiration(cache):
    cache.set("prompt", "model", "response")
    time.sleep(3)
    result = cache.get("prompt", "model")
    assert result is None


def test_cache_different_models(cache):
    cache.set("prompt", "model_a", "response_a")
    cache.set("prompt", "model_b", "response_b")
    
    assert cache.get("prompt", "model_a") == "response_a"
    assert cache.get("prompt", "model_b") == "response_b"


def test_cache_clear(cache):
    cache.set("prompt1", "model", "response1")
    cache.set("prompt2", "model", "response2")
    
    count = cache.clear()
    assert count == 2
    assert cache.get("prompt1", "model") is None
```

test_error_handler.py:
```python
import pytest
from src.handlers import retry_on_error, APIError, RateLimitError


def test_retry_succeeds_first_attempt():
    call_count = 0
    
    @retry_on_error(max_retries=3)
    def always_works():
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = always_works()
    assert result == "success"
    assert call_count == 1


def test_retry_succeeds_after_failures():
    call_count = 0
    
    @retry_on_error(max_retries=3, delay=0.1)
    def fails_twice():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("temporary error")
        return "success"
    
    result = fails_twice()
    assert result == "success"
    assert call_count == 3


def test_retry_exhausted():
    @retry_on_error(max_retries=2, delay=0.1)
    def always_fails():
        raise ValueError("persistent error")
    
    with pytest.raises(ValueError):
        always_fails()


def test_retry_specific_exceptions():
    @retry_on_error(max_retries=2, delay=0.1, exceptions=(TypeError,))
    def raises_value_error():
        raise ValueError("not caught")
    
    with pytest.raises(ValueError):
        raises_value_error()
```

Add to requirements.txt for testing:
```
pytest>=8.0.0
pytest-cov>=4.0.0
```

Run tests command:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Validation Checklist

Before completing any GenAI project task, verify:
- Prompts are externalized to YAML or separate files
- Model parameters are in config/
- Base class exists for LLM clients
- Rate limiting is implemented
- Token counting is in place
- Caching is configured
- Error handling includes retry logic
- Secrets use environment variables or secrets manager
- Request/response logging is enabled
- Tests exist for all utilities (rate limiter, cache, token counter, error handler)
- Tests pass before deployment

### Antipatterns to Flag

When reviewing code, flag these patterns:

WRONG:
```python
response = client.complete("Summarize this: " + text)
```
CORRECT:
```python
prompt = templates.get("summarize").format(text=text)
response = client.complete(prompt)
```

WRONG:
```python
client = OpenAI(model="gpt-4")
```
CORRECT:
```python
client = LLMClient(config["models"]["default"])
```

WRONG:
```python
response = api.call(prompt)
```
CORRECT:
```python
@retry_on_error(max_retries=3)
def call_api(prompt):
    return api.call(prompt)
```

WRONG:
```python
for item in items:
    process(item)
```
CORRECT:
```python
rate_limiter = RateLimiter(rpm=50)
for item in items:
    rate_limiter.acquire()
    process(item)
```
