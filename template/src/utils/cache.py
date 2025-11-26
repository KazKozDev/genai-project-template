import hashlib
import json
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timedelta


class ResponseCache:
    """File-based cache for LLM responses."""
    
    def __init__(self, cache_dir: str, ttl_seconds: int = 3600):
        """Initialize cache.
        
        Args:
            cache_dir: Directory for cache files
            ttl_seconds: Time-to-live for cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def _hash_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate hash key for cache lookup."""
        content = json.dumps({
            "prompt": prompt,
            "model": model,
            **kwargs
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, **kwargs) -> Optional[str]:
        """Retrieve cached response if available and valid.
        
        Args:
            prompt: Original prompt
            model: Model name
            **kwargs: Additional parameters that affect response
            
        Returns:
            Cached response or None
        """
        key = self._hash_key(prompt, model, **kwargs)
        path = self.cache_dir / f"{key}.json"
        
        if not path.exists():
            return None
        
        try:
            data = json.loads(path.read_text())
            cached_time = datetime.fromisoformat(data["timestamp"])
            
            if datetime.now() - cached_time > self.ttl:
                path.unlink()
                return None
            
            return data["response"]
        except (json.JSONDecodeError, KeyError):
            path.unlink()
            return None
    
    def set(self, prompt: str, model: str, response: str, **kwargs) -> None:
        """Store response in cache.
        
        Args:
            prompt: Original prompt
            model: Model name
            response: Response to cache
            **kwargs: Additional parameters
        """
        key = self._hash_key(prompt, model, **kwargs)
        path = self.cache_dir / f"{key}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "model": model,
            "response": response,
            "metadata": kwargs
        }
        path.write_text(json.dumps(data, indent=2))
    
    def clear(self) -> int:
        """Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count
    
    def clear_expired(self) -> int:
        """Clear only expired cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = 0
        now = datetime.now()
        
        for path in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                cached_time = datetime.fromisoformat(data["timestamp"])
                
                if now - cached_time > self.ttl:
                    path.unlink()
                    count += 1
            except (json.JSONDecodeError, KeyError):
                path.unlink()
                count += 1
        
        return count
    
    def stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total = 0
        valid = 0
        expired = 0
        size_bytes = 0
        now = datetime.now()
        
        for path in self.cache_dir.glob("*.json"):
            total += 1
            size_bytes += path.stat().st_size
            
            try:
                data = json.loads(path.read_text())
                cached_time = datetime.fromisoformat(data["timestamp"])
                
                if now - cached_time > self.ttl:
                    expired += 1
                else:
                    valid += 1
            except (json.JSONDecodeError, KeyError):
                expired += 1
        
        return {
            "total_entries": total,
            "valid_entries": valid,
            "expired_entries": expired,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2)
        }
