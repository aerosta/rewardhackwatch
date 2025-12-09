"""Caching utilities for analysis results."""

import hashlib
import json
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    value: Any
    created_at: float
    expires_at: Optional[float] = None
    hits: int = 0


class SimpleCache:
    """Simple in-memory cache."""

    def __init__(self, default_ttl: Optional[float] = None, max_size: int = 1000):
        self._cache: dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        entry = self._cache.get(key)
        if entry is None:
            return None

        # Check expiration
        if entry.expires_at and time.time() > entry.expires_at:
            del self._cache[key]
            return None

        entry.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        # Evict if needed
        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        expires_at = None
        if ttl or self.default_ttl:
            expires_at = time.time() + (ttl or self.default_ttl)

        self._cache[key] = CacheEntry(value=value, created_at=time.time(), expires_at=expires_at)

    def delete(self, key: str) -> bool:
        """Delete from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()

    def _evict_oldest(self):
        """Evict oldest entry."""
        if not self._cache:
            return

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]

    @property
    def size(self) -> int:
        """Get number of entries."""
        return len(self._cache)


def make_cache_key(obj: Any) -> str:
    """Create cache key from object."""
    content = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.md5(content.encode()).hexdigest()


def cached(cache: SimpleCache, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = make_cache_key((args, kwargs))

            # Check cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        wrapper.cache = cache
        return wrapper

    return decorator


# Global cache instance
_global_cache = SimpleCache(default_ttl=3600)  # 1 hour default


def get_global_cache() -> SimpleCache:
    """Get the global cache instance."""
    return _global_cache
