"""Tests for caching utilities."""

import time


class TestSimpleCache:
    """Tests for simple cache."""

    def test_set_and_get(self):
        """Test basic set and get."""
        from rewardhackwatch.utils.caching import SimpleCache

        cache = SimpleCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        """Test getting missing key."""
        from rewardhackwatch.utils.caching import SimpleCache

        cache = SimpleCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        from rewardhackwatch.utils.caching import SimpleCache

        cache = SimpleCache()
        cache.set("key", "value", ttl=0.1)
        assert cache.get("key") == "value"

        time.sleep(0.2)
        assert cache.get("key") is None

    def test_max_size_eviction(self):
        """Test max size eviction."""
        from rewardhackwatch.utils.caching import SimpleCache

        cache = SimpleCache(max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")

        assert cache.size == 3
        # Oldest key should be evicted
        assert cache.get("key1") is None

    def test_delete(self):
        """Test delete."""
        from rewardhackwatch.utils.caching import SimpleCache

        cache = SimpleCache()
        cache.set("key", "value")
        assert cache.delete("key")
        assert cache.get("key") is None
        assert not cache.delete("key")

    def test_clear(self):
        """Test clear."""
        from rewardhackwatch.utils.caching import SimpleCache

        cache = SimpleCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.size == 0


class TestCachedDecorator:
    """Tests for cached decorator."""

    def test_cached_function(self):
        """Test cached function."""
        from rewardhackwatch.utils.caching import SimpleCache, cached

        cache = SimpleCache()
        call_count = 0

        @cached(cache)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once due to caching


class TestMakeCacheKey:
    """Tests for cache key generation."""

    def test_consistent_keys(self):
        """Test that same input produces same key."""
        from rewardhackwatch.utils.caching import make_cache_key

        obj = {"a": 1, "b": 2}
        key1 = make_cache_key(obj)
        key2 = make_cache_key(obj)
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        """Test that different inputs produce different keys."""
        from rewardhackwatch.utils.caching import make_cache_key

        key1 = make_cache_key({"a": 1})
        key2 = make_cache_key({"a": 2})
        assert key1 != key2
