"""
Caching utilities for SnapScholar using the diskcache library.

This module provides a simple key-value store for caching expensive
operations like transcription fetching and AI-powered summarization.
It helps to avoid re-processing the same video multiple times.

To install the required library:
pip install diskcache
"""
from typing import Any, Optional
from diskcache import Cache
from config.settings import settings

# Initialize a single cache instance for the application.
# The cache directory is managed by the diskcache library.
CACHE_DIR = settings.TEMP_DIR / "app_cache"
cache = Cache(str(CACHE_DIR))


def get_from_cache(key: str) -> Optional[Any]:
    """
    Retrieve an item from the cache.

    Args:
        key: The unique key for the cached item.

    Returns:
        The cached item, or None if the key is not found.
    """
    if key in cache:
        print(f"  💿 Loaded from cache: '{key}'")
        return cache.get(key)
    return None


def save_to_cache(key: str, value: Any):
    """
    Save an item to the cache.

    Args:
        key: The unique key for the item.
        value: The Python object to cache.
    """
    cache.set(key, value)
    print(f"  💾 Saved to cache: '{key}'")


def clear_cache():
    """
    Clear the entire application cache.
    """
    count = len(cache)
    cache.clear()
    print(f"  🗑️  Cleared {count} items from the cache at {CACHE_DIR}")