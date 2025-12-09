"""Retry utilities for API calls."""

import functools
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error: Optional[Exception] = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_error

        return wrapper

    return decorator


class RetryContext:
    """Context manager for retry logic."""

    def __init__(self, max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.attempt = 0
        self.last_error: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.last_error = exc_val
            self.attempt += 1
            if self.attempt < self.max_attempts:
                time.sleep(self.delay * (self.backoff ** (self.attempt - 1)))
                return True  # Suppress exception, retry
        return False

    @property
    def should_retry(self) -> bool:
        """Check if we should continue retrying."""
        return self.attempt < self.max_attempts


def retry_with_result(
    func: Callable[..., T],
    validator: Callable[[T], bool],
    max_attempts: int = 3,
    delay: float = 1.0,
    *args,
    **kwargs,
) -> Optional[T]:
    """Retry function until result passes validation.

    Args:
        func: Function to call
        validator: Function to validate result
        max_attempts: Maximum attempts
        delay: Delay between attempts
    """
    for attempt in range(max_attempts):
        result = func(*args, **kwargs)
        if validator(result):
            return result
        if attempt < max_attempts - 1:
            time.sleep(delay)
    return None
