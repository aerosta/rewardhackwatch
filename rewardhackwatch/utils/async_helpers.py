"""Async utility functions."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")


async def run_in_thread(func: Callable[..., T], *args) -> T:
    """Run sync function in thread pool."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, func, *args)


async def gather_with_limit(coros: list, limit: int = 5) -> list[Any]:
    """Run coroutines with concurrency limit."""
    semaphore = asyncio.Semaphore(limit)

    async def limited(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*[limited(c) for c in coros])


async def timeout_with_default(coro, timeout: float, default: Any = None) -> Any:
    """Run coroutine with timeout, returning default on timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return default


async def run_with_retry(
    coro_factory: Callable[[], Any], max_attempts: int = 3, delay: float = 1.0
) -> Any:
    """Run async coroutine with retry logic."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay * (2**attempt))
    raise last_error


class AsyncBatchProcessor:
    """Process items in batches asynchronously."""

    def __init__(self, batch_size: int = 10, concurrency: int = 5):
        self.batch_size = batch_size
        self.concurrency = concurrency

    async def process(self, items: list[Any], processor: Callable[[Any], Any]) -> list[Any]:
        """Process items in batches."""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            batch_results = await gather_with_limit(
                [run_in_thread(processor, item) for item in batch], limit=self.concurrency
            )
            results.extend(batch_results)
        return results
