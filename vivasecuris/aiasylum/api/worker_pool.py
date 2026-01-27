"""Worker pool manager for controlling concurrent test run execution."""

import asyncio
import logging
from typing import Optional
from config.settings import settings

logger = logging.getLogger(__name__)


class WorkerPool:
    """Manages a semaphore-based worker pool to limit concurrent test run execution."""
    
    _instance: Optional['WorkerPool'] = None
    _semaphore: Optional[asyncio.Semaphore] = None
    
    def __init__(self):
        """Initialize the worker pool with semaphore."""
        max_workers = settings.max_concurrent_workers
        self._semaphore = asyncio.Semaphore(max_workers)
        logger.info(f"🔧 Worker pool initialized with {max_workers} max concurrent workers")
    
    @classmethod
    def get_instance(cls) -> 'WorkerPool':
        """Get or create the singleton worker pool instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def acquire(self, test_run_id: int) -> None:
        """Acquire a worker slot for a test run."""
        logger.debug(f"⏳ Waiting for worker slot for test run {test_run_id} (current: {self._semaphore._value}/{settings.max_concurrent_workers})")
        await self._semaphore.acquire()
        logger.info(f"✅ Acquired worker slot for test run {test_run_id} (available: {self._semaphore._value}/{settings.max_concurrent_workers})")
    
    def release(self, test_run_id: int) -> None:
        """Release a worker slot after test run completes."""
        self._semaphore.release()
        logger.info(f"🔓 Released worker slot for test run {test_run_id} (available: {self._semaphore._value}/{settings.max_concurrent_workers})")
    
    async def run_with_limit(self, test_run_id: int, coro):
        """Run a coroutine with worker pool limit."""
        await self.acquire(test_run_id)
        try:
            return await coro
        finally:
            self.release(test_run_id)
    
    @property
    def available_workers(self) -> int:
        """Get the number of available worker slots."""
        return self._semaphore._value
    
    @property
    def max_workers(self) -> int:
        """Get the maximum number of workers."""
        return settings.max_concurrent_workers


# Global worker pool instance
worker_pool = WorkerPool.get_instance()
