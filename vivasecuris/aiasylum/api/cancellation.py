"""Cancellation manager for test runs."""

import asyncio
import logging
from typing import Set, Dict, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class CancellationManager:
    """Manages cancellation flags and asyncio tasks for test runs."""
    
    def __init__(self):
        # Store cancelled test run IDs
        self._cancelled: Set[int] = set()
        # Store running asyncio tasks by test_run_id
        self._tasks: Dict[int, asyncio.Task] = {}
        # Lock for thread safety
        self._lock = Lock()
    
    def register_task(self, test_run_id: int, task: asyncio.Task):
        """Register an asyncio task for a test run."""
        with self._lock:
            self._tasks[test_run_id] = task
            logger.info(f"📝 Registered task for test run {test_run_id}")
            print(f"[CANCELLATION] Registered task for test run {test_run_id}")
    
    def unregister_task(self, test_run_id: int):
        """Unregister an asyncio task for a test run."""
        with self._lock:
            if test_run_id in self._tasks:
                del self._tasks[test_run_id]
                logger.debug(f"Unregistered task for test run {test_run_id}")
    
    def cancel(self, test_run_id: int) -> bool:
        """Mark a test run as cancelled and cancel its asyncio task if running."""
        with self._lock:
            if test_run_id not in self._cancelled:
                self._cancelled.add(test_run_id)
                logger.info(f"🛑 Test run {test_run_id} marked for cancellation")
                print(f"[CANCELLATION] Test run {test_run_id} added to cancelled set. Current cancelled: {self._cancelled}")
            
            # Cancel the asyncio task if it exists
            if test_run_id in self._tasks:
                task = self._tasks[test_run_id]
                if not task.done():
                    logger.info(f"🛑 Cancelling asyncio task for test run {test_run_id}")
                    print(f"[CANCELLATION] Cancelling asyncio task for test run {test_run_id}")
                    task.cancel()
                    return True
                else:
                    # Task already done, just remove it
                    del self._tasks[test_run_id]
                    logger.info(f"Task for test run {test_run_id} already completed")
            
            return test_run_id in self._cancelled
    
    def is_cancelled(self, test_run_id: int) -> bool:
        """Check if a test run is cancelled."""
        with self._lock:
            is_cancelled = test_run_id in self._cancelled
            if is_cancelled:
                logger.debug(f"Test run {test_run_id} is cancelled")
            return is_cancelled

    def has_active_task(self, test_run_id: int) -> bool:
        """A cancelled worker still owns its row until cleanup has finished."""
        with self._lock:
            task = self._tasks.get(test_run_id)
            return task is not None and not task.done()
    
    def clear(self, test_run_id: int):
        """Clear cancellation flag for a test run."""
        with self._lock:
            self._cancelled.discard(test_run_id)
            # Also clean up task if it exists
            if test_run_id in self._tasks:
                del self._tasks[test_run_id]
            logger.debug(f"Cleared cancellation flag for test run {test_run_id}")


# Global instance
cancellation_manager = CancellationManager()
