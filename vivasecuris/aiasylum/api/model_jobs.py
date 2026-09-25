"""The single slot for model-heavy jobs.

Interpretability runs and weight-surgery runs both load multi-billion-parameter
models into the same unified memory. Giving each router its own semaphore would
let one of each run concurrently, which is the contention MODEL_MODIFICATION.md
section 7 measured: a 16-token generation exceeding seven minutes while an
Ollama model held 5.5 GB of the same 24 GB, with nothing erroring to say so.

Deliberately not `api/worker_pool.py`, whose five slots are shared with test
runs -- a model-loading job would hold one of them for its entire duration.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import List, Optional
from pathlib import Path
import os


def try_process_lock():
    """Share the GPU slot with standalone validation processes on this host."""
    import fcntl
    path = Path(os.environ.get("AIASYLUM_MODEL_LOCK", "runs/.model-job.lock"))
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle

_slot: Optional[asyncio.Semaphore] = None
_held_by: Optional[str] = None
_waiting: List[str] = []
_context_lease: ContextVar[Optional[object]] = ContextVar("model_job_lease", default=None)
_active_lease: Optional[object] = None


def held_in_context() -> bool:
    """True only while this context's inherited lease is still the active one."""
    return _active_lease is not None and _context_lease.get() is _active_lease


def model_slot() -> asyncio.Semaphore:
    """The process-wide slot, created lazily so it binds to the running loop."""
    global _slot
    if _slot is None:
        _slot = asyncio.Semaphore(1)
    return _slot


@asynccontextmanager
async def hold(owner: str):
    """Take the slot, recording the holder and the queue behind it.

    A job queued here is indistinguishable from a hung one unless the wait is
    reported, which is the whole subject of `weights/progress.py`'s docstring.
    `slot_status()` is what lets a route say "waiting for interp run 4" instead
    of leaving a run at `pending` with no explanation.
    """
    global _held_by, _active_lease
    _waiting.append(owner)
    try:
        async with model_slot():
            process_lock = None
            context_token = None
            try:
                while process_lock is None:
                    process_lock = try_process_lock()
                    if process_lock is None:
                        await asyncio.sleep(0.25)
                if owner in _waiting:
                    _waiting.remove(owner)
                _held_by = owner
                _active_lease = object()
                context_token = _context_lease.set(_active_lease)
                yield
            finally:
                _active_lease = None
                if context_token is not None:
                    try:
                        _context_lease.reset(context_token)
                    except ValueError:
                        # An async generator may be closed by a finalizer in
                        # another context. Its inherited marker is now expired.
                        pass
                _held_by = None
                if process_lock is not None:
                    process_lock.close()
    finally:
        if owner in _waiting:
            _waiting.remove(owner)


def slot_status() -> dict:
    """Who holds the slot and who is queued, for progress events."""
    return {"held_by": _held_by, "waiting": list(_waiting)}
