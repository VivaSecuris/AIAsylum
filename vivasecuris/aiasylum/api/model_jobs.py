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
from typing import List, Optional

_slot: Optional[asyncio.Semaphore] = None
_held_by: Optional[str] = None
_waiting: List[str] = []


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
    global _held_by
    _waiting.append(owner)
    try:
        async with model_slot():
            if owner in _waiting:
                _waiting.remove(owner)
            _held_by = owner
            try:
                yield
            finally:
                _held_by = None
    finally:
        if owner in _waiting:
            _waiting.remove(owner)


def slot_status() -> dict:
    """Who holds the slot and who is queued, for progress events."""
    return {"held_by": _held_by, "waiting": list(_waiting)}
