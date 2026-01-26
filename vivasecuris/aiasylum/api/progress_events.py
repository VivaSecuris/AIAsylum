"""Progress event manager for real-time test run monitoring."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, AsyncIterator
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProgressEventManager:
    """Manages progress events for test runs using Server-Sent Events."""
    
    def __init__(self):
        # Store event queues for each test run
        self._event_queues: Dict[int, asyncio.Queue] = {}
        # Store active connections
        self._connections: Dict[int, List[asyncio.Queue]] = defaultdict(list)
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def subscribe(self, test_run_id: int) -> asyncio.Queue:
        """Subscribe to progress events for a test run."""
        async with self._lock:
            if test_run_id not in self._event_queues:
                self._event_queues[test_run_id] = asyncio.Queue()
            
            # Create a new queue for this connection
            connection_queue = asyncio.Queue()
            self._connections[test_run_id].append(connection_queue)
            logger.info(f"New subscription for test run {test_run_id} (total: {len(self._connections[test_run_id])})")
            return connection_queue
    
    async def unsubscribe(self, test_run_id: int, queue: asyncio.Queue):
        """Unsubscribe from progress events."""
        async with self._lock:
            if test_run_id in self._connections:
                try:
                    self._connections[test_run_id].remove(queue)
                    logger.info(f"Unsubscribed from test run {test_run_id} (remaining: {len(self._connections[test_run_id])})")
                except ValueError:
                    pass
                
                # Clean up if no more connections
                if not self._connections[test_run_id] and test_run_id in self._event_queues:
                    del self._event_queues[test_run_id]
    
    async def emit_event(
        self,
        test_run_id: int,
        event_type: str,
        data: Dict,
        message: Optional[str] = None
    ):
        """Emit a progress event for a test run."""
        event = {
            "test_run_id": test_run_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }
        if message:
            event["message"] = message
        
        async with self._lock:
            # Broadcast to all connections for this test run
            if test_run_id in self._connections:
                for queue in self._connections[test_run_id]:
                    try:
                        await queue.put(event)
                    except Exception as e:
                        logger.error(f"Error emitting event to queue: {e}")
    
    async def stream_events(self, test_run_id: int) -> AsyncIterator[str]:
        """Stream events as Server-Sent Events."""
        queue = await self.subscribe(test_run_id)
        
        try:
            # Send initial connection event
            initial_event = {
                'event_type': 'connected',
                'test_run_id': test_run_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': {}
            }
            logger.info(f"Sending initial connection event for test run {test_run_id}")
            yield f"data: {json.dumps(initial_event)}\n\n"
            
            while True:
                try:
                    # Wait for event with timeout to allow periodic heartbeats
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    logger.debug(f"Emitting event for test run {test_run_id}: {event.get('event_type')}")
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    heartbeat = {
                        'event_type': 'heartbeat',
                        'test_run_id': test_run_id,
                        'timestamp': datetime.utcnow().isoformat(),
                        'data': {}
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
                except Exception as e:
                    logger.error(f"Error in event stream for test run {test_run_id}: {e}", exc_info=True)
                    break
        finally:
            logger.info(f"Closing event stream for test run {test_run_id}")
            await self.unsubscribe(test_run_id, queue)


# Global instance
progress_event_manager = ProgressEventManager()
