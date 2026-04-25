from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
import time
from enum import Enum

class EventType(Enum):
    METRIC = "metric"
    LOG = "log"
    TRACE = "trace"
    COMPILER_START = "compiler_start"
    COMPILER_END = "compiler_end"
    PASS_START = "pass_start"
    PASS_END = "pass_end"
    RUNTIME_STEP = "runtime_step"
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    SHAPE_INFERENCE = "shape_inference"
    PRUNING_EVENT = "pruning_event"
    ACTOR_STEP = "actor_step"
    LEARNER_STEP = "learner_step"
    CUSTOM = "custom"


@dataclass
class Event:
    """A unified event object for the entire observability system."""
    type: EventType
    name: str
    value: Any = None
    step: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EventEmitter:
    """Central hub for emitting and subscribing to events."""
    
    def __init__(self):
        self._subscribers: List[Callable[[Event], None]] = []

    def subscribe(self, callback: Callable[[Event], None]):
        """Register a callback to handle emitted events."""
        self._subscribers.append(callback)

    def emit(self, event: Event):
        """Broadcast an event to all subscribers."""
        for subscriber in self._subscribers:
            subscriber(event)

    def emit_metric(self, name: str, value: Any, step: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
        self.emit(Event(
            type=EventType.METRIC,
            name=name,
            value=value,
            step=step,
            metadata=metadata or {}
        ))

    def emit_trace(self, name: str, type: EventType, duration: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        self.emit(Event(
            type=type,
            name=name,
            duration=duration,
            metadata=metadata or {}
        ))

    from contextlib import contextmanager
    @contextmanager
    def trace_pass(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracing a compiler or runtime pass."""
        start = time.time()
        self.emit(Event(type=EventType.PASS_START, name=name, metadata=metadata or {}))
        try:
            yield
        finally:
            self.emit(Event(
                type=EventType.PASS_END, 
                name=name, 
                duration=time.time() - start,
                metadata=metadata or {}
            ))


# Global emitter
_GLOBAL_EMITTER = EventEmitter()

def get_emitter() -> EventEmitter:
    return _GLOBAL_EMITTER
