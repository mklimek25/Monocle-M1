"""
Event Bus / Message Broker (lightweight, in-process).

This file is NEW in the "changed files" snapshot to demonstrate Framework Improvement #2.

Why:
- Replace direct callbacks between components (tight coupling) with publish/subscribe (looser coupling).
- Allow multiple subscribers to react to the same event without changing the publisher.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List


class EventType(Enum):
    """
    Keep these event types focused on *facts* that happened, not commands.
    """

    PROCESSING_COMPLETE = "processing_complete"  # FrameProcessor -> DataManager (+ optional others)
    IMAGE_READY = "image_ready"  # FrameProcessor -> GUI
    SCAN_ERROR = "scan_error"  # FrameProcessor -> GUI


@dataclass(frozen=True)
class Event:
    """
    A single published event.
    - event_type: what happened
    - source: who published it (string is fine; could be an enum later)
    - data: payload (dict/image/str/etc.)
    - timestamp: when it was published
    """

    event_type: EventType
    source: str
    data: Any
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """
    Simple synchronous event bus.

    Notes:
    - Handlers are called in the publisher's thread. In your system, that means
      FrameProcessor's background thread will invoke subscribers.
    - For Tkinter UI updates, subscribers should use `root.after(...)` to hop
      back to the main thread.
    - We swallow handler exceptions so one bad subscriber doesn't kill the publisher thread.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        for handler in self._subscribers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                # IMPORTANT: never let the publisher thread die due to a subscriber bug
                print(f"[EventBus] handler error for {event.event_type.value}: {e!r}")
