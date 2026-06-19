"""A persistent, always-on activity event stream.

This is distinct from :class:`~parcle.sync.watch.WatchEvent`, which is a transient
report of what the *watcher* just did (inventory deltas, collections) and is only
delivered to an in-memory callback. The **event stream** here is a durable audit
log of lifecycle and manual operations — the daemon starting/stopping, a single
skill ``push`` / ``pull``, a single conversation read — together with their
failures. Each entry is appended as one JSON line to ``<PARCLE_HOME>/events.jsonl``
so the history survives across processes.

Recording never raises: a failure to write the log must not break the operation
being logged.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Event", "EventLog"]

# Categories.
CAT_DAEMON = "daemon"
CAT_SKILL = "skill"
CAT_CONVERSATION = "conversation"

# Statuses.
STATUS_OK = "ok"
STATUS_ERROR = "error"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Event:
    """One entry in the activity stream."""

    timestamp: str
    category: str  # "daemon" | "skill" | "conversation"
    action: str  # daemon: start/stop; skill: push/pull; conversation: get_conversation/get_turn
    status: str = STATUS_OK  # "ok" | "error"
    name: Optional[str] = None  # skill key or session id
    agent_type: Optional[str] = None
    detail: Optional[str] = None  # extra context (e.g. "12 turn(s)", "seq=3")
    error: Optional[str] = None  # error message when status == "error"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "category": self.category,
            "action": self.action,
            "status": self.status,
        }
        for key in ("name", "agent_type", "detail", "error"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(
            timestamp=str(data.get("timestamp") or ""),
            category=str(data.get("category") or ""),
            action=str(data.get("action") or ""),
            status=str(data.get("status") or STATUS_OK),
            name=data.get("name"),
            agent_type=data.get("agent_type"),
            detail=data.get("detail"),
            error=data.get("error"),
        )

    def __str__(self) -> str:
        where = f" [{self.agent_type}]" if self.agent_type else ""
        what = f" {self.name}" if self.name else ""
        tail = f" {self.detail}" if self.detail else ""
        err = f" — {self.error}" if self.error else ""
        return f"{self.category} {self.action} ({self.status}):{what}{where}{tail}{err}"


class EventLog:
    """Append-only, thread-safe activity log persisted as JSONL.

    A single instance is shared by the skill library, the conversation store, and
    the daemon (see :class:`parcle.Sync`), so concurrent writes from the daemon
    thread and the caller's thread are serialized by one lock. ``path=None`` makes
    it an in-memory no-op store (it still fires ``on_event``), handy for tests.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        on_event: Optional[Callable[[Event], None]] = None,
    ) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._on_event = on_event or (lambda event: None)

    def record(
        self,
        category: str,
        action: str,
        *,
        status: str = STATUS_OK,
        name: Optional[str] = None,
        agent_type: Optional[str] = None,
        detail: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Event:
        """Append one event and return it. Never raises on a write failure."""
        event = Event(
            timestamp=_now_iso(),
            category=category,
            action=action,
            status=status,
            name=name,
            agent_type=agent_type,
            detail=detail,
            error=error,
        )
        if self.path is not None:
            with self._lock:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(event.to_jsonl() + "\n")
                except OSError:
                    pass
        try:
            self._on_event(event)
        except Exception:  # noqa: BLE001 - a bad callback must not break logging
            pass
        return event

    def read(
        self,
        *,
        limit: Optional[int] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Event]:
        """Read recorded events, oldest first.

        ``category`` / ``status`` filter the result; ``limit`` keeps only the most
        recent ``limit`` (after filtering). Returns an empty list if nothing has
        been recorded.
        """
        events: List[Event] = []
        if self.path is None or not self.path.is_file():
            return events
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(Event.from_dict(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        if category is not None:
            events = [e for e in events if e.category == category]
        if status is not None:
            events = [e for e in events if e.status == status]
        if limit is not None and limit >= 0:
            events = events[-limit:] if limit > 0 else []
        return events
