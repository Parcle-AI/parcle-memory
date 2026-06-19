"""Real-time watching: track skills and keep conversations current.

A :class:`Watcher` watches every detected agent's skill and transcript
directories and reacts to changes:

* **Skills** — the per-agent inventory is tracked so additions and removals are
  reported live (a real-time "who has what"). The watcher never writes to the
  library: capturing a skill into it (``push``) and installing one onto an agent
  (``pull``) are both manual.
* **Conversations** — changed transcripts are collected incrementally into the
  local store.

The watcher can run in the foreground (:meth:`Watcher.run`, blocking) or in a
background daemon thread (:meth:`Watcher.start` / :meth:`Watcher.stop`), which is
what :class:`parcle.Sync` uses. A thread-safe :class:`SyncState` tracks, with a
lock, whether each side is currently parsing (so readiness can be queried live)
and a snapshot of counts that is persisted to local storage and inherited on the
next run.

Watching is always real-time and event-driven via the ``watchfiles`` package (a
dependency of Parcle), backed by a periodic safety reconcile so a missed event is
still picked up. If the watch loop ever errors it is re-established after a short
backoff, so the daemon keeps watching rather than dying silently.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Set

from .agents import detect_agents
from .conversations import discover_sessions

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from . import Sync

__all__ = ["Watcher", "WatchEvent", "SyncState"]


# Backoff before re-establishing the watch after an unexpected loop failure.
_WATCH_RETRY_BACKOFF = 2.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WatchEvent:
    """Something the watcher did in response to a change."""

    kind: str  # "skill" | "conversation"
    action: str  # skill: agent_added/agent_removed; conversation: collected
    name: str  # skill key or session id
    agent_type: Optional[str] = None
    detail: str = ""

    def __str__(self) -> str:
        where = f" [{self.agent_type}]" if self.agent_type else ""
        tail = f" {self.detail}" if self.detail else ""
        return f"{self.kind} {self.action}: {self.name}{where}{tail}"


class SyncState:
    """Thread-safe sync state: per-side "busy" tracking + a persisted snapshot.

    "Busy" means the daemon is currently parsing that side (skills or
    conversations). Readiness is simply "not busy". The snapshot (counts and the
    last-event timestamp) is written to ``path`` after each cycle and loaded back
    on construction, so a fresh process inherits the previous view.
    """

    _KINDS = ("skills", "conversations")

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._active: Dict[str, int] = {kind: 0 for kind in self._KINDS}
        self._snapshot: Dict[str, Any] = {
            "skills": {},
            "conversations": {},
            "last_event_at": None,
            "last_error": None,  # most recent watch failure, if any
        }
        if path is not None:
            self._load()

    # -- busy / readiness ------------------------------------------------------

    def mark_active(self, kind: str) -> None:
        with self._lock:
            self._active[kind] = self._active.get(kind, 0) + 1

    def mark_idle(self, kind: str) -> None:
        with self._lock:
            self._active[kind] = max(0, self._active.get(kind, 0) - 1)

    @contextmanager
    def busy(self, kind: str) -> Iterator[None]:
        self.mark_active(kind)
        try:
            yield
        finally:
            self.mark_idle(kind)

    def ready(self, kind: str) -> bool:
        """True when ``kind`` is not currently being parsed."""
        with self._lock:
            return self._active.get(kind, 0) == 0

    # -- snapshot --------------------------------------------------------------

    def record_event(self) -> None:
        with self._lock:
            self._snapshot["last_event_at"] = _now_iso()

    def set_error(self, error: Optional[str]) -> None:
        with self._lock:
            self._snapshot["last_error"] = error

    def update(
        self,
        *,
        skills: Optional[Dict[str, Any]] = None,
        conversations: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            if skills is not None:
                self._snapshot["skills"] = skills
            if conversations is not None:
                self._snapshot["conversations"] = conversations

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = json.loads(json.dumps(self._snapshot))
            snap["ready"] = {
                "skills": self._active.get("skills", 0) == 0,
                "conversations": self._active.get("conversations", 0) == 0,
            }
            return snap

    # -- persistence -----------------------------------------------------------

    def persist(self) -> None:
        if self.path is None:
            return
        with self._lock:
            data = json.loads(json.dumps(self._snapshot))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        with self._lock:
            for key in ("skills", "conversations", "last_event_at"):
                if key in data:
                    self._snapshot[key] = data[key]


class Watcher:
    """Watch agents for skill and conversation changes and react to them."""

    def __init__(
        self,
        sync: "Sync",
        *,
        skills: bool = True,
        conversations: bool = True,
        on_event: Optional[Callable[[WatchEvent], None]] = None,
        state: Optional[SyncState] = None,
        reconcile_interval: float = 5.0,
    ) -> None:
        self.sync = sync
        self.skills = skills
        self.conversations = conversations
        self.on_event = on_event or (lambda event: None)
        self.state = state or SyncState()
        # Belt-and-suspenders: even with file-system events, periodically re-scan
        # so anything an event missed (atomic saves, the gap between backfill and
        # the watcher going live, network drives) is still picked up.
        self.reconcile_interval = reconcile_interval
        # Per-agent skill inventory (agent_type -> set of skill keys), kept so we
        # can report additions/removals as deltas.
        self._inventory: Dict[str, Set[str]] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop: Optional[threading.Event] = None
        self._startup: List[str] = []

    # -- roots -----------------------------------------------------------------

    def skill_roots(self) -> List[Path]:
        roots: List[Path] = []
        for agent in detect_agents():
            if self.skills and agent.skills_dir is not None and agent.skills_dir.is_dir():
                roots.append(agent.skills_dir)
        return roots

    def conversation_roots(self) -> List[Path]:
        roots: List[Path] = []
        for agent in detect_agents():
            if (
                self.conversations
                and agent.transcripts_root is not None
                and agent.transcripts_root.is_dir()
            ):
                roots.append(agent.transcripts_root)
        return roots

    def roots(self) -> List[Path]:
        return self.skill_roots() + self.conversation_roots()

    # -- reactions (event-loop-free, so they're easy to test) ------------------

    def sync_skills(self) -> List[WatchEvent]:
        """Track which skills each agent holds, reporting additions/removals.

        This never writes to the library — capturing a skill into the library is
        a manual ``push``. Watching only keeps the live per-agent inventory.
        """
        events = self._inventory_deltas()
        self._refresh_skill_state()
        return events

    def _inventory_deltas(self) -> List[WatchEvent]:
        events: List[WatchEvent] = []
        current = {
            agent_type: {s.key for s in skills}
            for agent_type, skills in self.sync.skills.inventory().items()
        }
        for agent_type, keys in current.items():
            previous = self._inventory.get(agent_type, set())
            for key in sorted(keys - previous):
                events.append(WatchEvent("skill", "agent_added", key, agent_type))
            for key in sorted(previous - keys):
                events.append(WatchEvent("skill", "agent_removed", key, agent_type))
        for agent_type in set(self._inventory) - set(current):
            for key in sorted(self._inventory[agent_type]):
                events.append(WatchEvent("skill", "agent_removed", key, agent_type))
        self._inventory = current
        return events

    def sync_conversations(self) -> List[WatchEvent]:
        """Incrementally collect any new conversation turns."""
        events: List[WatchEvent] = []
        for result in self.sync.conversations.collect_all():
            if result.new_turns:
                events.append(
                    WatchEvent(
                        "conversation",
                        "collected",
                        result.session_id,
                        result.agent_type,
                        f"+{result.new_turns} turn(s)",
                    )
                )
        self._refresh_conversation_state()
        return events

    def reconcile(self) -> List[WatchEvent]:
        """Run a full pass over both sides (skills + conversations).

        Used for the initial backfill and for the periodic safety re-scan. Cheap
        when nothing changed: ingest skips identical content and collection only
        reads bytes past each cursor.
        """
        events: List[WatchEvent] = []
        if self.skills:
            with self.state.busy("skills"):
                events.extend(self.sync_skills())
        if self.conversations:
            with self.state.busy("conversations"):
                events.extend(self.sync_conversations())
        if events:
            self.state.record_event()
        return events

    # Backfill is just the first full reconcile.
    initial_sync = reconcile

    def react(self, changed: List[Path]) -> List[WatchEvent]:
        """React to a batch of changed paths from the watcher."""
        events: List[WatchEvent] = []
        if self.skills and _any_under(changed, self.skill_roots()):
            with self.state.busy("skills"):
                events.extend(self.sync_skills())
        if self.conversations and _any_under(changed, self.conversation_roots()):
            with self.state.busy("conversations"):
                events.extend(self.sync_conversations())
        if events:
            self.state.record_event()
        return events

    # -- snapshot refresh ------------------------------------------------------

    def _refresh_skill_state(self) -> None:
        library = self.sync.skills.list()
        by_agent = {
            agent_type: [s.key for s in skills]
            for agent_type, skills in self.sync.skills.inventory().items()
        }
        # Drift between the library and each agent's copies. Purely informational:
        # ``missing_from_library`` are push candidates, ``diverged`` are skills
        # whose contents no longer match the library. Nothing is captured or
        # reconciled automatically.
        drift = self.sync.skills.drift()
        self.state.update(
            skills={
                "library": len(library),
                "by_agent": by_agent,
                "missing_from_library": drift["missing_from_library"],
                "diverged": drift["diverged"],
            }
        )

    def _refresh_conversation_state(self) -> None:
        sessions = caught_up = pending = turns = 0
        for agent in detect_agents():
            if not agent.supports_conversations:
                continue
            for session in discover_sessions(agent):
                progress = self.sync.conversations.progress(session)
                sessions += 1
                turns += progress["turn_count"]
                if progress["caught_up"]:
                    caught_up += 1
                else:
                    pending += 1
        self.state.update(
            conversations={
                "sessions": sessions,
                "caught_up": caught_up,
                "pending": pending,
                "turns": turns,
            }
        )

    # -- run loop --------------------------------------------------------------

    def run(self) -> None:
        """Watch until interrupted (blocking).

        Real-time and event-driven via ``watchfiles``, re-establishing the watch
        if it errors so it stays up.
        """
        self._startup = []
        self._serve(None)

    def start(self) -> None:
        """Start watching in a background daemon thread (non-blocking).

        Both sides are marked busy up front so readiness only flips to ready once
        the initial backfill has actually finished.
        """
        if self.is_running():
            return
        self._startup = []
        if self.skills:
            self.state.mark_active("skills")
            self._startup.append("skills")
        if self.conversations:
            self.state.mark_active("conversations")
            self._startup.append("conversations")
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, args=(self._stop,), name="parcle-sync", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """Signal the background thread to stop and wait for it to finish."""
        if self._stop is not None:
            self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._thread = None
        self._stop = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _serve(self, stop_event: Optional[threading.Event]) -> None:
        # 1. Backfill once, releasing the startup "busy" holds when it finishes.
        try:
            for event in self.initial_sync():
                self.on_event(event)
        finally:
            for kind in self._startup:
                self.state.mark_idle(kind)
            self._startup = []
        self.state.persist()

        roots = self.roots()
        if not roots:
            return  # nothing installed to watch

        try:
            from watchfiles import watch as watch_fn
        except ImportError as exc:  # pragma: no cover - watchfiles is a dependency
            raise RuntimeError(
                "Parcle's background sync requires the 'watchfiles' package "
                "(it is a dependency of Parcle). Run `pip install watchfiles`."
            ) from exc

        # 2. Watch for real-time changes, staying up across transient failures:
        #    if the watch loop ever errors, record it, back off, and re-establish
        #    the watch rather than letting the daemon thread die silently.
        while not (stop_event is not None and stop_event.is_set()):
            try:
                self._watch_loop(stop_event, roots, watch_fn)
                return  # clean stop
            except Exception as exc:  # noqa: BLE001 - keep watching, don't die
                if stop_event is not None and stop_event.is_set():
                    return
                self.state.set_error(str(exc))
                if stop_event is not None and stop_event.wait(_WATCH_RETRY_BACKOFF):
                    return
                if stop_event is None:
                    time.sleep(_WATCH_RETRY_BACKOFF)
                roots = self.roots() or roots

    def _watch_loop(
        self, stop_event: Optional[threading.Event], roots: List[Path], watch_fn: Any
    ) -> None:
        # The watch is (re)established; clear any error from a previous attempt.
        self.state.set_error(None)
        timeout_ms = max(1, int(self.reconcile_interval * 1000))
        for changes in watch_fn(
            *roots,
            stop_event=stop_event,
            raise_interrupt=False,
            rust_timeout=timeout_ms,
            yield_on_timeout=True,
        ):
            if stop_event is not None and stop_event.is_set():
                break
            if changes:
                events = self.react([Path(path) for _, path in changes])
            else:
                # Timeout tick with no events: periodic safety reconcile.
                events = self.reconcile()
            for event in events:
                self.on_event(event)
            self.state.persist()


def _any_under(paths: List[Path], roots: List[Path]) -> bool:
    for path in paths:
        for root in roots:
            if path == root or _is_within(path, root):
                return True
    return False


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
