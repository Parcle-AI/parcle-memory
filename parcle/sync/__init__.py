"""Local, multi-agent sync for skills and conversations.

Parcle is built for keeping AI coding agents in sync. Beyond cloud memory (see
:class:`parcle.Parcle`), this package syncs two more things across the agents on
your machine — entirely locally, with no cloud:

* **Skills** — a local master library of ``SKILL.md`` skill folders. You
  ``push`` skills into it from your agents and ``pull`` them back out; both are
  manual. Watching only tracks which skills each agent holds, never writing to
  the library (:mod:`parcle.sync.skills`).
* **Conversations** — every agent's chat transcript, collected into one unified
  canonical format (:mod:`parcle.sync.conversations`).

:class:`Sync` is the single local entry point; it needs no API key. Constructing
it starts a background daemon that backfills once and then keeps both sides
current; you query it live and stop it when done::

    from parcle import Sync

    sync = Sync()              # starts syncing in the background (non-blocking)
    sync.wait_until_ready()    # initial backfill done
    sync.skills.list()         # everything in the master library
    sync.status()              # readiness + counts + last activity
    sync.stop()                # or use `with Sync() as sync: ...`

Pass ``autostart=False`` for an inert object you drive manually (``push`` /
``pull`` / ``collect_all`` / foreground ``watch``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agents import (
    Agent,
    detect_agents,
    get_agent,
    known_agents,
    parcle_home,
)
from .conversations import (
    Block,
    CollectResult,
    ConversationError,
    ConversationSession,
    ConversationStore,
    Turn,
    discover_sessions,
    parse_records,
)
from .events import Event, EventLog
from .skills import Skill, SkillError, SkillLibrary, parse_frontmatter
from .watch import SyncState, WatchEvent, Watcher

__all__ = [
    # facade
    "Sync",
    # agents
    "Agent",
    "known_agents",
    "detect_agents",
    "get_agent",
    "parcle_home",
    # skills
    "Skill",
    "SkillError",
    "SkillLibrary",
    "parse_frontmatter",
    # conversations
    "Block",
    "Turn",
    "ConversationSession",
    "CollectResult",
    "ConversationStore",
    "ConversationError",
    "discover_sessions",
    "parse_records",
    # watching
    "Watcher",
    "WatchEvent",
    "SyncState",
    # event stream
    "Event",
    "EventLog",
]


class Sync:
    """The local entry point for skill and conversation sync (no API key).

    Groups the local sync features behind one object, mirroring how
    :class:`parcle.Parcle` is the entry point for cloud memory:

    * :attr:`skills` — the master skill library (push / pull / inventory).
    * :attr:`conversations` — the collected-conversation store.
    * a background daemon that watches your agents and keeps both current.

    By default, constructing ``Sync()`` immediately starts that daemon in a
    background thread: it does one full backfill (collect existing conversations
    and take an initial skill inventory), then watches for changes — all without
    blocking. Watching the skills side only tracks each agent's inventory; the
    library is changed only by manual ``push`` / ``pull``. Query it live::

        sync = Sync()
        sync.wait_until_ready()          # backfill done
        sync.skills.list()               # what you've pushed into the library
        sync.inventory()                 # which skills each agent holds
        sync.status()                    # counts + readiness + last activity
        sync.stop()                      # or use `with Sync() as sync: ...`

    The daemon lives only as long as the process; collection state is persisted
    locally (per-session cursors and ``<home>/state.json``) so a new process
    resumes incrementally rather than starting over.

    Parameters
    ----------
    home:
        Where the library, conversations, and state file live. Defaults to
        ``~/.parcle`` (or the ``PARCLE_HOME`` environment variable). Handy for
        tests and isolation.
    autostart:
        Start the background daemon on construction (default). Pass ``False`` to
        build an inert object and call :meth:`start` yourself.
    watch_skills / watch_conversations:
        Which sides the daemon should watch.
    on_event:
        Called with each :class:`~parcle.sync.watch.WatchEvent` the daemon emits.
    """

    def __init__(
        self,
        home: Optional[Path] = None,
        *,
        autostart: bool = True,
        watch_skills: bool = True,
        watch_conversations: bool = True,
        on_event: Optional[Callable[[WatchEvent], None]] = None,
        reconcile_interval: float = 5.0,
    ) -> None:
        base = Path(home).expanduser() if home is not None else parcle_home()
        self.home = base
        self._events = EventLog(base / "events.jsonl")
        self._daemon_active = False
        self.skills = SkillLibrary(base / "skills", event_log=self._events)
        self.conversations = ConversationStore(
            base / "conversations", event_log=self._events
        )
        self._state = SyncState(base / "state.json")
        self._watcher = Watcher(
            self,
            skills=watch_skills,
            conversations=watch_conversations,
            on_event=on_event,
            state=self._state,
            reconcile_interval=reconcile_interval,
        )
        if autostart:
            self.start()

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> "Sync":
        """Start the background sync daemon (non-blocking). Idempotent.

        Records a ``daemon`` ``start`` event the first time it actually starts
        (and a failure event if starting raises).
        """
        if self._daemon_active:
            return self
        try:
            self._watcher.start()
        except Exception as exc:  # noqa: BLE001 - log the failure, then re-raise
            self._events.record("daemon", "start", status="error", error=str(exc))
            raise
        self._daemon_active = True
        self._events.record("daemon", "start")
        return self

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """Stop the background daemon and wait for it to finish.

        Records a ``daemon`` ``stop`` event when a running daemon is stopped (and
        a failure event if stopping raises). A no-op if it is not running.
        """
        if not self._daemon_active:
            return
        try:
            self._watcher.stop(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - log the failure, then re-raise
            self._events.record("daemon", "stop", status="error", error=str(exc))
            raise
        self._daemon_active = False
        self._events.record("daemon", "stop")

    def is_running(self) -> bool:
        return self._watcher.is_running()

    def __enter__(self) -> "Sync":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- readiness -------------------------------------------------------------

    def skills_ready(self) -> bool:
        """True when the daemon is not currently parsing skills."""
        return self._state.ready("skills")

    def conversations_ready(self) -> bool:
        """True when the daemon is not currently collecting conversations."""
        return self._state.ready("conversations")

    def wait_until_ready(
        self,
        kind: Optional[str] = None,
        *,
        timeout: Optional[float] = None,
        poll_interval: float = 0.05,
    ) -> bool:
        """Block until the daemon is idle, returning whether it became ready.

        ``kind`` is ``"skills"``, ``"conversations"``, or ``None`` for both.
        Returns ``False`` if ``timeout`` seconds pass first.
        """
        kinds = ("skills", "conversations") if kind is None else (kind,)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if all(self._state.ready(k) for k in kinds):
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(poll_interval)

    # -- queries ---------------------------------------------------------------

    def list_skills(self) -> List[Skill]:
        """Every skill currently in the master library."""
        return self.skills.list()

    def inventory(self) -> Dict[str, List[Skill]]:
        """Which skills each detected agent currently holds."""
        return self.skills.inventory()

    def drift(self) -> Dict[str, List[str]]:
        """Which agent-held skills are missing from the library or have diverged.

        Returns ``{"missing_from_library": [...], "diverged": [...]}`` (each entry
        ``"<agent_type>/<key>"``); see :meth:`SkillLibrary.drift`.
        """
        return self.skills.drift()

    def list_conversations(self) -> List[Dict[str, Any]]:
        """Metadata for every collected conversation (newest activity first)."""
        return self.conversations.list_conversations()

    def get_conversation(self, agent_type: str, session_id: str) -> List[Turn]:
        """Every collected :class:`Turn` of one conversation, in order."""
        return self.conversations.get_conversation(agent_type, session_id)

    def get_turn(self, agent_type: str, session_id: str, seq: int) -> Turn:
        """A single collected :class:`Turn` (message) of a conversation by ``seq``."""
        return self.conversations.get_turn(agent_type, session_id, seq)

    def render_conversation(
        self, agent_type: str, session_id: str, *, separator: str = "\n\n"
    ) -> str:
        """One collected conversation rendered as a human-readable transcript."""
        return self.conversations.render_conversation(
            agent_type, session_id, separator=separator
        )

    def events(
        self,
        *,
        limit: Optional[int] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Event]:
        """Read the persistent activity stream, oldest first.

        Records cover daemon ``start`` / ``stop``, single skill ``push`` /
        ``pull``, and single conversation reads (``get_conversation`` /
        ``get_turn``), each with a success or failure ``status``. ``category``
        (``"daemon"`` / ``"skill"`` / ``"conversation"``) and ``status``
        (``"ok"`` / ``"error"``) filter the result; ``limit`` keeps the most
        recent N.
        """
        return self._events.read(limit=limit, category=category, status=status)

    def status(self) -> Dict[str, Any]:
        """A snapshot of sync state: readiness, counts, and last activity."""
        snap = self._state.snapshot()
        snap["running"] = self.is_running()
        return snap

    # -- foreground ------------------------------------------------------------

    def watch(self) -> None:
        """Watch in the foreground until interrupted (blocking).

        An alternative to the background daemon; mostly for scripts that want to
        block. Requires the ``watchfiles`` package.
        """
        self._watcher.run()
