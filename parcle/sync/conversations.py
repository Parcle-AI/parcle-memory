"""Collect agent conversations into one local, unified format.

Different agents store their chat transcripts in different shapes (Claude Code
and Codex both use line-delimited JSON, but with different record layouts).
This module discovers those transcripts, parses them into a single
agent-agnostic **canonical** model (:class:`Turn` / :class:`Block`), and writes
them — incrementally — into a local store as canonical JSONL (one turn per line).

Everything is local and read-only with respect to the agents: Parcle only reads
their transcript files and writes its own copies under ``<PARCLE_HOME>``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agents import Agent, detect_agents, parcle_home
from .events import CAT_CONVERSATION, STATUS_ERROR, EventLog

__all__ = [
    "Block",
    "Turn",
    "ConversationSession",
    "CollectResult",
    "ConversationStore",
    "ConversationError",
    "discover_sessions",
    "parse_records",
]


class ConversationError(Exception):
    """A conversation could not be read back (unknown session, missing turn, …)."""

# Block kinds (kept aligned with the canonical model upstream).
BLOCK_TEXT = "text"
BLOCK_THINKING = "thinking"
BLOCK_TOOL_CALL = "tool_call"
BLOCK_TOOL_RESULT = "tool_result"
BLOCK_IMAGE = "image"


# ---------------------------------------------------------------------------
# Canonical model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """One piece of a turn: text, thinking, a tool call, a tool result, …."""

    type: str
    text: Optional[str] = None  # text / thinking
    name: Optional[str] = None  # tool name (tool_call / tool_result)
    tool_input: Any = None  # tool_call input
    output: Optional[str] = None  # tool_result text
    status: Optional[str] = None  # tool_result status, e.g. "error"
    mime: Optional[str] = None  # image / file media type

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.type}
        for key in ("text", "name", "output", "status", "mime"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.tool_input is not None:
            out["tool_input"] = self.tool_input
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Block":
        """Rebuild a block from its :meth:`to_dict` form."""
        return cls(
            type=str(data.get("type") or BLOCK_TEXT),
            text=data.get("text"),
            name=data.get("name"),
            tool_input=data.get("tool_input"),
            output=data.get("output"),
            status=data.get("status"),
            mime=data.get("mime"),
        )


@dataclass(frozen=True)
class Turn:
    """One message in a conversation, numbered by ``seq`` within the session."""

    seq: int
    role: str  # user | assistant | tool | system | developer | …
    timestamp: Optional[str]
    blocks: Tuple[Block, ...]
    native_type: Optional[str] = None  # source record type, preserved verbatim

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "seq": self.seq,
            "role": self.role,
            "timestamp": self.timestamp,
            "blocks": [b.to_dict() for b in self.blocks],
        }
        if self.native_type is not None:
            out["native_type"] = self.native_type
        return out

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        """Rebuild a turn from its :meth:`to_dict` / stored JSONL form."""
        return cls(
            seq=int(data.get("seq", 0) or 0),
            role=str(data.get("role") or ""),
            timestamp=data.get("timestamp"),
            blocks=tuple(Block.from_dict(b) for b in data.get("blocks") or []),
            native_type=data.get("native_type"),
        )

    def render(self) -> str:
        """Render this turn as a human-readable ``[role] body`` string.

        Text and thinking blocks show their text; tool calls and tool results —
        which carry no ``.text`` — are shown with a ``[tool_call …]`` /
        ``[tool_result]`` marker, and images with ``[image …]``, so no turn ever
        renders blank. Multiple blocks are joined by newlines.
        """
        lines: List[str] = []
        for block in self.blocks:
            if block.type == BLOCK_TEXT:
                lines.append(block.text or "")
            elif block.type == BLOCK_THINKING:
                lines.append(f"(thinking) {block.text or ''}".rstrip())
            elif block.type == BLOCK_TOOL_CALL:
                lines.append(f"[tool_call {block.name}] {block.tool_input}".rstrip())
            elif block.type == BLOCK_TOOL_RESULT:
                marker = "[tool_result error]" if block.status == "error" else "[tool_result]"
                lines.append(f"{marker} {block.output or ''}".rstrip())
            elif block.type == BLOCK_IMAGE:
                lines.append(f"[image {block.mime or ''}]".rstrip())
            else:
                lines.append(f"[{block.type}]")
        body = "\n".join(line for line in lines if line) or "(empty)"
        return f"[{self.role}] {body}"


# ---------------------------------------------------------------------------
# Parsing: native records -> canonical turns
# ---------------------------------------------------------------------------


@dataclass
class _Partial:
    role: str
    timestamp: Optional[str]
    blocks: List[Block]
    native_type: Optional[str]


@dataclass
class ParseResult:
    turns: List[Turn] = field(default_factory=list)
    title: Optional[str] = None


def parse_records(
    fmt: str, records: List[dict], *, start_seq: int
) -> ParseResult:
    """Parse native ``records`` of format ``fmt`` into canonical turns.

    Turns are numbered from ``start_seq + 1``. Contentless telemetry records are
    skipped; everything carrying displayable content is preserved.

    Raises :class:`ValueError` for an unrecognized ``fmt``.
    """
    if fmt not in ("claude", "codex"):
        raise ValueError(f"Unknown transcript format: {fmt!r}")
    result = ParseResult()
    seq = start_seq
    for record in records:
        if not isinstance(record, dict):
            continue
        if fmt == "claude":
            if record.get("type") == "ai-title":
                title = record.get("aiTitle")
                if isinstance(title, str) and title.strip():
                    result.title = title.strip()
                continue
            partial = _parse_claude(record)
        elif fmt == "codex":
            partial = _parse_codex(record)
        else:
            partial = None
        if partial is None or not partial.blocks:
            continue
        seq += 1
        result.turns.append(
            Turn(
                seq=seq,
                role=partial.role,
                timestamp=partial.timestamp,
                blocks=tuple(partial.blocks),
                native_type=partial.native_type,
            )
        )
    return result


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


# -- Claude Code  (~/.claude/projects/<slug>/<sessionId>.jsonl) ----------------

_CLAUDE_SKIP_TYPES = {"queue-operation", "file-history-snapshot", "last-prompt"}


def _parse_claude(record: dict) -> Optional[_Partial]:
    record_type = record.get("type")
    if record_type in _CLAUDE_SKIP_TYPES:
        return None
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    role = str(message.get("role") or record_type or "")
    timestamp = record.get("timestamp")
    blocks = _claude_blocks(message.get("content"))
    if not blocks:
        return None
    return _Partial(
        role=role,
        timestamp=str(timestamp) if timestamp else None,
        blocks=blocks,
        native_type=str(record_type) if record_type else None,
    )


def _claude_blocks(content: Any) -> List[Block]:
    if content is None:
        return []
    if isinstance(content, str):
        return [Block(BLOCK_TEXT, text=content)] if content.strip() else []
    if not isinstance(content, list):
        return [Block(BLOCK_TEXT, text=json.dumps(content, ensure_ascii=False))]
    blocks: List[Block] = []
    for raw in content:
        if isinstance(raw, str):
            if raw.strip():
                blocks.append(Block(BLOCK_TEXT, text=raw))
            continue
        if not isinstance(raw, dict):
            continue
        block_type = raw.get("type")
        if block_type == "text":
            text = raw.get("text") or ""
            if text.strip():
                blocks.append(Block(BLOCK_TEXT, text=text))
        elif block_type == "thinking":
            text = raw.get("thinking") or ""
            if text.strip():
                blocks.append(Block(BLOCK_THINKING, text=text))
        elif block_type == "tool_use":
            blocks.append(
                Block(BLOCK_TOOL_CALL, name=raw.get("name"), tool_input=raw.get("input"))
            )
        elif block_type == "tool_result":
            output = _result_text(raw.get("content"))
            blocks.append(
                Block(
                    BLOCK_TOOL_RESULT,
                    output=output,
                    status="error" if raw.get("is_error") else None,
                )
            )
        elif block_type == "image":
            source = raw.get("source")
            mime = source.get("media_type") if isinstance(source, dict) else None
            blocks.append(Block(BLOCK_IMAGE, mime=mime))
        else:
            blocks.append(Block(BLOCK_TEXT, text=json.dumps(raw, ensure_ascii=False)))
    return blocks


def _result_text(content: Any) -> Optional[str]:
    parts: List[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif item.get("type") == "image":
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False))
    joined = "\n".join(part for part in parts if part)
    return joined or None


# -- Codex  (~/.codex/sessions/**/rollout-*.jsonl) -----------------------------

_CODEX_SKIP_PAYLOADS = {
    "token_count",
    "task_started",
    "task_complete",
    "turn_aborted",
    "thread_rolled_back",
    "context_compacted",
    "user_message",
    "agent_message",
    "patch_apply_end",
}


def _parse_codex(record: dict) -> Optional[_Partial]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    ptype = payload.get("type")
    if ptype in _CODEX_SKIP_PAYLOADS:
        return None
    timestamp = record.get("timestamp")
    ts = str(timestamp) if timestamp else None

    def partial(role: str, blocks: List[Block]) -> Optional[_Partial]:
        if not blocks:
            return None
        return _Partial(
            role=role,
            timestamp=ts,
            blocks=blocks,
            native_type=str(ptype) if ptype else None,
        )

    if ptype == "message":
        role = str(payload.get("role") or "assistant")
        return partial(role, _codex_message_blocks(payload.get("content")))

    if ptype == "reasoning":
        texts: List[str] = []
        for item in payload.get("summary") or []:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]))
        if isinstance(payload.get("content"), str) and payload["content"].strip():
            texts.append(payload["content"])
        return partial(
            "assistant",
            [Block(BLOCK_THINKING, text="\n".join(texts))] if texts else [],
        )

    if ptype in ("function_call", "custom_tool_call"):
        raw_input = (
            payload.get("arguments")
            if ptype == "function_call"
            else payload.get("input")
        )
        return partial(
            "assistant",
            [Block(BLOCK_TOOL_CALL, name=payload.get("name"), tool_input=_maybe_json(raw_input))],
        )

    if ptype in ("function_call_output", "custom_tool_call_output"):
        output = payload.get("output")
        if isinstance(output, (dict, list)):
            output = json.dumps(output, ensure_ascii=False)
        return partial(
            "tool", [Block(BLOCK_TOOL_RESULT, output=str(output) if output else None)]
        )

    if ptype in ("web_search_call", "web_search_end"):
        query = payload.get("query") or (payload.get("action") or {}).get("query")
        role = "assistant" if ptype == "web_search_call" else "tool"
        block = Block(
            BLOCK_TOOL_CALL if ptype == "web_search_call" else BLOCK_TOOL_RESULT,
            name="web_search",
            tool_input=query if ptype == "web_search_call" else None,
            output=str(query) if ptype == "web_search_end" else None,
        )
        return partial(role, [block])

    text = payload.get("message") or payload.get("text")
    if isinstance(text, str) and text.strip():
        return partial("assistant", [Block(BLOCK_TEXT, text=text)])
    return None


def _codex_message_blocks(content: Any) -> List[Block]:
    if isinstance(content, str):
        return [Block(BLOCK_TEXT, text=content)] if content.strip() else []
    if not isinstance(content, list):
        return []
    blocks: List[Block] = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
            blocks.append(Block(BLOCK_TEXT, text=item["text"]))
    return blocks


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversationSession:
    """A transcript file discovered for an agent, before it is collected."""

    agent_type: str
    session_id: str
    path: Path
    fmt: str
    project: Optional[str] = None
    title: Optional[str] = None
    cwd: Optional[str] = None
    git_branch: Optional[str] = None


def discover_sessions(agent: Agent) -> List[ConversationSession]:
    """Discover all transcript sessions for ``agent`` on this machine."""
    root = agent.transcripts_root
    if root is None or agent.transcript_glob is None or not root.is_dir():
        return []
    fmt = agent.transcript_format or agent.type
    if fmt == "codex":
        return _discover_codex(agent, root)
    sessions: List[ConversationSession] = []
    for path in sorted(root.glob(agent.transcript_glob)):
        if not path.is_file():
            continue
        sessions.append(
            ConversationSession(
                agent_type=agent.type,
                session_id=path.stem,
                path=path,
                fmt=fmt,
                project=path.parent.name,
            )
        )
    return sessions


def _discover_codex(agent: Agent, root: Path) -> List[ConversationSession]:
    title_index = _codex_title_index(root)
    sessions: List[ConversationSession] = []
    for path in sorted(root.glob(agent.transcript_glob or "**/rollout-*.jsonl")):
        if not path.is_file():
            continue
        meta = _codex_rollout_metadata(path)
        native_id = meta.get("id")
        sessions.append(
            ConversationSession(
                agent_type=agent.type,
                session_id=path.stem,
                path=path,
                fmt="codex",
                project=_basename(meta.get("cwd")) or _codex_project_from_path(path),
                title=title_index.get(native_id) if native_id else None,
                cwd=meta.get("cwd"),
                git_branch=meta.get("git_branch"),
            )
        )
    return sessions


def _codex_rollout_metadata(path: Path) -> Dict[str, Optional[str]]:
    meta: Dict[str, Optional[str]] = {"id": None, "cwd": None, "git_branch": None}
    try:
        with path.open("rb") as f:
            for index, raw_line in enumerate(f):
                if index >= 200:
                    break
                if not raw_line.strip():
                    continue
                try:
                    item = json.loads(raw_line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                payload = item.get("payload")
                if not isinstance(payload, dict):
                    continue
                if meta["id"] is None:
                    meta["id"] = _opt_str(payload.get("id"))
                if meta["cwd"] is None:
                    meta["cwd"] = _opt_str(payload.get("cwd"))
                if meta["git_branch"] is None:
                    git = payload.get("git")
                    meta["git_branch"] = (
                        _opt_str(git.get("branch")) if isinstance(git, dict) else None
                    )
                if all(meta.values()):
                    break
    except OSError:
        pass
    return meta


def _codex_title_index(sessions_root: Path) -> Dict[str, str]:
    index_path = sessions_root.parent / "session_index.jsonl"
    titles: Dict[str, str] = {}
    try:
        with index_path.open("rb") as f:
            for raw_line in f:
                if not raw_line.strip():
                    continue
                try:
                    item = json.loads(raw_line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                session_id = _opt_str(item.get("id"))
                title = _opt_str(item.get("thread_name"))
                if session_id and title:
                    titles[session_id] = title
    except OSError:
        pass
    return titles


def _codex_project_from_path(path: Path) -> Optional[str]:
    parent = path.parent
    if parent == parent.parent:
        return None
    if parent.name.isdigit() and parent.parent.name.isdigit() and parent.parent.parent.name.isdigit():
        return None
    return parent.name


def _basename(value: Optional[str]) -> Optional[str]:
    text = _opt_str(value)
    if text is None:
        return None
    return Path(text).name or None


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------
# Local store (incremental collection)
# ---------------------------------------------------------------------------


@dataclass
class CollectResult:
    """Outcome of collecting one session."""

    session_id: str
    agent_type: str
    new_turns: int
    turn_count: int
    new_bytes: int


class ConversationStore:
    """Holds collected conversations as canonical JSONL, one file per session.

    For each session, ``<root>/<agent_type>/<session_id>.jsonl`` holds the turns
    and ``<...>.meta.json`` holds metadata plus the incremental cursor
    (``parsed_through_offset`` / ``turn_count``). Re-running :meth:`collect` only
    reads bytes past the cursor, so it is cheap to run repeatedly.
    """

    def __init__(
        self, root: Optional[Path] = None, *, event_log: Optional[EventLog] = None
    ) -> None:
        self.root = root or (parcle_home() / "conversations")
        self.event_log = event_log

    def _record(
        self,
        action: str,
        *,
        agent_type: str,
        session_id: str,
        detail: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        if self.event_log is None:
            return
        self.event_log.record(
            CAT_CONVERSATION,
            action,
            status=STATUS_ERROR if error else "ok",
            name=session_id,
            agent_type=agent_type,
            detail=detail,
            error=error,
        )

    def _paths(self, session: ConversationSession) -> Tuple[Path, Path]:
        base = self.root / session.agent_type
        return base / f"{session.session_id}.jsonl", base / f"{session.session_id}.meta.json"

    def collect(
        self, session: ConversationSession, *, full: bool = False
    ) -> CollectResult:
        """Parse and append any new turns from ``session`` into the store.

        With ``full=True`` the session is re-read from the start, replacing any
        previously collected turns.
        """
        jsonl_path, meta_path = self._paths(session)
        meta = {} if full else _read_json(meta_path)
        offset = int(meta.get("parsed_through_offset", 0) or 0)
        turn_count = int(meta.get("turn_count", 0) or 0)

        if full and jsonl_path.exists():
            jsonl_path.unlink()
            offset = 0
            turn_count = 0

        lines, new_offset = _read_new_lines(session.path, offset)
        records: List[dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        parsed = parse_records(session.fmt, records, start_seq=turn_count)

        if parsed.turns:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with jsonl_path.open("a", encoding="utf-8") as f:
                for turn in parsed.turns:
                    f.write(turn.to_jsonl() + "\n")

        turn_count += len(parsed.turns)
        title = parsed.title or session.title or meta.get("title")
        _write_json(
            meta_path,
            {
                "agent_type": session.agent_type,
                "session_id": session.session_id,
                "native_path": str(session.path),
                "project": session.project,
                "title": title,
                "cwd": session.cwd,
                "git_branch": session.git_branch,
                "parsed_through_offset": new_offset,
                "turn_count": turn_count,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return CollectResult(
            session_id=session.session_id,
            agent_type=session.agent_type,
            new_turns=len(parsed.turns),
            turn_count=turn_count,
            new_bytes=max(0, new_offset - offset),
        )

    def progress(self, session: ConversationSession) -> Dict[str, Any]:
        """Report how far this session has been collected.

        Returns ``parsed_through_offset`` and ``turn_count`` from the stored
        cursor, the transcript's current ``size``, and ``caught_up`` (True when
        the cursor has reached the end of the transcript). This is what makes
        collection resumable across restarts: a fresh process reads the same
        cursor and only appends what is new.
        """
        _, meta_path = self._paths(session)
        meta = _read_json(meta_path)
        offset = int(meta.get("parsed_through_offset", 0) or 0)
        turn_count = int(meta.get("turn_count", 0) or 0)
        try:
            size = session.path.stat().st_size
        except OSError:
            size = 0
        return {
            "parsed_through_offset": offset,
            "turn_count": turn_count,
            "size": size,
            "caught_up": offset >= size,
        }

    def collect_agent(self, agent: Agent, *, full: bool = False) -> List[CollectResult]:
        """Collect every discovered session for ``agent``."""
        return [self.collect(s, full=full) for s in discover_sessions(agent)]

    def collect_all(self, *, full: bool = False) -> List[CollectResult]:
        """Collect every session across every detected agent."""
        results: List[CollectResult] = []
        for agent in detect_agents():
            if agent.supports_conversations:
                results.extend(self.collect_agent(agent, full=full))
        return results

    # -- read-back (retrieval) -------------------------------------------------

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List every collected conversation, newest activity first.

        Reads the stored ``*.meta.json`` sidecars (not the transcripts) and
        returns one metadata dict per session — ``agent_type``, ``session_id``,
        ``title``, ``project``, ``turn_count``, ``updated_at``, … — so you can
        browse what has been archived and then :meth:`get_conversation` the one
        you want. Returns an empty list when nothing has been collected.
        """
        out: List[Dict[str, Any]] = []
        if not self.root.is_dir():
            return out
        for meta_path in self.root.glob("*/*.meta.json"):
            meta = _read_json(meta_path)
            if meta:
                out.append(meta)
        out.sort(key=lambda m: str(m.get("updated_at") or ""), reverse=True)
        return out

    def _load(self, agent_type: str, session_id: str) -> List[Turn]:
        """Read every collected :class:`Turn` of one conversation (no logging).

        Raises :class:`ConversationError` if no such conversation has been
        collected.
        """
        jsonl_path = self.root / agent_type / f"{session_id}.jsonl"
        if not jsonl_path.is_file():
            raise ConversationError(
                f"No collected conversation {agent_type}/{session_id} at {jsonl_path}."
            )
        turns: List[Turn] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    turns.append(Turn.from_dict(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        return turns

    def get_conversation(self, agent_type: str, session_id: str) -> List[Turn]:
        """Return every collected :class:`Turn` of one conversation, in order.

        Records a ``get_conversation`` event (success or failure). Raises
        :class:`ConversationError` if no such conversation has been collected.
        """
        try:
            turns = self._load(agent_type, session_id)
        except Exception as exc:  # noqa: BLE001 - log the failure, then re-raise
            self._record(
                "get_conversation",
                agent_type=agent_type,
                session_id=session_id,
                error=str(exc),
            )
            raise
        self._record(
            "get_conversation",
            agent_type=agent_type,
            session_id=session_id,
            detail=f"{len(turns)} turn(s)",
        )
        return turns

    def get_turn(self, agent_type: str, session_id: str, seq: int) -> Turn:
        """Return a single :class:`Turn` (message) of a conversation by ``seq``.

        Records a ``get_turn`` event (success or failure). Raises
        :class:`ConversationError` if the conversation has not been collected or
        has no turn with that ``seq``.
        """
        try:
            found: Optional[Turn] = None
            for turn in self._load(agent_type, session_id):
                if turn.seq == seq:
                    found = turn
                    break
            if found is None:
                raise ConversationError(
                    f"Conversation {agent_type}/{session_id} has no turn with seq {seq}."
                )
        except Exception as exc:  # noqa: BLE001 - log the failure, then re-raise
            self._record(
                "get_turn",
                agent_type=agent_type,
                session_id=session_id,
                detail=f"seq={seq}",
                error=str(exc),
            )
            raise
        self._record(
            "get_turn",
            agent_type=agent_type,
            session_id=session_id,
            detail=f"seq={seq}",
        )
        return found

    def render_conversation(
        self, agent_type: str, session_id: str, *, separator: str = "\n\n"
    ) -> str:
        """Render a collected conversation as one human-readable transcript.

        Convenience over :meth:`get_conversation` + :meth:`Turn.render`: each turn
        is rendered and joined by ``separator`` (a blank line by default). Raises
        :class:`ConversationError` if the conversation has not been collected.
        """
        turns = self.get_conversation(agent_type, session_id)
        return separator.join(turn.render() for turn in turns)


def _read_new_lines(path: Path, offset: int) -> Tuple[List[str], int]:
    """Read whole lines from ``path`` starting at byte ``offset``.

    Returns the decoded complete lines and the new byte offset (the start of any
    trailing partial line is *not* consumed, so a half-written final line is
    re-read on the next pass).
    """
    try:
        with path.open("rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return [], offset
    if not data:
        return [], offset
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return [], offset  # no complete line yet
    complete = data[: last_nl + 1]
    new_offset = offset + len(complete)
    text = complete.decode("utf-8", errors="replace")
    return text.splitlines(), new_offset


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
