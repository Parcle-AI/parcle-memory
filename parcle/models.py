"""Typed response models returned by the Parcle client.

These are lightweight dataclasses parsed from the API's JSON responses. Unknown
fields are ignored, so the models stay forward-compatible as the API grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# A free-form tag attached to a source: a flat mapping of string keys to scalar
# values used for grouping and filtering memory within one user.
Tag = Dict[str, Any]


@dataclass
class User:
    """A memory namespace returned by :meth:`Parcle.create_user`."""

    user_id: str
    name: Optional[str] = None
    timezone: str = "UTC"
    is_new: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        return cls(
            user_id=data["user_id"],
            name=data.get("name"),
            timezone=data.get("timezone", "UTC"),
            is_new=bool(data.get("is_new", False)),
        )


@dataclass
class Message:
    """A single dialog message, in or out."""

    role: str
    content: str
    speaker: Optional[str] = None
    updated_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            speaker=data.get("speaker"),
            updated_at=data.get("updated_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.speaker is not None:
            out["speaker"] = self.speaker
        if self.updated_at is not None:
            out["updated_at"] = self.updated_at
        return out


@dataclass
class IngestDialogResult:
    """Result of :meth:`Parcle.ingest_dialog`."""

    session_id: str
    event_id: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestDialogResult":
        return cls(session_id=data["session_id"], event_id=data["event_id"])


@dataclass
class IngestFileResult:
    """Result of :meth:`Parcle.ingest_file`."""

    file_id: str
    event_id: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestFileResult":
        return cls(file_id=data["file_id"], event_id=data["event_id"])


@dataclass
class Event:
    """Ingestion status for a single write, from :meth:`Parcle.get_event`."""

    event_id: str
    status: str
    error: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """True once the ingested content is searchable."""
        return self.status == "ready"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def is_terminal(self) -> bool:
        """True once the event will not change state again."""
        return self.status in ("ready", "failed")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(
            event_id=data["event_id"],
            status=data["status"],
            error=data.get("error"),
        )


@dataclass
class Citation:
    """A source the search answer draws on."""

    type: str  # "session" | "file"
    id: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Citation":
        return cls(type=data["type"], id=data["id"])


@dataclass
class SearchResult:
    """Result of :meth:`Parcle.search` — an answer grounded in citations."""

    answer: str
    confidence: float
    citations: List[Citation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        return cls(
            answer=data["answer"],
            confidence=float(data["confidence"]),
            citations=[Citation.from_dict(c) for c in data.get("citations", [])],
        )


@dataclass
class Source:
    """A dialog session or file in a user's memory."""

    id: str
    type: str  # "session" | "file"
    updated_at: Optional[str] = None
    tag: Optional[Tag] = None
    name: Optional[str] = None  # filename, files only

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Source":
        return cls(
            id=data["id"],
            type=data["type"],
            updated_at=data.get("updated_at"),
            tag=data.get("tag"),
            name=data.get("name"),
        )


@dataclass
class SourcesPage:
    """One page of :meth:`Parcle.list_sources`."""

    sources: List[Source]
    page: int
    total_pages: int
    total: int

    def __iter__(self):
        return iter(self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourcesPage":
        return cls(
            sources=[Source.from_dict(s) for s in data.get("sources", [])],
            page=int(data.get("page", 1)),
            total_pages=int(data.get("total_pages", 1)),
            total=int(data.get("total", 0)),
        )


@dataclass
class Session:
    """A dialog session's original messages, from :meth:`Parcle.get_session`."""

    session_id: str
    messages: List[Message]
    tag: Optional[Tag] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            session_id=data["session_id"],
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            tag=data.get("tag"),
        )


@dataclass
class DeleteResult:
    """Result of any ``delete_by_*`` call."""

    deleted: bool
    deleted_count: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeleteResult":
        return cls(
            deleted=bool(data["deleted"]),
            deleted_count=int(data["deleted_count"]),
        )
