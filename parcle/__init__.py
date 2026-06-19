"""Parcle — sync for AI agents.

Parcle keeps the AI agents you use in sync. Three things travel with you:

* **Memory** — a cloud, per-user long-term memory: ingest conversations and
  files, then ask questions in natural language and get cited answers back
  (this module; see :class:`Parcle`).
* **Skills** and **Conversations** — local, multi-agent sync through the
  :class:`~parcle.sync.Sync` entry point; see :mod:`parcle.sync`.

    from parcle import Parcle

    client = Parcle(api_key="pk_live_...")
    client.ingest_dialog(user_id="ada", messages=[{"role": "user", "content": "..."}])
    result = client.search(user_id="ada", query="What food should I avoid?")
    print(result.answer, result.confidence, result.citations)
"""

from __future__ import annotations

from .client import Parcle
from .exceptions import (
    AuthenticationError,
    FileTooLargeError,
    InternalServerError,
    InvalidRequestError,
    NotFoundError,
    ParcleAPIError,
    ParcleConfigError,
    ParcleConnectionError,
    ParcleError,
    ParcleTimeoutError,
    RateLimitError,
    ServiceUnavailableError,
    UnsupportedFileTypeError,
    ValidationError,
)
from .models import (
    Citation,
    DeleteResult,
    Event,
    IngestDialogResult,
    IngestFileResult,
    Message,
    SearchResult,
    Session,
    Source,
    SourcesPage,
    User,
)
from .sync import Sync

__version__ = "0.2.0"

__all__ = [
    "Parcle",
    "Sync",
    "__version__",
    # models
    "Citation",
    "DeleteResult",
    "Event",
    "IngestDialogResult",
    "IngestFileResult",
    "Message",
    "SearchResult",
    "Session",
    "Source",
    "SourcesPage",
    "User",
    # exceptions
    "ParcleError",
    "ParcleConfigError",
    "ParcleConnectionError",
    "ParcleTimeoutError",
    "ParcleAPIError",
    "InvalidRequestError",
    "AuthenticationError",
    "NotFoundError",
    "FileTooLargeError",
    "UnsupportedFileTypeError",
    "ValidationError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
]
