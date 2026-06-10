"""Parcle — long-term memory for AI agents.

Ingest conversations and files into a per-user memory, then ask questions in
natural language and get cited answers back.

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

__version__ = "0.1.0"

__all__ = [
    "Parcle",
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
