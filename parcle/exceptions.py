"""Exception hierarchy for the Parcle client.

Every non-2xx API response is translated into a :class:`ParcleAPIError`
subclass chosen by HTTP status. The server's stable ``error.code`` string and
human-readable ``error.message`` are preserved on the exception, along with the
``request_id`` for support correlation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ParcleError(Exception):
    """Base class for every error raised by this library."""


class ParcleConfigError(ParcleError):
    """Raised for misconfiguration, e.g. a missing API key."""


class ParcleConnectionError(ParcleError):
    """Raised when the request never produced an HTTP response.

    Covers connection failures, DNS errors, and timeouts.
    """


class ParcleTimeoutError(ParcleError):
    """Raised when a polling helper exceeds its deadline."""


class ParcleAPIError(ParcleError):
    """Raised when the API returns a non-2xx response.

    Attributes
    ----------
    status_code:
        The HTTP status code of the response.
    code:
        The stable ``error.code`` string for programmatic handling.
    message:
        The human-readable ``error.message``.
    request_id:
        The server-assigned request id, useful for support.
    body:
        The raw decoded response body, when available.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        request_id: Optional[str] = None,
        body: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        self.body = body

    def __str__(self) -> str:
        parts = []
        if self.status_code is not None:
            parts.append(f"HTTP {self.status_code}")
        if self.code:
            parts.append(self.code)
        prefix = f"[{' '.join(parts)}] " if parts else ""
        suffix = f" (request_id={self.request_id})" if self.request_id else ""
        return f"{prefix}{self.message}{suffix}"


# -- Status-specific subclasses ------------------------------------------------


class InvalidRequestError(ParcleAPIError):
    """HTTP 400 — malformed JSON, wrong field type, or missing required field."""


class AuthenticationError(ParcleAPIError):
    """HTTP 401 — missing or invalid bearer key."""


class NotFoundError(ParcleAPIError):
    """HTTP 404 — unknown user, session, file, or event."""


class FileTooLargeError(ParcleAPIError):
    """HTTP 413 — uploaded file exceeds the size limit (10 MB)."""


class UnsupportedFileTypeError(ParcleAPIError):
    """HTTP 415 — unsupported extension or database-like file."""


class ValidationError(ParcleAPIError):
    """HTTP 422 — well-formed request that breaks a rule."""


class RateLimitError(ParcleAPIError):
    """HTTP 429 — too many requests."""


class InternalServerError(ParcleAPIError):
    """HTTP 500 — unexpected server failure."""


class ServiceUnavailableError(ParcleAPIError):
    """HTTP 503 — temporarily overloaded or down."""


_STATUS_TO_EXCEPTION: Dict[int, type] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    404: NotFoundError,
    413: FileTooLargeError,
    415: UnsupportedFileTypeError,
    422: ValidationError,
    429: RateLimitError,
    500: InternalServerError,
    503: ServiceUnavailableError,
}


def error_from_response(
    status_code: int, body: Optional[Any], *, fallback_message: str
) -> ParcleAPIError:
    """Build the appropriate :class:`ParcleAPIError` for a failed response."""
    code: Optional[str] = None
    message = fallback_message
    request_id: Optional[str] = None

    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            code = err.get("code")
            message = err.get("message") or message
            request_id = err.get("request_id")

    exc_cls = _STATUS_TO_EXCEPTION.get(status_code, ParcleAPIError)
    return exc_cls(
        message,
        status_code=status_code,
        code=code,
        request_id=request_id,
        body=body,
    )
