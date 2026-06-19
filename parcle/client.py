"""Synchronous client for the Parcle Memory API."""

from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, IO, List, Mapping, Optional, Sequence, Tuple, Union

import httpx

from .exceptions import (
    FileTooLargeError,
    ParcleAPIError,
    ParcleConfigError,
    ParcleConnectionError,
    ParcleTimeoutError,
    error_from_response,
)
from .models import (
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

__all__ = ["Parcle"]

DEFAULT_BASE_URL = "https://api.parcle.ai"
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIEVAL_TIMEOUT = 180.0
DEFAULT_WAIT_TIMEOUT = 180.0
DEFAULT_MAX_RETRIES = 2
# Statuses worth retrying with backoff: rate limit, server error, unavailable.
_RETRY_STATUS = frozenset({429, 500, 503})

# Upload size ceiling, mirroring the server's limit. Files above this are
# rejected client-side before any request is sent. The server still enforces
# the same limit with HTTP 413 for cases we can't measure locally.
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Content types for the supported upload extensions, filling gaps left by the
# stdlib's ``mimetypes`` (which does not know markdown, for example).
_EXTRA_CONTENT_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".msg": "application/vnd.ms-outlook",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".tsv": "text/tab-separated-values",
}

# What ``ingest_file`` accepts for ``file``: a path, a binary stream, raw bytes,
# or a ``(filename, content[, content_type])`` tuple.
FileInput = Union[
    str,
    os.PathLike,
    IO[bytes],
    bytes,
    Tuple[str, Union[bytes, IO[bytes]]],
    Tuple[str, Union[bytes, IO[bytes]], Optional[str]],
]

# A message passed to ``ingest_dialog``: a plain dict or a ``Message``.
MessageInput = Union[Mapping[str, Any], Message]

# A tag / tag_filter mapping.
TagFilter = Mapping[str, Any]


class Parcle:
    """Client for the Parcle Memory API.

    Parameters
    ----------
    api_key:
        Your Parcle API key. If omitted, the ``PARCLE_API_KEY`` environment
        variable is used.
    base_url:
        Override the API base URL (e.g. for a staging environment).
    timeout:
        Per-request timeout in seconds.
    retrieval_timeout:
        Timeout in seconds for retrieval requests such as :meth:`search`.
        Retrieval requests do not retry by default.
    max_retries:
        How many times to retry safe read requests that fail with a retryable
        status (429/500/503) or a connection error, using exponential backoff.
        Writes, deletes, and retrieval requests do not retry by default.
    http_client:
        Bring your own configured :class:`httpx.Client` (proxies, custom
        transport, …). Its lifetime is then yours to manage.

    The client may be used as a context manager to ensure the underlying
    connection pool is closed::

        with Parcle() as client:
            client.search(user_id="ada", query="...")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        retrieval_timeout: float = DEFAULT_RETRIEVAL_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        key = api_key or os.environ.get("PARCLE_API_KEY")
        if not key:
            raise ParcleConfigError(
                "No API key provided. Pass api_key=... or set the "
                "PARCLE_API_KEY environment variable."
            )
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.retrieval_timeout = retrieval_timeout
        self.max_retries = max(0, int(max_retries))

        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client, if this instance owns it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "Parcle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- users -----------------------------------------------------------------

    def create_user(
        self,
        user_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
        timezone: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> User:
        """Create or update a user.

        Omit ``user_id`` to let Parcle generate one. ``timezone`` is an IANA
        zone used only to interpret relative time in searches.
        """
        payload = _drop_none(
            {"user_id": user_id, "name": name, "timezone": timezone}
        )
        data = self._request("POST", "/v1/users", json_body=payload, timeout=timeout)
        return User.from_dict(data)

    # -- ingestion -------------------------------------------------------------

    def ingest_dialog(
        self,
        user_id: str,
        messages: Sequence[MessageInput],
        *,
        session_id: Optional[str] = None,
        tag: Optional[TagFilter] = None,
        timeout: Optional[float] = None,
        wait: bool = True,
        wait_timeout: Optional[float] = DEFAULT_WAIT_TIMEOUT,
        wait_poll_interval: float = 2.0,
    ) -> IngestDialogResult:
        """Append dialog messages to a user's memory.

        Omit ``session_id`` to start a new session; pass one to append. ``tag``
        applies only when a new session is created. By default, this waits
        until the write is searchable; pass ``wait=False`` to return as soon as
        the write has been accepted.
        """
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "session_id": session_id,
            "messages": [_message_to_dict(m) for m in messages],
        }
        if tag is not None:
            payload["tag"] = dict(tag)
        data = self._request(
            "POST",
            "/v1/memories/ingest_dialog",
            json_body=payload,
            timeout=timeout,
        )
        result = IngestDialogResult.from_dict(data)
        if wait:
            self.wait_until_ready(
                user_id,
                result.event_id,
                poll_interval=wait_poll_interval,
                timeout=wait_timeout,
            )
        return result

    def ingest_file(
        self,
        user_id: str,
        file: FileInput,
        *,
        updated_at: Optional[str] = None,
        tag: Optional[TagFilter] = None,
        timeout: Optional[float] = None,
        wait: bool = True,
        wait_timeout: Optional[float] = DEFAULT_WAIT_TIMEOUT,
        wait_poll_interval: float = 2.0,
    ) -> IngestFileResult:
        """Upload a file into a user's memory.

        ``file`` may be a path, an open binary stream, raw ``bytes``, or a
        ``(filename, content[, content_type])`` tuple. For streams and bytes a
        filename is required either via the stream's ``.name`` or the tuple
        form. By default, this waits until the file is searchable; pass
        ``wait=False`` to return as soon as the upload has been accepted.

        Files larger than ``MAX_FILE_SIZE_MB`` MB are rejected with
        :class:`~parcle.exceptions.FileTooLargeError` before any request is
        sent (when the size can be measured locally).
        """
        filename, content, content_type = _prepare_file(file)
        size = _content_size(content)
        if size is not None and size > MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"File {filename!r} is {size} bytes, which exceeds the "
                f"{MAX_FILE_SIZE_MB} MB upload limit.",
                status_code=413,
                code="file_too_large",
            )
        files = {"file": (filename, content, content_type)}
        form: Dict[str, Any] = {"user_id": user_id}
        if updated_at is not None:
            form["updated_at"] = updated_at
        if tag is not None:
            form["tag"] = json.dumps(tag)
        data = self._request(
            "POST",
            "/v1/memories/ingest_files",
            files=files,
            data=form,
            timeout=timeout,
        )
        result = IngestFileResult.from_dict(data)
        if wait:
            self.wait_until_ready(
                user_id,
                result.event_id,
                poll_interval=wait_poll_interval,
                timeout=wait_timeout,
            )
        return result

    # -- events ----------------------------------------------------------------

    def get_event(
        self, user_id: str, event_id: str, *, timeout: Optional[float] = None
    ) -> Event:
        """Fetch the ingestion status for a single write."""
        data = self._request(
            "POST",
            "/v1/memories/events",
            json_body={"user_id": user_id, "event_id": event_id},
            timeout=timeout,
            max_retries=self.max_retries,
        )
        return Event.from_dict(data)

    def wait_until_ready(
        self,
        user_id: str,
        event_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: Optional[float] = DEFAULT_WAIT_TIMEOUT,
        raise_on_failed: bool = True,
    ) -> Event:
        """Poll :meth:`get_event` until the event is ready or failed.

        Returns the terminal :class:`~parcle.models.Event`. Raises
        :class:`~parcle.exceptions.ParcleTimeoutError` if ``timeout`` seconds
        pass first, and (by default) :class:`~parcle.exceptions.ParcleAPIError`
        if ingestion failed.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            event = self.get_event(user_id, event_id)
            if event.is_terminal:
                if event.is_failed and raise_on_failed:
                    raise ParcleAPIError(
                        event.error or "Ingestion failed.",
                        code="ingestion_failed",
                    )
                return event
            if deadline is not None and time.monotonic() >= deadline:
                raise ParcleTimeoutError(
                    f"Event {event_id!r} not ready after {timeout}s "
                    f"(last status: {event.status})."
                )
            time.sleep(poll_interval)

    # -- search ----------------------------------------------------------------

    def search(
        self,
        user_id: str,
        query: str,
        *,
        tag_filter: Optional[TagFilter] = None,
        timezone: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> SearchResult:
        """Ask a natural-language question over a user's memory.

        Returns an ``answer`` grounded in source ``citations``, with a
        ``confidence`` in ``[0, 1]``. Retrieval uses a longer default timeout
        than ordinary API calls and is not retried.

        Retrieval is delivered as a Server-Sent Events stream: the server may
        take longer than a CDN's idle limit, so it emits keepalives until the
        answer is ready. This is handled transparently — the streaming is an
        implementation detail and the call still returns a single
        :class:`~parcle.models.SearchResult`.
        """
        payload = _drop_none(
            {
                "user_id": user_id,
                "query": query,
                "tag_filter": dict(tag_filter) if tag_filter is not None else None,
                "timezone": timezone,
            }
        )
        data = self._stream_search(
            payload,
            timeout=self.retrieval_timeout if timeout is None else timeout,
        )
        return SearchResult.from_dict(data)

    def _stream_search(
        self, payload: Dict[str, Any], *, timeout: Optional[float]
    ) -> Dict[str, Any]:
        """Run the SSE search and return the decoded ``final`` event payload.

        Failures from the synchronous ``prepare`` phase arrive as a normal
        non-2xx HTTP response (with their real status code) and are translated
        by :meth:`_parse_response`. Once the stream has started the status is
        fixed at 200, so failures from the long ``run`` phase arrive in-band as
        an ``error`` event and become a :class:`~parcle.exceptions.ParcleAPIError`.
        """
        url = f"{self.base_url}/v1/memories/search"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }
        request_kwargs: Dict[str, Any] = {"headers": headers, "json": payload}
        if timeout is not None:
            # httpx resets its read timeout on every chunk, so the periodic
            # keepalive would otherwise keep the connection open indefinitely.
            # Two guards bound the call: httpx's read timeout catches a fully
            # silent (stalled) connection, while the wall-clock deadline below
            # caps an alive-but-slow search that keeps sending keepalives. The
            # deadline is only checked between received lines, so it fires
            # within roughly one keepalive interval of expiring.
            request_kwargs["timeout"] = timeout
        deadline = None if timeout is None else time.monotonic() + timeout

        try:
            with self._client.stream("POST", url, **request_kwargs) as response:
                if not response.is_success:
                    response.read()
                    return self._parse_response(response)

                event_name: Optional[str] = None
                data_lines: List[str] = []
                for line in response.iter_lines():
                    if deadline is not None and time.monotonic() >= deadline:
                        # Match the non-streaming timeout path, which surfaces a
                        # connection-level timeout as ParcleConnectionError.
                        raise ParcleConnectionError(
                            f"Request to {url} timed out."
                        )
                    if line == "":
                        if data_lines:
                            result = self._handle_search_event(event_name, data_lines)
                            if result is not None:
                                return result
                        event_name = None
                        data_lines = []
                        continue
                    if line.startswith(":"):
                        continue  # comment line (keepalive); ignore
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        # SSE strips at most one leading space after the colon.
                        chunk = line[len("data:") :]
                        if chunk.startswith(" "):
                            chunk = chunk[1:]
                        data_lines.append(chunk)
        except httpx.TimeoutException as exc:
            err = ParcleConnectionError(f"Request to {url} timed out.")
            err.__cause__ = exc
            raise err
        except httpx.HTTPError as exc:
            err = ParcleConnectionError(f"Request to {url} failed: {exc}")
            err.__cause__ = exc
            raise err

        raise ParcleConnectionError(
            "Search stream ended before delivering a result."
        )

    @staticmethod
    def _handle_search_event(
        event_name: Optional[str], data_lines: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Dispatch one SSE event; return the ``final`` payload or raise on error.

        Returns ``None`` for events that are neither ``final`` nor ``error`` so
        the caller keeps reading.
        """
        raw = "\n".join(data_lines)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            data = None

        if event_name == "error":
            payload = data if isinstance(data, dict) else {}
            raise ParcleAPIError(
                payload.get("message") or "Search failed.",
                code=payload.get("code"),
                request_id=payload.get("request_id"),
                body=payload or None,
            )
        if event_name == "final":
            return data if isinstance(data, dict) else {}
        return None

    # -- sources & sessions ----------------------------------------------------

    def list_sources(
        self,
        user_id: str,
        *,
        type: Optional[str] = None,
        tag_filter: Optional[TagFilter] = None,
        page: int = 1,
        limit: int = 50,
        order: str = "desc",
        timeout: Optional[float] = None,
    ) -> SourcesPage:
        """List a user's sources (dialog sessions and files), page by page.

        ``type`` is ``"session"`` or ``"file"``; omit for both.
        """
        payload = _drop_none(
            {
                "user_id": user_id,
                "type": type,
                "tag_filter": dict(tag_filter) if tag_filter is not None else None,
                "page": page,
                "limit": limit,
                "order": order,
            }
        )
        data = self._request(
            "POST",
            "/v1/memories/sources",
            json_body=payload,
            timeout=timeout,
            max_retries=self.max_retries,
        )
        return SourcesPage.from_dict(data)

    def iter_sources(
        self,
        user_id: str,
        *,
        type: Optional[str] = None,
        tag_filter: Optional[TagFilter] = None,
        limit: int = 50,
        order: str = "desc",
        timeout: Optional[float] = None,
    ):
        """Yield every source across all pages, fetching lazily."""
        page = 1
        while True:
            result = self.list_sources(
                user_id,
                type=type,
                tag_filter=tag_filter,
                page=page,
                limit=limit,
                order=order,
                timeout=timeout,
            )
            for source in result.sources:
                yield source
            if page >= result.total_pages or not result.sources:
                return
            page += 1

    def get_session(
        self, user_id: str, session_id: str, *, timeout: Optional[float] = None
    ) -> Session:
        """Read a dialog session's original messages in chronological order."""
        data = self._request(
            "POST",
            "/v1/memories/sessions",
            json_body={"user_id": user_id, "session_id": session_id},
            timeout=timeout,
            max_retries=self.max_retries,
        )
        return Session.from_dict(data)

    # -- deletion --------------------------------------------------------------

    def delete_by_session(
        self, user_id: str, session_id: str, *, timeout: Optional[float] = None
    ) -> DeleteResult:
        """Delete all memory derived from a dialog session."""
        return self._delete(
            "/v1/memories/by_session",
            {"user_id": user_id, "session_id": session_id},
            timeout=timeout,
        )

    def delete_by_file(
        self, user_id: str, file_id: str, *, timeout: Optional[float] = None
    ) -> DeleteResult:
        """Delete all memory derived from a file."""
        return self._delete(
            "/v1/memories/by_file",
            {"user_id": user_id, "file_id": file_id},
            timeout=timeout,
        )

    def delete_by_tag(
        self,
        user_id: str,
        tag_filter: TagFilter,
        *,
        timeout: Optional[float] = None,
    ) -> DeleteResult:
        """Delete all memory whose source tags match ``tag_filter``.

        ``tag_filter`` must be non-empty.
        """
        if not tag_filter:
            raise ParcleConfigError("delete_by_tag requires a non-empty tag_filter.")
        return self._delete(
            "/v1/memories/by_tag",
            {"user_id": user_id, "tag_filter": dict(tag_filter)},
            timeout=timeout,
        )

    def _delete(
        self, path: str, body: Dict[str, Any], *, timeout: Optional[float] = None
    ) -> DeleteResult:
        data = self._request("DELETE", path, json_body=body, timeout=timeout)
        return DeleteResult.from_dict(data)

    # -- transport -------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        request_kwargs: Dict[str, Any] = {
            "headers": headers,
            "json": json_body,
            "data": data,
            "files": files,
        }
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        effective_retries = 0 if max_retries is None else max(0, int(max_retries))
        attempts = effective_retries + 1

        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                response = self._client.request(method, url, **request_kwargs)
            except httpx.TimeoutException as exc:
                last_exc = ParcleConnectionError(f"Request to {url} timed out.")
                last_exc.__cause__ = exc
            except httpx.HTTPError as exc:
                last_exc = ParcleConnectionError(f"Request to {url} failed: {exc}")
                last_exc.__cause__ = exc
            else:
                if response.status_code in _RETRY_STATUS and attempt < attempts - 1:
                    self._sleep_for_retry(response, attempt)
                    continue
                return self._parse_response(response)

            # Reached only on a connection-level failure; back off and retry.
            if attempt < attempts - 1:
                time.sleep(_backoff_seconds(attempt))
                continue
            assert last_exc is not None
            raise last_exc

        # Unreachable, but keeps type-checkers happy.
        assert last_exc is not None
        raise last_exc

    def _parse_response(self, response: httpx.Response) -> Dict[str, Any]:
        body: Any
        try:
            body = response.json() if response.content else None
        except (json.JSONDecodeError, ValueError):
            body = None

        if response.is_success:
            return body if isinstance(body, dict) else {}

        raise error_from_response(
            response.status_code,
            body,
            fallback_message=(
                response.text or f"Request failed with status {response.status_code}."
            ),
        )

    @staticmethod
    def _sleep_for_retry(response: httpx.Response, attempt: int) -> None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                time.sleep(float(retry_after))
                return
            except ValueError:
                pass
        time.sleep(_backoff_seconds(attempt))


# -- module-level helpers ------------------------------------------------------


def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys whose value is None so they are omitted from the request."""
    return {k: v for k, v in d.items() if v is not None}


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff: 0.5s, 1s, 2s, … capped at 8s."""
    return min(0.5 * (2 ** attempt), 8.0)


def _message_to_dict(message: MessageInput) -> Dict[str, Any]:
    if isinstance(message, Message):
        return message.to_dict()
    if isinstance(message, Mapping):
        if "role" not in message or "content" not in message:
            raise ParcleConfigError(
                "Each message requires 'role' and 'content' keys."
            )
        return _drop_none(dict(message))
    raise TypeError(
        f"Message must be a dict or Message, got {type(message).__name__}."
    )


def _guess_content_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in _EXTRA_CONTENT_TYPES:
        return _EXTRA_CONTENT_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _content_size(content: Union[bytes, IO[bytes]]) -> Optional[int]:
    """Best-effort byte length of upload content, or ``None`` if unknown.

    Bytes are measured directly; a seekable stream is measured without
    consuming it (seek to end, then restore the position). Non-seekable streams
    return ``None`` so the upload proceeds and the server enforces the limit.
    """
    if isinstance(content, (bytes, bytearray)):
        return len(content)
    seek = getattr(content, "seek", None)
    tell = getattr(content, "tell", None)
    if not callable(seek) or not callable(tell):
        return None
    try:
        pos = tell()
        seek(0, os.SEEK_END)
        size = tell()
        seek(pos)
    except (OSError, ValueError):
        return None
    return size


def _prepare_file(
    file: FileInput,
) -> Tuple[str, Union[bytes, IO[bytes]], str]:
    """Normalise the ``file`` argument into ``(filename, content, content_type)``."""
    # (filename, content) or (filename, content, content_type)
    if isinstance(file, tuple):
        if len(file) == 2:
            filename, content = file
            content_type = None
        elif len(file) == 3:
            filename, content, content_type = file
        else:
            raise ParcleConfigError(
                "file tuple must be (filename, content[, content_type])."
            )
        return filename, content, content_type or _guess_content_type(filename)

    # Path-like → open and read.
    if isinstance(file, (str, os.PathLike)):
        path = Path(file)
        if not path.is_file():
            raise ParcleConfigError(f"File not found: {path}")
        return path.name, path.read_bytes(), _guess_content_type(path.name)

    # Raw bytes with no filename — we can't infer one.
    if isinstance(file, (bytes, bytearray)):
        raise ParcleConfigError(
            "Raw bytes need a filename; pass a (filename, bytes) tuple instead."
        )

    # Assume a binary stream; derive the filename from .name if present.
    name = getattr(file, "name", None)
    if not name:
        raise ParcleConfigError(
            "Could not determine a filename for the file stream; pass a "
            "(filename, stream) tuple instead."
        )
    filename = os.path.basename(name)
    return filename, file, _guess_content_type(filename)
