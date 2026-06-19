"""Tests for the Parcle client, mocking HTTP with respx."""

from __future__ import annotations

import inspect
import json

import httpx
import pytest
import respx

import parcle
from parcle import (
    AuthenticationError,
    FileTooLargeError,
    NotFoundError,
    Parcle,
    ParcleConfigError,
    RateLimitError,
    ValidationError,
)
from parcle.client import MAX_FILE_SIZE_BYTES

BASE = "https://api.parcle.ai"


@pytest.fixture
def client():
    with Parcle(api_key="pk_test_123") as c:
        yield c


# -- construction --------------------------------------------------------------


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("PARCLE_API_KEY", "pk_env_456")
    with Parcle() as c:
        assert c.api_key == "pk_env_456"


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("PARCLE_API_KEY", raising=False)
    with pytest.raises(ParcleConfigError):
        Parcle()


def test_explicit_key_beats_env(monkeypatch):
    monkeypatch.setenv("PARCLE_API_KEY", "pk_env")
    with Parcle(api_key="pk_explicit") as c:
        assert c.api_key == "pk_explicit"


def test_public_methods_accept_timeout():
    methods = [
        "create_user",
        "ingest_dialog",
        "ingest_file",
        "get_event",
        "wait_until_ready",
        "search",
        "list_sources",
        "iter_sources",
        "get_session",
        "delete_by_session",
        "delete_by_file",
        "delete_by_tag",
    ]

    for method in methods:
        assert "timeout" in inspect.signature(getattr(Parcle, method)).parameters


def test_wait_until_ready_default_timeout_is_180():
    timeout = inspect.signature(Parcle.wait_until_ready).parameters["timeout"]
    assert timeout.default == 180.0


def test_ingest_waits_by_default():
    dialog_wait = inspect.signature(Parcle.ingest_dialog).parameters["wait"]
    file_wait = inspect.signature(Parcle.ingest_file).parameters["wait"]
    assert dialog_wait.default is True
    assert file_wait.default is True


# -- users ---------------------------------------------------------------------


@respx.mock
def test_create_user(client):
    route = respx.post(f"{BASE}/v1/users").mock(
        return_value=httpx.Response(
            200,
            json={
                "user_id": "user_123",
                "name": "Project A",
                "timezone": "America/New_York",
                "is_new": True,
            },
        )
    )
    user = client.create_user("user_123", name="Project A", timezone="America/New_York")
    assert user.user_id == "user_123"
    assert user.is_new is True
    assert user.timezone == "America/New_York"

    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "user_id": "user_123",
        "name": "Project A",
        "timezone": "America/New_York",
    }
    assert route.calls.last.request.headers["Authorization"] == "Bearer pk_test_123"


@respx.mock
def test_create_user_omits_none(client):
    route = respx.post(f"{BASE}/v1/users").mock(
        return_value=httpx.Response(200, json={"user_id": "u", "is_new": True})
    )
    client.create_user()
    assert json.loads(route.calls.last.request.content) == {}


@respx.mock
def test_create_user_timeout_override(client):
    route = respx.post(f"{BASE}/v1/users").mock(
        return_value=httpx.Response(200, json={"user_id": "u", "is_new": True})
    )
    client.create_user(timeout=45.0)
    assert route.calls.last.request.extensions["timeout"]["read"] == 45.0


# -- ingest dialog -------------------------------------------------------------


@respx.mock
def test_ingest_dialog(client):
    route = respx.post(f"{BASE}/v1/memories/ingest_dialog").mock(
        return_value=httpx.Response(
            200, json={"session_id": "sess_1", "event_id": "evt_1"}
        )
    )
    result = client.ingest_dialog(
        user_id="ada",
        messages=[
            {"role": "user", "content": "hi"},
            parcle.Message(role="assistant", content="hello", speaker="bot"),
        ],
        tag={"app": "x"},
        wait=False,
    )
    assert result.session_id == "sess_1"
    assert result.event_id == "evt_1"

    sent = json.loads(route.calls.last.request.content)
    assert sent["user_id"] == "ada"
    assert sent["session_id"] is None
    assert sent["tag"] == {"app": "x"}
    assert sent["messages"][0] == {"role": "user", "content": "hi"}
    assert sent["messages"][1] == {
        "role": "assistant",
        "content": "hello",
        "speaker": "bot",
    }


def test_ingest_dialog_bad_message(client):
    with pytest.raises(ParcleConfigError):
        client.ingest_dialog(user_id="ada", messages=[{"role": "user"}])


@respx.mock
def test_ingest_dialog_timeout_override(client):
    route = respx.post(f"{BASE}/v1/memories/ingest_dialog").mock(
        return_value=httpx.Response(
            200, json={"session_id": "sess_1", "event_id": "evt_1"}
        )
    )
    client.ingest_dialog(
        user_id="ada",
        messages=[{"role": "user", "content": "hi"}],
        timeout=60.0,
        wait=False,
    )
    assert route.calls.last.request.extensions["timeout"]["read"] == 60.0


@respx.mock
def test_ingest_dialog_waits_by_default(client):
    ingest_route = respx.post(f"{BASE}/v1/memories/ingest_dialog").mock(
        return_value=httpx.Response(
            200, json={"session_id": "sess_1", "event_id": "evt_1"}
        )
    )
    event_route = respx.post(f"{BASE}/v1/memories/events").mock(
        return_value=httpx.Response(
            200, json={"event_id": "evt_1", "status": "ready", "error": None}
        )
    )

    result = client.ingest_dialog(
        user_id="ada",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result.session_id == "sess_1"
    assert ingest_route.call_count == 1
    assert event_route.call_count == 1


@respx.mock
def test_ingest_dialog_wait_options(client, monkeypatch):
    captured = {}

    def fake_wait(user_id, event_id, *, poll_interval=2.0, timeout=180.0, **_):
        captured.update(
            user_id=user_id,
            event_id=event_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    monkeypatch.setattr(client, "wait_until_ready", fake_wait)
    respx.post(f"{BASE}/v1/memories/ingest_dialog").mock(
        return_value=httpx.Response(
            200, json={"session_id": "sess_1", "event_id": "evt_1"}
        )
    )

    client.ingest_dialog(
        user_id="ada",
        messages=[{"role": "user", "content": "hi"}],
        wait_timeout=45.0,
        wait_poll_interval=0.5,
    )

    assert captured == {
        "user_id": "ada",
        "event_id": "evt_1",
        "poll_interval": 0.5,
        "timeout": 45.0,
    }


# -- ingest file ---------------------------------------------------------------


@respx.mock
def test_ingest_file_from_path(client, tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# hello")
    route = respx.post(f"{BASE}/v1/memories/ingest_files").mock(
        return_value=httpx.Response(
            200, json={"file_id": "file_1", "event_id": "evt_2"}
        )
    )
    result = client.ingest_file("ada", str(f), tag={"source": "notes"}, wait=False)
    assert result.file_id == "file_1"

    req = route.calls.last.request
    assert b'name="user_id"' in req.content
    assert b"ada" in req.content
    assert b"notes.md" in req.content
    assert b"text/markdown" in req.content
    assert b'{"source": "notes"}' in req.content


@respx.mock
def test_ingest_file_from_tuple(client):
    respx.post(f"{BASE}/v1/memories/ingest_files").mock(
        return_value=httpx.Response(
            200, json={"file_id": "file_2", "event_id": "evt_3"}
        )
    )
    result = client.ingest_file("ada", ("data.txt", b"raw bytes"), wait=False)
    assert result.file_id == "file_2"


@respx.mock
def test_ingest_file_waits_by_default(client, tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# hello")
    ingest_route = respx.post(f"{BASE}/v1/memories/ingest_files").mock(
        return_value=httpx.Response(
            200, json={"file_id": "file_1", "event_id": "evt_2"}
        )
    )
    event_route = respx.post(f"{BASE}/v1/memories/events").mock(
        return_value=httpx.Response(
            200, json={"event_id": "evt_2", "status": "ready", "error": None}
        )
    )

    result = client.ingest_file("ada", str(f))

    assert result.file_id == "file_1"
    assert ingest_route.call_count == 1
    assert event_route.call_count == 1


def test_ingest_file_missing_path(client):
    with pytest.raises(ParcleConfigError):
        client.ingest_file("ada", "does-not-exist.pdf")


def test_ingest_file_raw_bytes_rejected(client):
    with pytest.raises(ParcleConfigError):
        client.ingest_file("ada", b"no filename here")


@respx.mock
def test_ingest_file_too_large_bytes_no_request(client):
    route = respx.post(f"{BASE}/v1/memories/ingest_files")
    oversized = b"x" * (MAX_FILE_SIZE_BYTES + 1)
    with pytest.raises(FileTooLargeError) as excinfo:
        client.ingest_file("ada", ("big.txt", oversized))
    assert excinfo.value.status_code == 413
    assert route.call_count == 0


@respx.mock
def test_ingest_file_too_large_path_no_request(client, tmp_path):
    route = respx.post(f"{BASE}/v1/memories/ingest_files")
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))
    with pytest.raises(FileTooLargeError):
        client.ingest_file("ada", str(f))
    assert route.call_count == 0


@respx.mock
def test_ingest_file_at_limit_is_allowed(client):
    route = respx.post(f"{BASE}/v1/memories/ingest_files").mock(
        return_value=httpx.Response(
            200, json={"file_id": "file_ok", "event_id": "evt_ok"}
        )
    )
    at_limit = b"x" * MAX_FILE_SIZE_BYTES
    result = client.ingest_file("ada", ("edge.txt", at_limit), wait=False)
    assert result.file_id == "file_ok"
    assert route.call_count == 1


def test_content_size_seekable_stream_is_measured_and_restored():
    import io

    from parcle.client import _content_size

    stream = io.BytesIO(b"x" * 1234)
    stream.seek(5)  # a non-zero starting position must be preserved
    assert _content_size(stream) == 1234
    assert stream.tell() == 5


def test_content_size_unseekable_stream_returns_none():
    from parcle.client import _content_size

    class _Unseekable:
        def seek(self, *args):
            raise OSError("not seekable")

        def tell(self):
            raise OSError("not seekable")

    # Unknown size means the client cannot block locally; the server's HTTP 413
    # remains the backstop for such streams.
    assert _content_size(_Unseekable()) is None


# -- events & polling ----------------------------------------------------------


@respx.mock
def test_get_event(client):
    respx.post(f"{BASE}/v1/memories/events").mock(
        return_value=httpx.Response(
            200, json={"event_id": "evt_1", "status": "processing", "error": None}
        )
    )
    event = client.get_event("ada", "evt_1")
    assert event.status == "processing"
    assert not event.is_terminal


@respx.mock
def test_get_event_timeout_override(client):
    route = respx.post(f"{BASE}/v1/memories/events").mock(
        return_value=httpx.Response(
            200, json={"event_id": "evt_1", "status": "processing", "error": None}
        )
    )
    client.get_event("ada", "evt_1", timeout=15.0)
    assert route.calls.last.request.extensions["timeout"]["read"] == 15.0


@respx.mock
def test_wait_until_ready(client, monkeypatch):
    monkeypatch.setattr(parcle.client.time, "sleep", lambda *_: None)
    responses = [
        httpx.Response(200, json={"event_id": "e", "status": "queued", "error": None}),
        httpx.Response(200, json={"event_id": "e", "status": "processing", "error": None}),
        httpx.Response(200, json={"event_id": "e", "status": "ready", "error": None}),
    ]
    respx.post(f"{BASE}/v1/memories/events").mock(side_effect=responses)
    event = client.wait_until_ready("ada", "e", poll_interval=0)
    assert event.is_ready


@respx.mock
def test_wait_until_ready_failed_raises(client, monkeypatch):
    monkeypatch.setattr(parcle.client.time, "sleep", lambda *_: None)
    respx.post(f"{BASE}/v1/memories/events").mock(
        return_value=httpx.Response(
            200, json={"event_id": "e", "status": "failed", "error": "boom"}
        )
    )
    with pytest.raises(parcle.ParcleAPIError) as exc:
        client.wait_until_ready("ada", "e")
    assert "boom" in str(exc.value)


# -- search --------------------------------------------------------------------


def _sse_event(name, data):
    """One SSE event block, mirroring the server's ``format_sse``."""
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


# An SSE keepalive comment line, emitted by the server between heartbeats.
_SSE_KEEPALIVE = ": keepalive\n\n"


def _sse_response(*chunks, status=200):
    """A streaming ``text/event-stream`` response built from raw SSE chunks."""
    return httpx.Response(
        status,
        text="".join(chunks),
        headers={"content-type": "text/event-stream"},
    )


@respx.mock
def test_search(client):
    route = respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=_sse_response(
            _sse_event(
                "final",
                {
                    "answer": "You are allergic to peanuts.",
                    "confidence": 0.92,
                    "citations": [{"type": "session", "id": "sess_1"}],
                },
            )
        )
    )
    result = client.search(
        "ada", "What should I avoid?", tag_filter={"app": "x"}, timezone="UTC"
    )
    assert result.answer == "You are allergic to peanuts."
    assert result.confidence == pytest.approx(0.92)
    assert result.citations[0].type == "session"
    assert result.citations[0].id == "sess_1"

    sent = json.loads(route.calls.last.request.content)
    assert sent["tag_filter"] == {"app": "x"}
    assert sent["timezone"] == "UTC"


@respx.mock
def test_search_ignores_keepalives_before_final(client):
    """Heartbeat comment lines preceding the answer are ignored."""
    respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=_sse_response(
            _SSE_KEEPALIVE,
            _SSE_KEEPALIVE,
            _sse_event("final", {"answer": "ok", "confidence": 0.5, "citations": []}),
        )
    )
    result = client.search("ada", "q")
    assert result.answer == "ok"


@respx.mock
def test_search_in_band_error(client):
    """A run-phase failure arrives as an in-band ``error`` event (HTTP 200)."""
    respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=_sse_response(
            _SSE_KEEPALIVE,
            _sse_event(
                "error",
                {
                    "code": "prompt_injection_detected",
                    "message": "blocked",
                    "request_id": "req_abc",
                },
            ),
        )
    )
    with pytest.raises(parcle.ParcleAPIError) as exc_info:
        client.search("ada", "q")
    err = exc_info.value
    assert err.code == "prompt_injection_detected"
    assert err.message == "blocked"
    assert err.request_id == "req_abc"


@respx.mock
def test_search_stream_ends_without_final(client):
    """A stream that closes before delivering a result is a connection error."""
    respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=_sse_response(_SSE_KEEPALIVE)
    )
    with pytest.raises(parcle.ParcleConnectionError):
        client.search("ada", "q")


@respx.mock
def test_search_prepare_error_keeps_status(client):
    """A pre-stream failure keeps its real HTTP status and typed exception."""
    respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=httpx.Response(
            404, json={"error": {"code": "user_not_found", "message": "no user"}}
        )
    )
    with pytest.raises(NotFoundError) as exc_info:
        client.search("ada", "q")
    assert exc_info.value.status_code == 404


@respx.mock
def test_search_omits_optional(client):
    route = respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=_sse_response(
            _sse_event("final", {"answer": "a", "confidence": 0.1, "citations": []})
        )
    )
    client.search("ada", "q")
    sent = json.loads(route.calls.last.request.content)
    assert "tag_filter" not in sent
    assert "timezone" not in sent


@respx.mock
def test_search_uses_retrieval_timeout():
    route = respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=_sse_response(
            _sse_event("final", {"answer": "a", "confidence": 0.1, "citations": []})
        )
    )
    with Parcle(api_key="pk_test", retrieval_timeout=180.0) as c:
        c.search("ada", "q")

    assert route.calls.last.request.extensions["timeout"]["read"] == 180.0


@respx.mock
def test_search_timeout_override():
    route = respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=_sse_response(
            _sse_event("final", {"answer": "a", "confidence": 0.1, "citations": []})
        )
    )
    with Parcle(api_key="pk_test", retrieval_timeout=180.0) as c:
        c.search("ada", "q", timeout=45.0)

    assert route.calls.last.request.extensions["timeout"]["read"] == 45.0


@respx.mock
def test_search_does_not_retry(monkeypatch):
    monkeypatch.setattr(parcle.client.time, "sleep", lambda *_: None)
    route = respx.post(f"{BASE}/v1/memories/search").mock(
        side_effect=[
            httpx.Response(
                503, json={"error": {"code": "unavailable", "message": "down"}}
            ),
            _sse_response(
                _sse_event("final", {"answer": "a", "confidence": 0.5, "citations": []})
            ),
        ]
    )
    with Parcle(api_key="pk_test", max_retries=2) as c:
        with pytest.raises(parcle.ServiceUnavailableError):
            c.search("ada", "q")

    assert route.call_count == 1


@respx.mock
def test_ingest_dialog_does_not_retry(monkeypatch):
    monkeypatch.setattr(parcle.client.time, "sleep", lambda *_: None)
    route = respx.post(f"{BASE}/v1/memories/ingest_dialog").mock(
        side_effect=[
            httpx.Response(
                503, json={"error": {"code": "unavailable", "message": "down"}}
            ),
            httpx.Response(200, json={"session_id": "sess_1", "event_id": "evt_1"}),
        ]
    )
    with Parcle(api_key="pk_test", max_retries=2) as c:
        with pytest.raises(parcle.ServiceUnavailableError):
            c.ingest_dialog("ada", [{"role": "user", "content": "hi"}])

    assert route.call_count == 1


# -- sources & sessions --------------------------------------------------------


@respx.mock
def test_list_sources(client):
    respx.post(f"{BASE}/v1/memories/sources").mock(
        return_value=httpx.Response(
            200,
            json={
                "sources": [
                    {
                        "id": "file_1",
                        "type": "file",
                        "name": "notes.md",
                        "tag": {"source": "x"},
                        "updated_at": "2026-06-04T09:30:00Z",
                    }
                ],
                "page": 1,
                "total_pages": 1,
                "total": 1,
            },
        )
    )
    page = client.list_sources("ada", type="file")
    assert len(page) == 1
    assert page.sources[0].name == "notes.md"
    assert list(page)[0].type == "file"


@respx.mock
def test_iter_sources_paginates(client):
    pages = {
        1: {
            "sources": [{"id": "s1", "type": "session"}],
            "page": 1,
            "total_pages": 2,
            "total": 2,
        },
        2: {
            "sources": [{"id": "s2", "type": "session"}],
            "page": 2,
            "total_pages": 2,
            "total": 2,
        },
    }

    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json=pages[body["page"]])

    respx.post(f"{BASE}/v1/memories/sources").mock(side_effect=handler)
    ids = [s.id for s in client.iter_sources("ada")]
    assert ids == ["s1", "s2"]


@respx.mock
def test_get_session(client):
    respx.post(f"{BASE}/v1/memories/sessions").mock(
        return_value=httpx.Response(
            200,
            json={
                "session_id": "sess_1",
                "tag": {"app": "x"},
                "messages": [
                    {"role": "user", "content": "hi", "speaker": "Alice"}
                ],
            },
        )
    )
    session = client.get_session("ada", "sess_1")
    assert session.session_id == "sess_1"
    assert session.messages[0].speaker == "Alice"


# -- deletion ------------------------------------------------------------------


@respx.mock
def test_delete_by_session(client):
    route = respx.delete(f"{BASE}/v1/memories/by_session").mock(
        return_value=httpx.Response(200, json={"deleted": True, "deleted_count": 3})
    )
    result = client.delete_by_session("ada", "sess_1")
    assert result.deleted is True
    assert result.deleted_count == 3
    assert json.loads(route.calls.last.request.content) == {
        "user_id": "ada",
        "session_id": "sess_1",
    }


@respx.mock
def test_delete_by_session_timeout_override(client):
    route = respx.delete(f"{BASE}/v1/memories/by_session").mock(
        return_value=httpx.Response(200, json={"deleted": True, "deleted_count": 3})
    )
    client.delete_by_session("ada", "sess_1", timeout=90.0)
    assert route.calls.last.request.extensions["timeout"]["read"] == 90.0


@respx.mock
def test_delete_by_tag(client):
    respx.delete(f"{BASE}/v1/memories/by_tag").mock(
        return_value=httpx.Response(200, json={"deleted": True, "deleted_count": 42})
    )
    result = client.delete_by_tag("ada", {"project": ["alpha", "beta"]})
    assert result.deleted_count == 42


def test_delete_by_tag_empty_rejected(client):
    with pytest.raises(ParcleConfigError):
        client.delete_by_tag("ada", {})


# -- error handling ------------------------------------------------------------


@respx.mock
def test_401_maps_to_auth_error(client):
    respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "code": "unauthorized",
                    "message": "bad key",
                    "request_id": "req_1",
                }
            },
        )
    )
    with pytest.raises(AuthenticationError) as exc:
        client.search("ada", "q")
    assert exc.value.code == "unauthorized"
    assert exc.value.request_id == "req_1"
    assert exc.value.status_code == 401


@respx.mock
def test_404_maps_to_not_found(client):
    respx.post(f"{BASE}/v1/memories/sessions").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"code": "session_not_found", "message": "nope"}},
        )
    )
    with pytest.raises(NotFoundError) as exc:
        client.get_session("ada", "missing")
    assert exc.value.code == "session_not_found"


@respx.mock
def test_422_maps_to_validation(client):
    respx.post(f"{BASE}/v1/memories/ingest_dialog").mock(
        return_value=httpx.Response(
            422,
            json={"error": {"code": "validation_failed", "message": "bad role"}},
        )
    )
    with pytest.raises(ValidationError):
        client.ingest_dialog("ada", [{"role": "user", "content": "x"}])


@respx.mock
def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(parcle.client.time, "sleep", lambda *_: None)
    responses = [
        httpx.Response(503, json={"error": {"code": "unavailable", "message": "down"}}),
        httpx.Response(200, json={"session_id": "sess_1", "tag": {}, "messages": []}),
    ]
    respx.post(f"{BASE}/v1/memories/sessions").mock(side_effect=responses)
    with Parcle(api_key="pk_test", max_retries=2) as c:
        result = c.get_session("ada", "sess_1")
    assert result.session_id == "sess_1"


@respx.mock
def test_read_retries_exhausted_raises(monkeypatch):
    monkeypatch.setattr(parcle.client.time, "sleep", lambda *_: None)
    route = respx.post(f"{BASE}/v1/memories/sessions").mock(
        return_value=httpx.Response(
            429, json={"error": {"code": "rate_limited", "message": "slow down"}}
        )
    )
    with Parcle(api_key="pk_test", max_retries=1) as c:
        with pytest.raises(RateLimitError):
            c.get_session("ada", "sess_1")

    assert route.call_count == 2
