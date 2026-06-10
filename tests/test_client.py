"""Tests for the Parcle client, mocking HTTP with respx."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

import parcle
from parcle import (
    AuthenticationError,
    NotFoundError,
    Parcle,
    ParcleConfigError,
    RateLimitError,
    ValidationError,
)

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
    result = client.ingest_file("ada", str(f), tag={"source": "notes"})
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
    result = client.ingest_file("ada", ("data.txt", b"raw bytes"))
    assert result.file_id == "file_2"


def test_ingest_file_missing_path(client):
    with pytest.raises(ParcleConfigError):
        client.ingest_file("ada", "does-not-exist.pdf")


def test_ingest_file_raw_bytes_rejected(client):
    with pytest.raises(ParcleConfigError):
        client.ingest_file("ada", b"no filename here")


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


@respx.mock
def test_search(client):
    route = respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "You are allergic to peanuts.",
                "confidence": 0.92,
                "citations": [{"type": "session", "id": "sess_1"}],
            },
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
def test_search_omits_optional(client):
    route = respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=httpx.Response(
            200, json={"answer": "a", "confidence": 0.1, "citations": []}
        )
    )
    client.search("ada", "q")
    sent = json.loads(route.calls.last.request.content)
    assert "tag_filter" not in sent
    assert "timezone" not in sent


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
        httpx.Response(200, json={"answer": "a", "confidence": 0.5, "citations": []}),
    ]
    respx.post(f"{BASE}/v1/memories/search").mock(side_effect=responses)
    with Parcle(api_key="pk_test", max_retries=2) as c:
        result = c.search("ada", "q")
    assert result.answer == "a"


@respx.mock
def test_retries_exhausted_raises(monkeypatch):
    monkeypatch.setattr(parcle.client.time, "sleep", lambda *_: None)
    respx.post(f"{BASE}/v1/memories/search").mock(
        return_value=httpx.Response(
            429, json={"error": {"code": "rate_limited", "message": "slow down"}}
        )
    )
    with Parcle(api_key="pk_test", max_retries=1) as c:
        with pytest.raises(RateLimitError):
            c.search("ada", "q")
