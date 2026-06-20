# REST API Reference

Use the Parcle Memory API from any language over HTTP. All endpoints accept and
return JSON (except file uploads, which use multipart form data).

## Base URL

```
https://api.parcle.ai
```

## Authentication

Every request must include your API key as a Bearer token:

```
Authorization: Bearer pmem_...
```

You can also pass the key via the `PARCLE_API_KEY` environment variable in the
Python SDK — but when calling the REST API directly, use the header.

## Common Headers

| Header | Value |
|--------|-------|
| `Authorization` | `Bearer pmem_...` |
| `Content-Type` | `application/json` (for JSON endpoints) |
| `Accept` | `application/json` or `text/event-stream` (for search) |

---

## Endpoints

### Create or Update a User

Create a user namespace. Call this once per user before ingesting any data.

```
POST /v1/users
```

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | No | Your own user identifier. Omit to have one generated. |
| `name` | string | No | Display name. |
| `timezone` | string | No | IANA timezone (e.g. `"America/New_York"`). Used to interpret relative time in searches. Defaults to `"UTC"`. |

**Response**

```json
{
  "user_id": "ada",
  "name": "Ada Lovelace",
  "timezone": "UTC",
  "is_new": true
}
```

**curl**

```bash
curl -X POST https://api.parcle.ai/v1/users \
  -H "Authorization: Bearer pmem_..." \
  -H "Content-Type: application/json" \
  -d '{"user_id": "ada", "name": "Ada Lovelace"}'
```

---

### Ingest Dialog

Append conversation messages to a user's memory.

```
POST /v1/memories/ingest_dialog
```

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | |
| `messages` | array | Yes | Array of `{"role": "user"|"assistant", "content": "..."}`. Optional fields: `speaker`, `updated_at`. |
| `session_id` | string | No | Omit to start a new session; pass a previous value to append. |
| `tag` | object | No | Key-value metadata applied when creating a new session. |

**Response**

```json
{
  "session_id": "ses_abc123",
  "event_id": "evt_xyz789"
}
```

**curl**

```bash
curl -X POST https://api.parcle.ai/v1/memories/ingest_dialog \
  -H "Authorization: Bearer pmem_..." \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "ada",
    "messages": [
      {"role": "user", "content": "I'\''m allergic to peanuts."},
      {"role": "assistant", "content": "Got it — I'\''ll avoid peanuts."}
    ]
  }'
```

**JavaScript (fetch)**

```javascript
const res = await fetch("https://api.parcle.ai/v1/memories/ingest_dialog", {
  method: "POST",
  headers: {
    Authorization: "Bearer pmem_...",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    user_id: "ada",
    messages: [
      { role: "user", content: "I'm allergic to peanuts." },
      { role: "assistant", content: "Got it — I'll avoid peanuts." },
    ],
  }),
});
const { session_id, event_id } = await res.json();
```

---

### Ingest File

Upload a file (PDF, Markdown, text, DOCX, PPTX, XLSX, …) into a user's memory.

```
POST /v1/memories/ingest_files
```

This endpoint uses **multipart/form-data**, not JSON.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | form field | Yes | |
| `file` | file | Yes | The file to upload. Max 10 MB. |
| `updated_at` | form field | No | ISO 8601 timestamp for the file's content date. |
| `tag` | form field | No | JSON-encoded key-value metadata. |

**Response**

```json
{
  "file_id": "file_abc123",
  "event_id": "evt_xyz789"
}
```

**curl**

```bash
curl -X POST https://api.parcle.ai/v1/memories/ingest_files \
  -H "Authorization: Bearer pmem_..." \
  -F "user_id=ada" \
  -F "file=@diet-notes.pdf"
```

---

### Check Ingestion Status

Poll this endpoint after ingestion to know when content is searchable.

```
POST /v1/memories/events
```

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | |
| `event_id` | string | Yes | From `ingest_dialog` or `ingest_files` response. |

**Response**

```json
{
  "event_id": "evt_xyz789",
  "status": "ready",
  "error": null
}
```

`status` is one of: `"pending"`, `"processing"`, `"ready"`, `"failed"`.

Poll until `status` is `"ready"` or `"failed"`. A 2-second interval is
recommended.

---

### Search

Ask a natural-language question over a user's memory.

```
POST /v1/memories/search
```

This endpoint returns a **Server-Sent Events (SSE) stream**. Set `Accept:
text/event-stream`. The stream emits keepalive comments (`: ping`) until the
answer is ready, then a `final` event with the result.

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | |
| `query` | string | Yes | Natural-language question. |
| `tag_filter` | object | No | Only search sources matching these tags. |
| `timezone` | string | No | Override the user's default timezone for this query. |

**SSE response**

```
: ping

: ping

event: final
data: {"answer": "You're allergic to peanuts.", "confidence": 0.92, "citations": [{"type": "session", "id": "ses_abc123"}]}
```

If an error occurs after the stream has started, it arrives as an `error` event:

```
event: error
data: {"message": "Search failed.", "code": "internal_error"}
```

**Parsed result**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Synthesized answer grounded in the user's memory. |
| `confidence` | number | `0.0` – `1.0` confidence score. |
| `citations` | array | `[{"type": "session"|"file", "id": "..."}]` |

**curl**

```bash
curl -N -X POST https://api.parcle.ai/v1/memories/search \
  -H "Authorization: Bearer pmem_..." \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"user_id": "ada", "query": "What food should I avoid?"}'
```

**JavaScript (fetch)**

```javascript
const res = await fetch("https://api.parcle.ai/v1/memories/search", {
  method: "POST",
  headers: {
    Authorization: "Bearer pmem_...",
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  },
  body: JSON.stringify({
    user_id: "ada",
    query: "What food should I avoid?",
  }),
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  const lines = buffer.split("\n");
  buffer = lines.pop(); // keep incomplete line in buffer

  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const data = JSON.parse(line.slice(6));
      console.log(data); // { answer, confidence, citations }
    }
  }
}
```

---

### List Sources

List a user's ingested dialog sessions and files.

```
POST /v1/memories/sources
```

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | |
| `type` | string | No | `"session"` or `"file"`. Omit for both. |
| `tag_filter` | object | No | Filter by source tags. |
| `page` | integer | No | Page number (default `1`). |
| `limit` | integer | No | Results per page (default `50`). |
| `order` | string | No | `"desc"` (default) or `"asc"`. |

**Response**

```json
{
  "sources": [
    {
      "id": "ses_abc123",
      "type": "session",
      "updated_at": "2025-06-20T12:00:00Z",
      "tag": {"project": "diet-app"}
    }
  ],
  "page": 1,
  "total_pages": 3,
  "total": 142
}
```

---

### Get Session

Retrieve the original messages from a dialog session.

```
POST /v1/memories/sessions
```

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | |
| `session_id` | string | Yes | |

**Response**

```json
{
  "session_id": "ses_abc123",
  "messages": [
    {"role": "user", "content": "I'm allergic to peanuts."},
    {"role": "assistant", "content": "Got it — I'll avoid peanuts."}
  ],
  "tag": {"project": "diet-app"}
}
```

---

### Delete by Session

```
DELETE /v1/memories/by_session
```

**Request body**

```json
{"user_id": "ada", "session_id": "ses_abc123"}
```

**Response**

```json
{"deleted": true, "deleted_count": 1}
```

---

### Delete by File

```
DELETE /v1/memories/by_file
```

**Request body**

```json
{"user_id": "ada", "file_id": "file_abc123"}
```

**Response**

```json
{"deleted": true, "deleted_count": 1}
```

---

### Delete by Tag

Delete all sources whose tags match the filter.

```
DELETE /v1/memories/by_tag
```

**Request body**

```json
{"user_id": "ada", "tag_filter": {"project": "diet-app"}}
```

`tag_filter` must be non-empty.

**Response**

```json
{"deleted": true, "deleted_count": 5}
```

---

## Error Format

All errors return a JSON body:

```json
{
  "message": "A human-readable error message.",
  "code": "error_code",
  "request_id": "req_..."
}
```

Common HTTP status codes:

| Status | Meaning |
|--------|---------|
| 400 | Bad request (missing/invalid fields) |
| 401 | Invalid or missing API key |
| 413 | File exceeds 10 MB limit |
| 429 | Rate limited — retry after `Retry-After` header |
| 500 | Server error — safe to retry with backoff |
| 503 | Service unavailable — safe to retry with backoff |

## Rate Limiting

When rate-limited (HTTP 429), the response includes a `Retry-After` header
indicating how many seconds to wait. Implement exponential backoff for retries
on 429, 500, and 503 responses.
