<div align="center">

# Parcle

**Long-term memory for AI agents**

Ingest conversations and files, then ask questions in natural language and get
cited answers back. Give every user a private, persistent agent memory.

</div>

---

## Why Parcle?

LLMs forget everything between calls. Parcle gives every user a private memory you
can write to and search:

- 🧠 **Per-user memory** — scope everything to a `user_id`.
- 💬 **Ingest anything** — chat transcripts and files (PDF, Markdown, text, …) go in the same place.
- 🔎 **Ask, don't query** — search returns a synthesized **answer** with **citations**, not just raw chunks.

## Installation

```bash
pip install parcle
```

## Quickstart

```python
from parcle import Parcle

# Reads PARCLE_API_KEY from the environment if api_key is omitted.
client = Parcle(api_key="pk_live_...")

# 1. Write a conversation into a user's memory.
#    Ingestion is incremental: omit session_id to start a new session, then
#    pass the returned session_id back to append more turns to the same one.
result = client.ingest_dialog(
    user_id="ada",
    messages=[
        {"role": "user", "content": "I'm allergic to peanuts."},
        {"role": "assistant", "content": "Got it — I'll avoid peanuts in suggestions."},
    ],
)
client.ingest_dialog(
    user_id="ada",
    session_id=result.session_id,  # append to the same session
    messages=[
        {"role": "user", "content": "Also, I don't eat shellfish."},
    ],
)

# 2. ...or ingest a file (PDF, Markdown, text, …).
client.ingest_file(user_id="ada", file="diet-notes.pdf")

# 3. Ask a question. You get an answer with confidence and citations.
result = client.search(user_id="ada", query="What food should I avoid?")

print(result.answer)      # "You're allergic to peanuts, so avoid them."
print(result.confidence)  # 0.92
print(result.citations)   # [Citation(type="session", id="...")]
```