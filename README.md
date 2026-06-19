<div align="center">

# Parcle

**Sync for AI agents**

Parcle keeps the AI coding agents you use in sync. Three things travel with you:
**memory**, **skills**, and **conversations** — so switching agents doesn't mean starting over.

<p>
  <a href="https://pypi.org/project/parcle/"><img alt="PyPI" src="https://img.shields.io/pypi/v/parcle"></a>
  <a href="https://pypi.org/project/parcle/"><img alt="License" src="https://img.shields.io/pypi/l/parcle"></a>
</p>

</div>

---

## Why Parcle?

Your work is spread across agents — Claude Code, Codex, Cursor — and across
machines. Parcle is built to sync what matters between them:

- 🧠 **Memory** — a private, per-user long-term memory in the cloud. Write
  conversations and files into it, then **ask** instead of query and get a
  synthesized answer with citations.
- 🛠️ **Skills** — a local library of `SKILL.md` skill folders. `push` a skill
  from one agent and `pull` it into the others, so every agent shares the same
  capabilities.
- 💬 **Conversations** — collect every agent's chat transcript into one unified,
  local format you own.

**Memory** is a cloud API used through the `Parcle` client below. **Skills** and
**conversations** are local and multi-agent, driven by the `Sync` Python entry
point — no cloud, nothing leaves your machine.

## Installation

```bash
pip install parcle
```

## Memory — quickstart

```python
from parcle import Parcle

# Reads PARCLE_API_KEY from the environment if api_key is omitted.
client = Parcle(api_key="pk_live_...")

# 1. Write a conversation into a user's memory.
#    Ingestion is incremental: omit session_id to start a new session, then
#    pass the returned session_id back to append more turns to the same one.
dialog = client.ingest_dialog(
    user_id="ada",
    messages=[
        {"role": "user", "content": "I'm allergic to peanuts."},
        {"role": "assistant", "content": "Got it — I'll avoid peanuts in suggestions."},
    ],
)
client.ingest_dialog(
    user_id="ada",
    session_id=dialog.session_id,  # append to the same session
    messages=[
        {"role": "user", "content": "Also, I don't eat shellfish."},
    ],
)

# 2. ...or ingest a file (PDF, Markdown, text, …).
client.ingest_file(user_id="ada", file="diet-notes.pdf")

# Ingestion waits until content is searchable by default. Pass wait=False if you want to enqueue writes and call wait_until_ready(...) yourself.

# 3. Ask a question. You get an answer with confidence and citations.
result = client.search(user_id="ada", query="What food should I avoid?")

print(result.answer)      # "You're allergic to peanuts, so avoid them."
print(result.confidence)  # 0.92
print(result.citations)   # [Citation(type='session', id='...')]
```

## Skills & conversations — local sync

`Sync` is the local entry point — no API key, no cloud. Everything lives under
`~/.parcle` (override with `PARCLE_HOME`). Constructing `Sync()` starts a
background daemon that backfills once, then watches your agents

```python
from parcle import Sync
from parcle.sync import get_agent

sync = Sync()                  # starts the background daemon
sync.wait_until_ready()        # wait for the initial backfill

# Skills: see who holds what, move one from claude to codex, then clean up.
sync.inventory()                              # {agent_type: [Skill, ...]}
sync.skills.push("my-skill", get_agent("claude"))   # agent -> library
sync.skills.pull("my-skill", get_agent("codex"))    # library -> one agent (force=True to overwrite)
sync.list_skills()                            # what's in the library
sync.skills.uninstall("my-skill", get_agent("codex"))
sync.skills.remove("my-skill")

# Conversations: collected incrementally; read them back to hand off between agents.
convos = sync.list_conversations()            # [{agent_type, session_id, title, turn_count}, ...]
for turn in sync.get_conversation("claude", convos[0]["session_id"]):
    print(f"#{turn.seq} {turn.render()}")

# Keep the background daemon alive so it goes on syncing
try:
    while True:
        time.sleep(60)
        print(sync.status())
except KeyboardInterrupt:
    sync.stop()

sync.stop()                    
```

Supported agents: **Claude Code** & **Codex** (skills + conversations), **Cursor**
& **OpenClaw** (skills).
