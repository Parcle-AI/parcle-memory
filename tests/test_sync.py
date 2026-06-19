"""Tests for the local, multi-agent sync features (skills + conversations)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from parcle.sync import (
    Agent,
    ConversationError,
    ConversationStore,
    Event,
    EventLog,
    SkillError,
    SkillLibrary,
    Sync,
    Turn,
    Watcher,
    detect_agents,
    discover_sessions,
    get_agent,
    known_agents,
    parse_frontmatter,
    parse_records,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point HOME and PARCLE_HOME at temp dirs so nothing touches the real machine."""
    home = tmp_path / "home"
    home.mkdir()
    parcle_home = tmp_path / "parcle"
    monkeypatch.setenv("PARCLE_HOME", str(parcle_home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return home


def _write_skill(directory: Path, name: str, description: str = "", body: str = "payload") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    desc = f"description: {description}\n" if description else ""
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\n{desc}---\n# {name}\n", encoding="utf-8"
    )
    (directory / "extra.txt").write_text(body, encoding="utf-8")


# -- frontmatter ---------------------------------------------------------------


def test_parse_frontmatter_scalars_and_block():
    text = (
        "---\n"
        "name: my-skill\n"
        'description: "quoted desc"\n'
        "notes: |\n"
        "  line one\n"
        "  line two\n"
        "---\n"
        "body\n"
    )
    fm = parse_frontmatter(text)
    assert fm["name"] == "my-skill"
    assert fm["description"] == "quoted desc"
    assert fm["notes"] == "line one\nline two"


def test_parse_frontmatter_missing_returns_empty():
    assert parse_frontmatter("no frontmatter here") == {}


# -- agents --------------------------------------------------------------------


def test_known_agents_and_lookup(env):
    types = {a.type for a in known_agents()}
    assert {"claude", "codex", "cursor", "openclaw"} <= types
    assert get_agent("claude").display_name == "Claude Code"
    with pytest.raises(KeyError):
        get_agent("nope")


def test_codex_and_openclaw_support_skills(env):
    # Codex syncs both skills and conversations; OpenClaw is skills-only.
    assert get_agent("codex").supports_skills
    assert get_agent("codex").supports_conversations
    assert get_agent("openclaw").supports_skills
    assert not get_agent("openclaw").supports_conversations


def test_push_and_pull_with_codex_and_openclaw(env):
    # A skill captured from Codex can be pulled onto OpenClaw, and vice versa.
    _write_skill(env / ".codex" / "skills" / "tool", "tool")
    library = SkillLibrary()
    library.push("tool", get_agent("codex"))
    library.pull("tool", get_agent("openclaw"))
    assert (env / ".openclaw" / "skills" / "tool" / "SKILL.md").is_file()


def test_codex_home_override_moves_both_dirs(tmp_path, monkeypatch):
    custom = tmp_path / "codexhome"
    monkeypatch.setenv("CODEX_HOME", str(custom))
    codex = get_agent("codex")
    assert codex.skills_dir == custom / "skills"
    assert codex.transcripts_root == custom / "sessions"


def test_detect_agents_only_present(env):
    assert detect_agents() == []  # nothing installed in the fresh temp home
    _write_skill(env / ".claude" / "skills" / "s", "s")
    assert {a.type for a in detect_agents()} == {"claude"}


# -- skills: push / pull -------------------------------------------------------


def test_push_then_pull_across_agents(env):
    _write_skill(env / ".claude" / "skills" / "greeter", "greeter", "hi")
    (env / ".cursor" / "skills").mkdir(parents=True)
    library = SkillLibrary()

    pushed = library.push("greeter", get_agent("claude"))
    assert pushed.name == "greeter"
    assert (library.root / "greeter" / "SKILL.md").is_file()
    assert [s.key for s in library.list()] == ["greeter"]

    library.pull("greeter", get_agent("cursor"))
    assert (env / ".cursor" / "skills" / "greeter" / "SKILL.md").is_file()


def test_pull_all_installs_into_every_skill_agent(env):
    _write_skill(env / ".claude" / "skills" / "g", "g")
    (env / ".cursor" / "skills").mkdir(parents=True)
    library = SkillLibrary()
    library.push("g", get_agent("claude"))

    # claude already holds 'g' (it was pushed from there), so distributing to
    # every agent needs force=True under the symmetric overwrite-protection.
    installed = library.pull_all("g", force=True)
    assert set(installed) == {"claude", "cursor"}
    assert (env / ".cursor" / "skills" / "g" / "SKILL.md").is_file()


def test_push_missing_skill_raises(env):
    (env / ".claude" / "skills").mkdir(parents=True)
    with pytest.raises(SkillError):
        SkillLibrary().push("ghost", get_agent("claude"))


def test_push_without_skill_md_raises(env):
    bad = env / ".claude" / "skills" / "bad"
    bad.mkdir(parents=True)
    (bad / "readme.txt").write_text("not a skill", encoding="utf-8")
    with pytest.raises(SkillError):
        SkillLibrary().push("bad", get_agent("claude"))


def test_pull_to_agent_without_skills_dir_raises(env):
    _write_skill(env / ".claude" / "skills" / "g", "g")
    library = SkillLibrary()
    library.push("g", get_agent("claude"))
    no_skills = Agent(type="x", display_name="X", skills_dir=None)
    with pytest.raises(SkillError):
        library.pull("g", no_skills)


def test_push_fails_on_name_conflict(env):
    _write_skill(env / ".claude" / "skills" / "g", "g", body="v1")
    _write_skill(env / ".cursor" / "skills" / "g", "g", body="v2")
    library = SkillLibrary()
    library.push("g", get_agent("claude"))  # first push wins
    with pytest.raises(SkillError):
        library.push("g", get_agent("cursor"))  # same name -> conflict
    # The library was left untouched by the failed push.
    assert "v1" in (library.root / "g" / "extra.txt").read_text(encoding="utf-8")


def test_push_force_overwrites_on_conflict(env):
    _write_skill(env / ".claude" / "skills" / "g", "g", body="v1")
    _write_skill(env / ".cursor" / "skills" / "g", "g", body="v2")
    library = SkillLibrary()
    library.push("g", get_agent("claude"))
    library.push("g", get_agent("cursor"), force=True)  # explicit overwrite
    assert "v2" in (library.root / "g" / "extra.txt").read_text(encoding="utf-8")


def test_pull_fails_on_existing_agent_skill(env):
    # The agent already has its own edited copy; a plain pull must not clobber it.
    _write_skill(env / ".claude" / "skills" / "g", "g", body="lib")
    _write_skill(env / ".cursor" / "skills" / "g", "g", body="agent-edit")
    library = SkillLibrary()
    library.push("g", get_agent("claude"))
    with pytest.raises(SkillError):
        library.pull("g", get_agent("cursor"))
    # The agent's copy was left untouched by the refused pull.
    assert "agent-edit" in (env / ".cursor" / "skills" / "g" / "extra.txt").read_text(encoding="utf-8")


def test_pull_force_overwrites_agent_skill(env):
    _write_skill(env / ".claude" / "skills" / "g", "g", body="lib")
    _write_skill(env / ".cursor" / "skills" / "g", "g", body="agent-edit")
    library = SkillLibrary()
    library.push("g", get_agent("claude"))
    library.pull("g", get_agent("cursor"), force=True)
    assert "lib" in (env / ".cursor" / "skills" / "g" / "extra.txt").read_text(encoding="utf-8")


def test_pull_into_fresh_agent_still_works(env):
    # No existing copy on the agent -> pull installs without needing force.
    _write_skill(env / ".claude" / "skills" / "g", "g")
    (env / ".cursor" / "skills").mkdir(parents=True)
    library = SkillLibrary()
    library.push("g", get_agent("claude"))
    library.pull("g", get_agent("cursor"))
    assert (env / ".cursor" / "skills" / "g" / "SKILL.md").is_file()


def test_pull_all_fails_when_an_agent_already_holds_it(env):
    _write_skill(env / ".claude" / "skills" / "g", "g", body="lib")
    _write_skill(env / ".cursor" / "skills" / "g", "g", body="agent-edit")
    library = SkillLibrary()
    library.push("g", get_agent("claude"))
    with pytest.raises(SkillError):
        library.pull_all("g")
    library.pull_all("g", force=True)  # force overwrites every agent's copy
    assert "lib" in (env / ".cursor" / "skills" / "g" / "extra.txt").read_text(encoding="utf-8")


# -- skills: remove / uninstall ------------------------------------------------


def test_remove_deletes_from_library(env):
    _write_skill(env / ".claude" / "skills" / "g", "g")
    library = SkillLibrary()
    library.push("g", get_agent("claude"))
    assert [s.key for s in library.list()] == ["g"]
    library.remove("g")
    assert library.list() == []


def test_remove_missing_skill_raises(env):
    with pytest.raises(SkillError):
        SkillLibrary().remove("ghost")


def test_uninstall_deletes_from_agent(env):
    _write_skill(env / ".claude" / "skills" / "g", "g")
    library = SkillLibrary()
    library.uninstall("g", get_agent("claude"))
    assert not (env / ".claude" / "skills" / "g").exists()


def test_uninstall_missing_skill_raises(env):
    (env / ".claude" / "skills").mkdir(parents=True)
    with pytest.raises(SkillError):
        SkillLibrary().uninstall("ghost", get_agent("claude"))


# -- skills: drift detection ---------------------------------------------------


def test_drift_reports_missing_and_diverged(env):
    # claude/a is identical to the library; claude/b is not in the library;
    # cursor/a was edited on the agent side, so it has diverged.
    _write_skill(env / ".claude" / "skills" / "a", "a", body="shared")
    _write_skill(env / ".claude" / "skills" / "b", "b")
    _write_skill(env / ".cursor" / "skills" / "a", "a", body="shared")
    library = SkillLibrary()
    library.push("a", get_agent("claude"))  # library copy == claude/a == cursor/a

    # No drift yet for 'a' (both agents match the library); 'b' is missing.
    drift = library.drift()
    assert drift["missing_from_library"] == ["claude/b"]
    assert drift["diverged"] == []

    # Edit the agent-side copy of 'a' on cursor -> it diverges from the library.
    _write_skill(env / ".cursor" / "skills" / "a", "a", body="changed")
    drift = library.drift()
    assert drift["missing_from_library"] == ["claude/b"]
    assert drift["diverged"] == ["cursor/a"]


def test_status_reports_drift_fields(env):
    _write_skill(env / ".claude" / "skills" / "a", "a", body="shared")
    _write_skill(env / ".claude" / "skills" / "b", "b")
    with Sync() as sync:
        assert sync.wait_until_ready(timeout=10)
        sync.skills.push("a", get_agent("claude"))
        sync._watcher.sync_skills()  # refresh the snapshot after the manual push
        skills = sync.status()["skills"]
    assert skills["missing_from_library"] == ["claude/b"]
    assert "not_in_library" not in skills


# -- conversations: parsing ----------------------------------------------------


def test_parse_records_claude_blocks_and_title():
    records = [
        {"type": "user", "timestamp": "t1", "message": {"role": "user", "content": "hi"}},
        {
            "type": "assistant",
            "timestamp": "t2",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}},
                ],
            },
        },
        {"type": "ai-title", "aiTitle": "A title"},
        {"type": "queue-operation"},  # contentless -> skipped
    ]
    result = parse_records("claude", records, start_seq=0)
    assert [t.seq for t in result.turns] == [1, 2]
    assert result.turns[0].role == "user"
    assert result.turns[1].blocks[1].type == "tool_call"
    assert result.turns[1].blocks[1].name == "Bash"
    assert result.title == "A title"


def test_parse_records_codex_dedup_and_roles():
    records = [
        {"timestamp": "t1", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]}},
        {"timestamp": "t2", "payload": {"type": "user_message", "message": "q"}},  # duplicate -> skipped
        {"timestamp": "t3", "payload": {"type": "function_call", "name": "shell", "arguments": "{\"a\":1}"}},
        {"timestamp": "t4", "payload": {"type": "token_count"}},  # telemetry -> skipped
    ]
    result = parse_records("codex", records, start_seq=5)
    assert [t.seq for t in result.turns] == [6, 7]
    assert result.turns[0].role == "user"
    assert result.turns[1].blocks[0].type == "tool_call"
    assert result.turns[1].blocks[0].tool_input == {"a": 1}  # JSON string decoded


def test_parse_records_continues_seq_numbering():
    rec = [{"type": "user", "timestamp": "t", "message": {"role": "user", "content": "x"}}]
    assert parse_records("claude", rec, start_seq=10).turns[0].seq == 11


def test_parse_records_unknown_format_raises():
    with pytest.raises(ValueError):
        parse_records("bogus", [], start_seq=0)


# -- conversations: discovery + incremental collection -------------------------


def _write_claude_session(home: Path, project: str, session: str, lines: list[dict]) -> Path:
    path = home / ".claude" / "projects" / project / f"{session}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def test_discover_claude_sessions(env):
    _write_claude_session(env, "proj", "s1", [{"type": "user", "message": {"role": "user", "content": "hi"}}])
    sessions = discover_sessions(get_agent("claude"))
    assert len(sessions) == 1
    assert sessions[0].session_id == "s1"
    assert sessions[0].project == "proj"


def test_collect_is_incremental(env):
    path = _write_claude_session(
        env, "p", "s",
        [{"type": "user", "timestamp": "t1", "message": {"role": "user", "content": "one"}}],
    )
    store = ConversationStore()
    [first] = store.collect_agent(get_agent("claude"))
    assert first.new_turns == 1 and first.turn_count == 1

    # No change -> nothing new on a second pass.
    [again] = store.collect_agent(get_agent("claude"))
    assert again.new_turns == 0 and again.turn_count == 1

    # Append a turn -> only the new one is collected, seq continues.
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "assistant", "timestamp": "t2", "message": {"role": "assistant", "content": "two"}}) + "\n")
    [third] = store.collect_agent(get_agent("claude"))
    assert third.new_turns == 1 and third.turn_count == 2

    out = (store.root / "claude" / "s.jsonl").read_text(encoding="utf-8").strip().splitlines()
    seqs = [json.loads(line)["seq"] for line in out]
    assert seqs == [1, 2]


def test_collect_full_resets(env):
    _write_claude_session(
        env, "p", "s",
        [{"type": "user", "timestamp": "t1", "message": {"role": "user", "content": "one"}}],
    )
    store = ConversationStore()
    store.collect_agent(get_agent("claude"))
    [full] = store.collect_agent(get_agent("claude"), full=True)
    assert full.turn_count == 1  # re-collected from scratch, not appended to 2
    out = (store.root / "claude" / "s.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(out) == 1


# -- Sync facade ---------------------------------------------------------------


def test_sync_facade_uses_parcle_home(env, tmp_path, monkeypatch):
    sync = Sync(autostart=False)
    assert sync.skills.root == tmp_path / "parcle" / "skills"
    assert sync.conversations.root == tmp_path / "parcle" / "conversations"


def test_sync_facade_home_override(tmp_path):
    custom = tmp_path / "custom"
    sync = Sync(home=custom, autostart=False)
    assert sync.home == custom
    assert sync.skills.root == custom / "skills"
    assert sync.conversations.root == custom / "conversations"


def test_inventory_reports_per_agent_holdings(env):
    _write_skill(env / ".claude" / "skills" / "a", "a")
    _write_skill(env / ".claude" / "skills" / "b", "b")
    _write_skill(env / ".cursor" / "skills" / "a", "a")
    inv = Sync(autostart=False).inventory()
    assert {k for k in inv} == {"claude", "cursor"}
    assert {s.key for s in inv["claude"]} == {"a", "b"}
    assert {s.key for s in inv["cursor"]} == {"a"}


# -- watcher -------------------------------------------------------------------


def test_watcher_initial_sync_tracks_inventory_and_collects(env):
    _write_skill(env / ".claude" / "skills" / "g", "g")
    _write_claude_session(
        env, "p", "s",
        [{"type": "user", "timestamp": "t", "message": {"role": "user", "content": "hi"}}],
    )
    watcher = Watcher(Sync(autostart=False))
    events = watcher.initial_sync()
    kinds = {(e.kind, e.action) for e in events}
    # Watching tracks inventory; it never writes to the library.
    assert ("skill", "agent_added") in kinds
    assert ("conversation", "collected") in kinds
    assert not any(e.kind == "skill" and e.action in ("added", "updated") for e in events)
    assert watcher.sync.skills.list() == []  # library untouched by watching


def test_watcher_reports_inventory_deltas(env):
    _write_skill(env / ".claude" / "skills" / "a", "a")
    watcher = Watcher(Sync(autostart=False), conversations=False)
    watcher.sync_skills()  # establishes the baseline inventory

    _write_skill(env / ".claude" / "skills" / "b", "b")
    events = watcher.sync_skills()
    actions = {(e.action, e.name) for e in events}
    assert ("agent_added", "b") in actions

    (env / ".claude" / "skills" / "a" / "SKILL.md").unlink()
    import shutil as _sh
    _sh.rmtree(env / ".claude" / "skills" / "a")
    events = watcher.sync_skills()
    assert ("agent_removed", "a") in {(e.action, e.name) for e in events}


def test_watcher_react_only_fires_for_relevant_roots(env):
    _write_skill(env / ".claude" / "skills" / "g", "g")
    watcher = Watcher(Sync(autostart=False))
    watcher.initial_sync()  # consume the baseline

    # A path unrelated to any watched root -> no work.
    assert watcher.react([env / "unrelated" / "x"]) == []

    # A change under the skills root -> skill sync runs.
    _write_skill(env / ".cursor" / "skills" / "g", "g")
    cursor_root = get_agent("cursor").skills_dir
    events = watcher.react([cursor_root / "g" / "SKILL.md"])
    assert any(e.kind == "skill" for e in events)


def test_watcher_has_no_roots_without_agents(env):
    # Fresh temp home: nothing installed, so there is nothing to watch.
    watcher = Watcher(Sync(autostart=False))
    assert watcher.roots() == []


# -- daemon: readiness, status, lifecycle --------------------------------------


def test_readiness_is_independent_per_side(env):
    sync = Sync(autostart=False)
    assert sync.skills_ready() and sync.conversations_ready()
    # Marking one side busy must not affect the other.
    with sync._state.busy("skills"):
        assert not sync.skills_ready()
        assert sync.conversations_ready()
    assert sync.skills_ready()  # released


def test_wait_until_ready_times_out_while_busy(env):
    sync = Sync(autostart=False)
    sync._state.mark_active("conversations")
    try:
        assert sync.wait_until_ready("conversations", timeout=0.1) is False
        assert sync.wait_until_ready("skills", timeout=0.1) is True
    finally:
        sync._state.mark_idle("conversations")


def test_autostart_daemon_backfills_then_is_ready(env):
    _write_skill(env / ".claude" / "skills" / "g", "g")
    _write_claude_session(
        env, "p", "s",
        [{"type": "user", "timestamp": "t", "message": {"role": "user", "content": "hi"}}],
    )
    with Sync() as sync:  # autostart
        assert sync.wait_until_ready(timeout=10) is True
        assert sync.is_running()
        # Backfill tracked the agent's skill (inventory) and collected the
        # conversation. The library is NOT auto-populated — that needs a push.
        assert sync.skills.list() == []
        assert {s.key for s in sync.inventory()["claude"]} == {"g"}
        status = sync.status()
        assert status["ready"] == {"skills": True, "conversations": True}
        assert status["skills"]["library"] == 0
        assert status["skills"]["by_agent"]["claude"] == ["g"]
        assert status["conversations"]["turns"] == 1
    assert not sync.is_running()  # context exit stopped the daemon


def test_daemon_tracks_inventory_changes_after_start(env):
    # A skill added after the daemon is live shows up in the live inventory
    # (events + the periodic safety reconcile catch it even in the startup gap).
    # The library stays empty — watching never writes to it.
    _write_skill(env / ".claude" / "skills" / "g", "g")
    with Sync(reconcile_interval=0.2) as sync:
        assert sync.wait_until_ready(timeout=10)
        assert {s.key for s in sync.inventory().get("claude", [])} == {"g"}
        _write_skill(env / ".cursor" / "skills" / "h", "h")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if any(s.key == "h" for s in sync.inventory().get("cursor", [])):
                break
            time.sleep(0.2)
        assert {s.key for s in sync.inventory()["cursor"]} == {"h"}
        assert sync.skills.list() == []  # library untouched


def test_state_is_inherited_across_processes(env, tmp_path):
    _write_skill(env / ".claude" / "skills" / "g", "g")
    with Sync() as sync:
        assert sync.wait_until_ready(timeout=10)
    # A fresh instance over the same home inherits the persisted snapshot.
    reborn = Sync(autostart=False)
    status = reborn.status()
    assert status["skills"]["by_agent"]["claude"] == ["g"]
    assert status["last_event_at"] is not None


def test_conversation_collection_resumes_across_instances(env):
    path = _write_claude_session(
        env, "p", "s",
        [{"type": "user", "timestamp": "t1", "message": {"role": "user", "content": "one"}}],
    )
    Sync(autostart=False).conversations.collect_all()  # first process

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "assistant", "timestamp": "t2", "message": {"role": "assistant", "content": "two"}}) + "\n")

    # A new instance resumes from the persisted cursor and only appends the new turn.
    [result] = Sync(autostart=False).conversations.collect_agent(get_agent("claude"))
    assert result.new_turns == 1 and result.turn_count == 2
    store = Sync(autostart=False).conversations
    seqs = [
        json.loads(line)["seq"]
        for line in (store.root / "claude" / "s.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert seqs == [1, 2]


# -- conversations: read-back (retrieval) --------------------------------------


def _collect_two_sessions(env):
    _write_claude_session(
        env, "p", "s1",
        [
            {"type": "user", "timestamp": "t1", "message": {"role": "user", "content": "hi"}},
            {"type": "ai-title", "aiTitle": "Session One"},
            {"type": "assistant", "timestamp": "t2", "message": {"role": "assistant", "content": "hello"}},
        ],
    )
    _write_claude_session(
        env, "p", "s2",
        [{"type": "user", "timestamp": "t3", "message": {"role": "user", "content": "yo"}}],
    )
    Sync(autostart=False).conversations.collect_all()


def test_list_conversations_reports_each_session(env):
    _collect_two_sessions(env)
    listed = Sync(autostart=False).list_conversations()
    by_id = {c["session_id"]: c for c in listed}
    assert set(by_id) == {"s1", "s2"}
    assert by_id["s1"]["agent_type"] == "claude"
    assert by_id["s1"]["turn_count"] == 2
    assert by_id["s1"]["title"] == "Session One"


def test_list_conversations_empty_when_nothing_collected(env):
    assert Sync(autostart=False).list_conversations() == []


def test_get_conversation_returns_turns_in_order(env):
    _collect_two_sessions(env)
    turns = Sync(autostart=False).get_conversation("claude", "s1")
    assert [t.seq for t in turns] == [1, 2]
    assert turns[0].role == "user"
    assert turns[0].blocks[0].text == "hi"
    assert turns[1].role == "assistant"


def test_get_conversation_unknown_session_raises(env):
    with pytest.raises(ConversationError):
        Sync(autostart=False).get_conversation("claude", "nope")


def test_get_turn_returns_single_message(env):
    _collect_two_sessions(env)
    turn = Sync(autostart=False).get_turn("claude", "s1", 2)
    assert turn.seq == 2
    assert turn.role == "assistant"
    assert turn.blocks[0].text == "hello"


def test_get_turn_unknown_seq_raises(env):
    _collect_two_sessions(env)
    with pytest.raises(ConversationError):
        Sync(autostart=False).get_turn("claude", "s1", 99)


# -- conversations: rendering --------------------------------------------------


def test_turn_render_text_and_tool_blocks():
    records = [
        {"type": "assistant", "timestamp": "t", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "running it"},
            {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}},
        ]}},
        {"type": "user", "timestamp": "t", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "file.txt", "is_error": False},
        ]}},
    ]
    turns = parse_records("claude", records, start_seq=0).turns
    rendered = turns[0].render()
    assert rendered.startswith("[assistant] ")
    assert "running it" in rendered
    assert "[tool_call Bash]" in rendered
    # A turn whose only block is a tool_result (no .text) still renders, not blank.
    assert turns[1].render().startswith("[user] [tool_result]")
    assert "file.txt" in turns[1].render()


def test_turn_render_empty_blocks_is_not_blank():
    turn = Turn(seq=1, role="user", timestamp=None, blocks=())
    assert turn.render() == "[user] (empty)"


def test_render_conversation_joins_turns(env):
    _collect_two_sessions(env)
    text = Sync(autostart=False).render_conversation("claude", "s1")
    assert "[user] hi" in text
    assert "[assistant] hello" in text
    assert text.count("\n\n") == 1  # two turns -> exactly one separator


def test_render_conversation_unknown_session_raises(env):
    with pytest.raises(ConversationError):
        Sync(autostart=False).render_conversation("claude", "nope")


# -- activity event stream -----------------------------------------------------


def test_event_log_records_and_reads_back(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.record("skill", "push", name="g", agent_type="claude")
    log.record("skill", "pull", status="error", name="h", agent_type="cursor", error="boom")
    events = log.read()
    assert [(e.category, e.action, e.status) for e in events] == [
        ("skill", "push", "ok"),
        ("skill", "pull", "error"),
    ]
    assert events[1].error == "boom"
    # Filters and limit.
    assert [e.action for e in log.read(status="error")] == ["pull"]
    assert [e.action for e in log.read(category="skill", limit=1)] == ["pull"]
    assert log.read(limit=0) == []  # limit=0 means none, not all


def test_event_log_without_path_is_noop_but_returns_event(tmp_path):
    log = EventLog(None)
    event = log.record("daemon", "start")
    assert event.category == "daemon" and event.action == "start"
    assert log.read() == []


def test_event_stream_records_daemon_start_and_stop(env):
    sync = Sync(autostart=False)
    sync.start()
    sync.stop()
    actions = [(e.category, e.action, e.status) for e in sync.events()]
    assert ("daemon", "start", "ok") in actions
    assert ("daemon", "stop", "ok") in actions


def test_event_stream_records_skill_push_and_pull(env):
    _write_skill(env / ".claude" / "skills" / "g", "g")
    (env / ".cursor" / "skills").mkdir(parents=True)
    sync = Sync(autostart=False)
    sync.skills.push("g", get_agent("claude"))
    sync.skills.pull("g", get_agent("cursor"))
    events = sync.events(category="skill")
    assert any(
        e.action == "push" and e.status == "ok" and e.name == "g" and e.agent_type == "claude"
        for e in events
    )
    assert any(
        e.action == "pull" and e.status == "ok" and e.name == "g" and e.agent_type == "cursor"
        for e in events
    )


def test_event_stream_records_skill_failure(env):
    (env / ".claude" / "skills").mkdir(parents=True)
    sync = Sync(autostart=False)
    with pytest.raises(SkillError):
        sync.skills.push("ghost", get_agent("claude"))
    errors = sync.events(status="error")
    assert any(e.category == "skill" and e.action == "push" and e.error for e in errors)


def test_event_stream_records_conversation_reads(env):
    _write_claude_session(
        env, "p", "s1",
        [
            {"type": "user", "timestamp": "t1", "message": {"role": "user", "content": "hi"}},
            {"type": "assistant", "timestamp": "t2", "message": {"role": "assistant", "content": "yo"}},
        ],
    )
    sync = Sync(autostart=False)
    sync.conversations.collect_all()
    sync.get_conversation("claude", "s1")
    sync.get_turn("claude", "s1", 1)
    events = sync.events(category="conversation")
    actions = [(e.action, e.status) for e in events]
    assert ("get_conversation", "ok") in actions
    assert ("get_turn", "ok") in actions
    # get_turn must not also emit a get_conversation event.
    assert actions.count(("get_conversation", "ok")) == 1


def test_event_stream_records_conversation_read_failure(env):
    sync = Sync(autostart=False)
    with pytest.raises(ConversationError):
        sync.get_conversation("claude", "missing")
    errors = sync.events(status="error")
    assert any(
        e.category == "conversation" and e.action == "get_conversation" and e.error
        for e in errors
    )


def test_event_stream_persists_across_instances(env):
    _write_skill(env / ".claude" / "skills" / "g", "g")
    Sync(autostart=False).skills.push("g", get_agent("claude"))
    # A fresh instance over the same home reads the previously recorded events.
    reborn = Sync(autostart=False)
    assert any(e.category == "skill" and e.action == "push" for e in reborn.events())
