"""The local agents Parcle knows how to sync skills and conversations for.

Each :class:`Agent` describes, for one coding agent installed on this machine,
where it keeps its **skills** (a directory of ``SKILL.md`` bundles) and where it
keeps its **conversation transcripts** (line-delimited JSON files). Everything is
local: paths are resolved under the user's home directory, nothing is uploaded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "Agent",
    "KNOWN_AGENTS",
    "known_agents",
    "detect_agents",
    "get_agent",
    "parcle_home",
]


@dataclass(frozen=True)
class Agent:
    """A coding agent installed locally that Parcle can sync.

    ``skills_dir`` is where the agent reads installed skills from; ``None`` means
    the agent has no skill directory Parcle manages. ``transcripts_root`` /
    ``transcript_glob`` / ``transcript_format`` describe where the agent stores
    conversation transcripts and how to parse them; ``None`` means Parcle does
    not collect conversations for this agent.
    """

    type: str
    display_name: str
    skills_dir: Optional[Path] = None
    transcripts_root: Optional[Path] = None
    transcript_glob: Optional[str] = None
    transcript_format: Optional[str] = None

    @property
    def supports_skills(self) -> bool:
        return self.skills_dir is not None

    @property
    def supports_conversations(self) -> bool:
        return self.transcripts_root is not None and self.transcript_glob is not None

    def skills_present(self) -> bool:
        return self.skills_dir is not None and self.skills_dir.is_dir()

    def conversations_present(self) -> bool:
        return self.transcripts_root is not None and self.transcripts_root.is_dir()

    @property
    def present(self) -> bool:
        """True if this agent appears to be installed (any of its dirs exist)."""
        return self.skills_present() or self.conversations_present()


def _home() -> Path:
    return Path.home()


def _codex_base() -> Path:
    """Codex's home directory: ``CODEX_HOME`` if set, else ``~/.codex``."""
    codex_home = os.environ.get("CODEX_HOME")
    return Path(codex_home).expanduser() if codex_home else _home() / ".codex"


def known_agents() -> List[Agent]:
    """Return the built-in agent registry, with paths resolved for this user.

    Recomputed on each call so that ``HOME`` / ``CODEX_HOME`` overrides (handy in
    tests) take effect.
    """
    home = _home()
    return [
        Agent(
            type="claude",
            display_name="Claude Code",
            skills_dir=home / ".claude" / "skills",
            transcripts_root=home / ".claude" / "projects",
            transcript_glob="*/*.jsonl",
            transcript_format="claude",
        ),
        Agent(
            type="codex",
            display_name="Codex",
            skills_dir=_codex_base() / "skills",
            transcripts_root=_codex_base() / "sessions",
            transcript_glob="**/rollout-*.jsonl",
            transcript_format="codex",
        ),
        Agent(
            type="cursor",
            display_name="Cursor",
            skills_dir=home / ".cursor" / "skills",
            transcripts_root=None,
            transcript_glob=None,
            transcript_format=None,
        ),
        Agent(
            type="openclaw",
            display_name="OpenClaw",
            skills_dir=home / ".openclaw" / "skills",
            transcripts_root=None,
            transcript_glob=None,
            transcript_format=None,
        ),
    ]


# Convenience snapshot for callers that just want the list; prefer
# :func:`known_agents` when home-directory overrides matter.
KNOWN_AGENTS: List[Agent] = known_agents()


def detect_agents() -> List[Agent]:
    """Return only the agents that appear to be installed on this machine."""
    return [agent for agent in known_agents() if agent.present]


def get_agent(agent_type: str) -> Agent:
    """Look up a known agent by its ``type`` (e.g. ``"claude"``).

    Raises :class:`KeyError` if no such agent is registered.
    """
    index: Dict[str, Agent] = {a.type: a for a in known_agents()}
    try:
        return index[agent_type]
    except KeyError:
        raise KeyError(
            f"Unknown agent {agent_type!r}. Known agents: "
            + ", ".join(a.type for a in known_agents())
        )


def parcle_home() -> Path:
    """The local directory holding the skill library and collected conversations.

    Defaults to ``~/.parcle``; override with the ``PARCLE_HOME`` environment
    variable.
    """
    override = os.environ.get("PARCLE_HOME")
    return Path(override).expanduser() if override else _home() / ".parcle"
