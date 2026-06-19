"""Local skill library: the master library, plus push/pull and inventory.

A *skill* is a directory containing a ``SKILL.md`` file (with YAML frontmatter
that has at least a ``name``) plus any supporting files. Each agent installs
skills under its own directory (see :mod:`parcle.sync.agents`).

Parcle keeps a single local **master library** — a directory of skill folders —
on your machine. The library is only ever changed by **manual** actions:

* **push (manual):** copy one skill from an agent into the library. It **fails
  if the library already has a skill with that name**; pass ``force=True`` to
  overwrite deliberately.
* **pull (manual):** copy a library skill into one or more agents. Symmetrically,
  it **fails if the agent already has a skill with that name**; pass
  ``force=True`` to overwrite the agent's copy deliberately.
* **remove / uninstall (manual):** delete a skill from the library, or from one
  agent.

Nothing is ever written automatically — distributing or capturing a skill is
always your explicit call. :meth:`SkillLibrary.inventory` reports which skills
each agent currently holds (the live "who has what" view, e.g. while watching),
and :meth:`SkillLibrary.drift` reports which agent-held skills are missing from
the library or have diverged from their library copy.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .agents import Agent, detect_agents, get_agent, parcle_home
from .events import CAT_SKILL, STATUS_ERROR, EventLog

__all__ = [
    "Skill",
    "SkillError",
    "SkillLibrary",
    "parse_frontmatter",
]

# Names skipped when copying a skill folder, so transient junk never travels.
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".DS_Store")


class SkillError(Exception):
    """A skill operation could not be completed (missing skill, no SKILL.md, …)."""


def parse_frontmatter(text: str) -> Dict[str, str]:
    """Parse YAML-ish ``SKILL.md`` frontmatter into ``{key: value}``.

    Handles plain scalars, quoted scalars, and block scalars (``|`` literal /
    ``>`` folded). Lists and nested maps are skipped — only scalar top-level keys
    such as ``name`` and ``description`` are needed.
    """
    out: Dict[str, str] = {}
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---", text or "", re.S)
    if not m:
        return out
    lines = m.group(1).split("\n")
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip("\r")
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1] in (" ", "\t"):
            continue  # indented line (list item / nested) — not a top-level key
        ci = line.find(":")
        if ci <= 0:
            continue
        key = line[:ci].strip()
        val = line[ci + 1 :].strip()
        if val[:1] in ("|", ">"):
            folded = val[0] == ">"
            block: List[str] = []
            base: Optional[int] = None
            while i < n:
                bl = lines[i].rstrip("\r")
                if bl.strip() == "":
                    block.append("")
                    i += 1
                    continue
                indent = len(bl) - len(bl.lstrip(" "))
                if indent == 0:
                    break  # dedented to top level → block ended
                if base is None:
                    base = indent
                block.append(bl[base:] if len(bl) >= base else bl.strip())
                i += 1
            while block and block[-1] == "":
                block.pop()
            out[key] = (
                " ".join(s.strip() for s in block if s.strip())
                if folded
                else "\n".join(block)
            )
        else:
            out[key] = val.strip("\"'")
    return out


@dataclass(frozen=True)
class Skill:
    """A skill found in some directory (a library entry or an agent install)."""

    key: str  # the skill's directory name — the handle used to push/pull
    path: Path  # absolute path to the skill directory
    name: Optional[str] = None  # SKILL.md frontmatter name
    description: Optional[str] = None  # SKILL.md frontmatter description

    @classmethod
    def read(cls, directory: Path) -> "Skill":
        """Read a skill from ``directory``, parsing its ``SKILL.md`` frontmatter.

        Raises :class:`SkillError` if there is no readable ``SKILL.md``.
        """
        skill_md = _find_skill_md(directory)
        if skill_md is None:
            raise SkillError(f"No SKILL.md found in {directory}")
        try:
            text = skill_md.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SkillError(f"Could not read {skill_md}: {exc}") from exc
        fm = parse_frontmatter(text)
        return cls(
            key=directory.name,
            path=directory,
            name=fm.get("name"),
            description=fm.get("description"),
        )


def _find_skill_md(directory: Path) -> Optional[Path]:
    if not directory.is_dir():
        return None
    for child in directory.iterdir():
        if child.is_file() and child.name.upper() == "SKILL.MD":
            return child
    return None


def _iter_skill_dirs(root: Path) -> List[Path]:
    """Return the immediate sub-directories of ``root`` that contain a SKILL.md."""
    if not root.is_dir():
        return []
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and _find_skill_md(d)]
    return dirs


class SkillLibrary:
    """The local hub of skills, stored as a directory of skill folders.

    Defaults to ``<PARCLE_HOME>/skills``. ``push`` copies an agent's skill into
    the library (failing on a name conflict unless ``force=True``); ``pull``
    copies a library skill into an agent. The library is never modified
    automatically.
    """

    def __init__(
        self, root: Optional[Path] = None, *, event_log: Optional[EventLog] = None
    ) -> None:
        self.root = root or (parcle_home() / "skills")
        self.event_log = event_log

    def _record(
        self,
        action: str,
        *,
        name: str,
        agent_type: Optional[str],
        error: Optional[str] = None,
    ) -> None:
        if self.event_log is None:
            return
        self.event_log.record(
            CAT_SKILL,
            action,
            status=STATUS_ERROR if error else "ok",
            name=name,
            agent_type=agent_type,
            error=error,
        )

    # -- listing ---------------------------------------------------------------

    def list(self) -> List[Skill]:
        """List the skills currently in the library."""
        return [Skill.read(d) for d in _iter_skill_dirs(self.root)]

    @staticmethod
    def list_agent_skills(agent: Agent) -> List[Skill]:
        """List the skills installed for ``agent``."""
        if agent.skills_dir is None:
            return []
        return [Skill.read(d) for d in _iter_skill_dirs(agent.skills_dir)]

    def inventory(self) -> Dict[str, List[Skill]]:
        """Report which skills each detected agent currently holds.

        Returns a map of ``agent_type -> [Skill, …]`` for every detected agent
        that has a skills directory. This is the live "who has what" view.
        """
        out: Dict[str, List[Skill]] = {}
        for agent in detect_agents():
            if agent.supports_skills:
                out[agent.type] = self.list_agent_skills(agent)
        return out

    # -- push / pull (manual only) ---------------------------------------------

    def push(self, name: str, agent: Agent, *, force: bool = False) -> Skill:
        """Copy the skill ``name`` from ``agent`` into the library.

        Fails with :class:`SkillError` if the library already has a skill with
        that name, unless ``force=True`` is passed to overwrite it. Also raises
        :class:`SkillError` if the agent has no skill directory or the named
        skill is missing / has no ``SKILL.md``.
        """
        try:
            if agent.skills_dir is None:
                raise SkillError(f"Agent {agent.type!r} has no skills directory.")
            source = agent.skills_dir / name
            if not source.is_dir():
                raise SkillError(
                    f"Agent {agent.type!r} has no skill {name!r} at {source}."
                )
            Skill.read(source)  # validates SKILL.md
            destination = self.root / name
            if destination.exists() and not force:
                raise SkillError(
                    f"Library already has a skill named {name!r}. "
                    f"Pass force=True to overwrite it."
                )
            _copy_tree(source, destination)
            skill = Skill.read(destination)
        except Exception as exc:  # noqa: BLE001 - log the failure, then re-raise
            self._record("push", name=name, agent_type=agent.type, error=str(exc))
            raise
        self._record("push", name=name, agent_type=agent.type)
        return skill

    def pull(self, name: str, agent: Agent, *, force: bool = False) -> Skill:
        """Install the library skill ``name`` into ``agent``.

        Fails with :class:`SkillError` if the agent already has a skill with that
        name, unless ``force=True`` is passed to overwrite the agent's copy. This
        mirrors :meth:`push`, so an edit you made on the agent side is never
        silently clobbered. Also raises :class:`SkillError` if the library has no
        such skill or the agent has no skills directory.
        """
        try:
            if agent.skills_dir is None:
                raise SkillError(f"Agent {agent.type!r} has no skills directory.")
            source = self.root / name
            if not source.is_dir():
                raise SkillError(f"Library has no skill {name!r} at {source}.")
            skill = Skill.read(source)  # validates SKILL.md
            destination = agent.skills_dir / name
            if destination.exists() and not force:
                raise SkillError(
                    f"Agent {agent.type!r} already has a skill named {name!r}. "
                    f"Pass force=True to overwrite it."
                )
            _copy_tree(source, destination)
        except Exception as exc:  # noqa: BLE001 - log the failure, then re-raise
            self._record("pull", name=name, agent_type=agent.type, error=str(exc))
            raise
        self._record("pull", name=name, agent_type=agent.type)
        return skill

    def pull_all(self, name: str, *, force: bool = False) -> Dict[str, Skill]:
        """Install the library skill ``name`` into every detected agent that
        supports skills. Returns a map of ``agent_type -> installed Skill``.

        Passes ``force`` through to each :meth:`pull`; with the default
        ``force=False`` this raises :class:`SkillError` on the first agent that
        already holds a skill of that name.
        """
        installed: Dict[str, Skill] = {}
        for agent in detect_agents():
            if not agent.supports_skills:
                continue
            installed[agent.type] = self.pull(name, agent, force=force)
        return installed

    # -- remove / uninstall (manual only) --------------------------------------

    def remove(self, name: str) -> None:
        """Delete the skill ``name`` from the library.

        Raises :class:`SkillError` if the library has no such skill.
        """
        target = self.root / name
        if not target.is_dir():
            raise SkillError(f"Library has no skill {name!r} at {target}.")
        shutil.rmtree(target)

    def uninstall(self, name: str, agent: Agent) -> None:
        """Delete the skill ``name`` from ``agent``'s skills directory.

        Raises :class:`SkillError` if the agent has no skills directory or no
        such skill installed.
        """
        if agent.skills_dir is None:
            raise SkillError(f"Agent {agent.type!r} has no skills directory.")
        target = agent.skills_dir / name
        if not target.is_dir():
            raise SkillError(
                f"Agent {agent.type!r} has no skill {name!r} at {target}."
            )
        shutil.rmtree(target)

    # -- drift (library vs agent copies) ---------------------------------------

    def drift(self) -> Dict[str, List[str]]:
        """Classify each agent-held skill against the library copy.

        Returns ``{"missing_from_library": [...], "diverged": [...]}`` where each
        entry is ``"<agent_type>/<key>"``:

        * **missing_from_library** — the agent holds the skill but the library
          does not (a ``push`` candidate).
        * **diverged** — both hold a skill of that name but their contents
          differ (out of sync — ``push`` or ``pull`` to reconcile).

        Skills that the library and agent hold identically are not listed.
        """
        library = {s.key: s for s in self.list()}
        missing: List[str] = []
        diverged: List[str] = []
        for agent_type, skills in self.inventory().items():
            for skill in skills:
                lib = library.get(skill.key)
                if lib is None:
                    missing.append(f"{agent_type}/{skill.key}")
                elif not _skill_trees_equal(lib.path, skill.path):
                    diverged.append(f"{agent_type}/{skill.key}")
        return {
            "missing_from_library": sorted(missing),
            "diverged": sorted(diverged),
        }


def _copy_tree(source: Path, destination: Path) -> None:
    """Replace ``destination`` with a copy of ``source`` (minus ignored junk)."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, ignore=_IGNORE)


# Files/dirs skipped when fingerprinting a skill, mirroring the copy step so a
# pull/push round-trip compares equal despite transient junk.
_IGNORE_NAMES = {"__pycache__", ".git", ".DS_Store"}


def _tree_fingerprint(root: Path) -> Dict[str, str]:
    """Map each non-ignored file under ``root`` to a digest of its contents.

    Keys are POSIX-style relative paths so the same skill on two machines (or two
    agents) fingerprints identically. ``*.pyc`` files and the dirs in
    ``_IGNORE_NAMES`` are skipped, matching what :func:`_copy_tree` copies.
    """
    out: Dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if path.suffix == ".pyc" or any(part in _IGNORE_NAMES for part in rel.parts):
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = ""
        out[rel.as_posix()] = digest
    return out


def _skill_trees_equal(a: Path, b: Path) -> bool:
    """True if skill directories ``a`` and ``b`` have identical content."""
    return _tree_fingerprint(a) == _tree_fingerprint(b)
