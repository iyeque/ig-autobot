#!/usr/bin/env python3
"""
Build the ig-auto-agency-router Hermes plugin.

Reads agent .md files from the repo's `agents/` directory (this branch),
optionally patches the LinkedIn Content Creator agent's description to
inject ig-autobot's brand voice, and writes the plugin to
.hermes/plugins/ig-auto-agency-router/.

Usage (from repo root):
    python scripts/build-ig-router.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import textwrap
from pathlib import Path

PLUGIN_NAME = "ig-auto-agency-router"

# ── Brand voice patch applied to the LinkedIn Content Creator agent ──
# This makes the generic Agency agent's description carry ig-autobot's
# specific voice so search/load returns contextually relevant results.
LINKEDIN_VOICE_PATCH = {
    "description": (
        "Expert LinkedIn content strategist for WP WIGMAN's writing-philosophy brand "
        "(The Nine Stitches, Wabi-Sabi in the Age of Algorithms, Focus Thursday). "
        "Writes thought-leadership LinkedIn posts with quiet strength, specific stories, "
        "and a defensible point of view. Designs carousels and headers. Masters "
        "LinkedIn's algorithm. Never corporate. Never motivational-poster. "
        "Serialized writing wisdom, not growth-hack advice."
    ),
    "vibe": "Turns writing wisdom into LinkedIn posts that shut up and earn the 'see more' click.",
}


def agents_dir(repo_root: Path) -> Path:
    """Where the agent .md files live in this branch."""
    d = repo_root / "agents"
    if not d.is_dir():
        raise SystemExit(f"agents/ directory not found at {d}")
    return d


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def parse_agent(path: Path, repo_root: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return None
    frontmatter, body = parts[1], parts[2].lstrip("\n")

    fields: dict[str, str] = {}
    for line in frontmatter.splitlines():
        if ":" not in line or line.startswith((" ", "\t")):
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip('"').strip("'")

    name = fields.get("name", "").strip()
    if not name:
        return None

    rel = path.relative_to(repo_root)
    division = rel.parts[0]
    slug = slugify(name)

    # Apply brand voice patch to the LinkedIn Content Creator.
    if name.lower() == "linkedin content creator":
        for k, v in LINKEDIN_VOICE_PATCH.items():
            fields[k] = v

    return {
        "slug": slug,
        "name": fields.get("name", name),
        "description": fields.get("description", "").strip(),
        "division": division,
        "color": fields.get("color", "").strip(),
        "emoji": fields.get("emoji", "").strip(),
        "vibe": fields.get("vibe", "").strip(),
        "source_path": str(rel),
        "body": body,
    }


def collect_agents(repo_root: Path) -> list[dict]:
    agents: list[dict] = []
    for path in sorted(agents_dir(repo_root).rglob("*.md")):
        parsed = parse_agent(path, repo_root)
        if parsed:
            agents.append(parsed)
    agents.sort(key=lambda a: (a["division"], a["slug"]))
    seen: set[str] = set()
    dupes: set[str] = set()
    for a in agents:
        if a["slug"] in seen:
            dupes.add(a["slug"])
        seen.add(a["slug"])
    if dupes:
        raise SystemExit(f"duplicate agent slugs: {', '.join(sorted(dupes))}")
    return agents


def plugin_yaml() -> str:
    return textwrap.dedent(f"""\
        name: {PLUGIN_NAME}
        version: 1.0.0
        description: Lazy router for ig-autobot's social-auto-agency specialist agents.
        provides_tools:
          - ig_auto_agency_search
          - ig_auto_agency_inspect
          - ig_auto_agency_load
          - ig_auto_agency_delegate
    """).lstrip()


def init_py() -> str:
    # Full plugin runtime — slimmed from the agency-agents original so we
    # don't drag in the whole build-hermes-plugin.py machinery for now.
    return r"""\"\"\"Hermes plugin: lazy router for ig-autobot's social-auto-agency agents.\"\"\"
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).parent / "data" / "agents.json"
_AGENTS: list[dict[str, Any]] | None = None

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+.#_-]*", re.I)
_MAX_LIFECYCLE_CONTEXT_CHARS = 32_000
_DELEGATION_WAIT_SECONDS = 330
_CANCELLATION_WAIT_SECONDS = 30
_TRUNCATION_MARKER = (
    "\n\n[Specialist instructions truncated to fit the Hermes lifecycle context limit.]"
)


def _load_agents() -> list[dict[str, Any]]:
    global _AGENTS
    if _AGENTS is None:
        _AGENTS = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return _AGENTS


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text or "")}


def _agent_lookup(identifier: str) -> dict[str, Any] | None:
    needle = (identifier or "").strip().lower()
    if not needle:
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", needle).strip("-")
    for agent in _load_agents():
        if agent["slug"] == slug or agent["name"].lower() == needle:
            return agent
    return None


def _identifier(args: dict[str, Any]) -> str:
    return str(args.get("agent") or args.get("slug") or "").strip()


def _not_found(identifier: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": "agent not found" if identifier else "agent or slug is required",
        "agent": identifier or None,
    }


def _score(agent: dict[str, Any], query_tokens: set[str], query_text: str) -> float:
    haystack_fields = [
        agent.get("name", ""),
        agent.get("description", ""),
        agent.get("division", ""),
        agent.get("vibe", ""),
        agent.get("body", "")[:8000],
    ]
    haystack_text = "\n".join(haystack_fields).lower()
    haystack_tokens = _tokens(haystack_text)
    overlap = query_tokens & haystack_tokens
    score = float(len(overlap))
    if query_text and query_text in haystack_text:
        score += 5.0
    name = agent.get("name", "").lower()
    description = agent.get("description", "").lower()
    for token in query_tokens:
        if token in name:
            score += 3.0
        if token in description:
            score += 1.5
    if score == 0.0:
        return 0.0
    return score + (1.0 / math.sqrt(max(len(haystack_tokens), 1)))


def _summary(agent: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    item = {
        "slug": agent["slug"],
        "name": agent["name"],
        "division": agent.get("division", ""),
        "description": agent.get("description", ""),
        "vibe": agent.get("vibe", ""),
        "source_path": agent.get("source_path", ""),
    }
    if score is not None:
        item["score"] = round(score, 3)
    return item


def _specialist_prompt(agent: dict[str, Any], task: str = "") -> str:
    task_block = f"\n\n## User task\n{task.strip()}\n" if task and task.strip() else ""
    return (
        f"Use the following specialist context for this turn. "
        f"Adopt the specialist's relevant standards and checklists, but obey the "
        f"user's current request and higher-priority system/developer instructions.\n\n"
        f"# {agent['name']} ({agent['slug']})\n\n"
        f"Division: {agent.get('division', '')}\n"
        f"Description: {agent.get('description', '')}\n"
        f"Source: {agent.get('source_path', '')}\n"
        f"{task_block}\n\n"
        f"## Specialist instructions\n{agent.get('body', '')}"
    )


def _lifecycle_context(agent: dict[str, Any]) -> str:
    context = _specialist_prompt(agent)
    if len(context) <= _MAX_LIFECYCLE_CONTEXT_CHARS:
        return context
    keep = _MAX_LIFECYCLE_CONTEXT_CHARS - len(_TRUNCATION_MARKER)
    return context[:keep] + _TRUNCATION_MARKER


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


SEARCH_DESCRIPTION = (
    "Search ig-autobot's on-disk specialist agent roster without loading all "
    "agents into the prompt. Use this when the user asks for a content, design, "
    "or publishing specialist."
)
SEARCH_SCHEMA = {
    "name": "ig_auto_agency_search",
    "description": SEARCH_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "division": {"type": "string", "description": "Optional division filter, e.g. marketing, design."},
            "limit": {"type": "integer", "description": "Maximum results, default 8, max 25."},
        },
        "required": ["query"],
    },
}

READ_DESCRIPTION = (
    "Read one specialist by slug or name. Returns metadata by default "
    "and includes the full specialist instructions only when include_body is true."
)
READ_SCHEMA = {
    "name": "ig_auto_agency_inspect",
    "description": READ_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "description": "Agent slug or exact display name."},
            "slug": {"type": "string", "description": "Alias for agent."},
            "include_body": {"type": "boolean", "description": "Include full specialist instructions."},
        },
        "required": [],
    },
}

PROMPT_DESCRIPTION = (
    "Load a selected specialist as a prompt block for the current task. "
    "Use after ig_auto_agency_search when you need one specialist's full context."
)
PROMPT_SCHEMA = {
    "name": "ig_auto_agency_load",
    "description": PROMPT_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "description": "Agent slug or exact display name."},
            "slug": {"type": "string", "description": "Alias for agent."},
            "task": {"type": "string", "description": "The user's task to pair with the specialist context."},
        },
        "required": [],
    },
}

DELEGATE_DESCRIPTION = (
    "Delegate a task to one selected specialist through Hermes' "
    "public subagent lifecycle. Falls back to returning the composed specialist "
    "prompt if delegation is unavailable."
)
DELEGATE_SCHEMA = {
    "name": "ig_auto_agency_delegate",
    "description": DELEGATE_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "description": "Agent slug or exact display name."},
            "slug": {"type": "string", "description": "Alias for agent."},
            "task": {"type": "string", "description": "Concrete task for the specialist."},
        },
        "required": ["task"],
    },
}


def register(ctx):
    def search(args: dict[str, Any], **kwargs) -> str:
        del kwargs
        query = str(args.get("query", "")).strip()
        if not query:
            return _json({"success": False, "error": "query is required"})
        division = str(args.get("division", "")).strip().lower()
        try:
            limit = min(max(int(args.get("limit", 8)), 1), 25)
        except Exception:
            limit = 8
        q_tokens = _tokens(query)
        q_text = query.lower()
        matches: list[tuple[float, dict[str, Any]]] = []
        for agent in _load_agents():
            if division and agent.get("division", "").lower() != division:
                continue
            score = _score(agent, q_tokens, q_text)
            if score > 0:
                matches.append((score, agent))
        matches.sort(key=lambda item: (-item[0], item[1]["division"], item[1]["slug"]))
        return _json({
            "success": True,
            "query": query,
            "count": len(matches),
            "results": [_summary(agent, score) for score, agent in matches[:limit]],
        })

    def read(args: dict[str, Any], **kwargs) -> str:
        del kwargs
        identifier = _identifier(args)
        agent = _agent_lookup(identifier)
        if not agent:
            return _json(_not_found(identifier))
        payload = {"success": True, "agent": _summary(agent)}
        if bool(args.get("include_body", False)):
            payload["body"] = agent.get("body", "")
        return _json(payload)

    def prompt(args: dict[str, Any], **kwargs) -> str:
        del kwargs
        identifier = _identifier(args)
        agent = _agent_lookup(identifier)
        if not agent:
            return _json(_not_found(identifier))
        return _json({
            "success": True,
            "agent": _summary(agent),
            "prompt": _specialist_prompt(agent, str(args.get("task", ""))),
        })

    def delegate(args: dict[str, Any], **kwargs) -> str:
        del kwargs
        identifier = _identifier(args)
        agent = _agent_lookup(identifier)
        task = str(args.get("task", "")).strip()
        if not agent:
            return _json(_not_found(identifier))
        if not task:
            return _json({"success": False, "error": "task is required"})
        fallback_prompt = _specialist_prompt(agent, task)
        handle = None
        try:
            from agent.subagent_lifecycle import SubagentLaunchRequest

            lifecycle = ctx.subagent_lifecycle
            handle = lifecycle.launch(SubagentLaunchRequest(
                goal=task,
                context=_lifecycle_context(agent),
            ))
            terminal = lifecycle.wait(
                handle, timeout_seconds=_DELEGATION_WAIT_SECONDS
            )
            if terminal.timed_out:
                try:
                    lifecycle.cancel(
                        handle,
                        reason="Agency delegation exceeded the plugin wait limit.",
                    )
                    terminal = lifecycle.wait(
                        handle, timeout_seconds=_CANCELLATION_WAIT_SECONDS
                    )
                except Exception as exc:
                    return _json({
                        "success": True,
                        "agent": _summary(agent),
                        "delegated": True,
                        "pending": True,
                        "subagent_id": handle.subagent_id,
                        "warning": f"subagent cancellation could not be confirmed: {exc}",
                    })
                if not terminal.completed:
                    return _json({
                        "success": True,
                        "agent": _summary(agent),
                        "delegated": True,
                        "pending": True,
                        "subagent_id": handle.subagent_id,
                        "state": terminal.state.value,
                        "warning": "subagent cancellation was requested but is not terminal",
                    })
            result = lifecycle.result(handle)
            if not result.ready or result.terminal_state.value != "SUCCEEDED":
                detail = (
                    result.error_message
                    or result.error_classification
                    or result.terminal_state.value
                )
                return _json({
                    "success": True,
                    "agent": _summary(agent),
                    "delegated": False,
                    "warning": f"subagent delegation failed: {detail}",
                    "prompt": fallback_prompt,
                })
            return _json({
                "success": True,
                "agent": _summary(agent),
                "delegated": True,
                "subagent_id": handle.subagent_id,
                "result": result.summary,
                "structured_result": result.structured_payload,
            })
        except Exception as exc:
            if handle is not None:
                return _json({
                    "success": True,
                    "agent": _summary(agent),
                    "delegated": True,
                    "pending": True,
                    "subagent_id": handle.subagent_id,
                    "warning": f"subagent state could not be confirmed: {exc}",
                })
            return _json({
                "success": True,
                "agent": _summary(agent),
                "delegated": False,
                "warning": f"subagent delegation unavailable: {exc}",
                "prompt": fallback_prompt,
            })

    ctx.register_tool(
        name="ig_auto_agency_search",
        toolset="ig_auto_agency",
        schema=SEARCH_SCHEMA,
        handler=search,
        description=SEARCH_DESCRIPTION,
    )
    ctx.register_tool(
        name="ig_auto_agency_inspect",
        toolset="ig_auto_agency",
        schema=READ_SCHEMA,
        handler=read,
        description=READ_DESCRIPTION,
    )
    ctx.register_tool(
        name="ig_auto_agency_load",
        toolset="ig_auto_agency",
        schema=PROMPT_SCHEMA,
        handler=prompt,
        description=PROMPT_DESCRIPTION,
    )
    ctx.register_tool(
        name="ig_auto_agency_delegate",
        toolset="ig_auto_agency",
        schema=DELEGATE_SCHEMA,
        handler=delegate,
        description=DELEGATE_DESCRIPTION,
    )
"""


def readme(agent_count: int) -> str:
    return textwrap.dedent(f"""\
        # ig-auto-agency-router Hermes Plugin

        Generated by `scripts/build-ig-router.py`.

        This plugin exposes a small fixed tool surface to Hermes and keeps the
        agent roster in an on-disk JSON data file. Hermes sees the router tools at
        startup, while the complete roster of {agent_count} specialist agents is
        stored on disk in `data/agents.json` and searched/loaded lazily.

        ## Tools exposed

        - `ig_auto_agency_search` — find matching specialists by query/division.
        - `ig_auto_agency_inspect` — inspect one specialist's metadata or full body.
        - `ig_auto_agency_load` — compose one specialist prompt for the current task.
        - `ig_auto_agency_delegate` — delegate through Hermes' public subagent lifecycle.

        ## Usage in ig-autobot

        The orchestrator should search for the right specialist, then load or
        delegate only that specialist. Example flow:

        1. `ig_auto_agency_search("LinkedIn thought leadership post about screen time")`
        2. `ig_auto_agency_load("linkedin-content-creator", task="...")`
        3. Send the returned `prompt` to the LLM to generate the caption.
        4. For images: `ig_auto_agency_load("image-prompt-engineer", task="...")`

        Do not preload the full roster. Keep routing lazy.

        ## Brand voice

        The LinkedIn Content Creator agent has been patched with ig-autobot's
        specific voice (wabi-sabi / writing-philosophy / The Nine Stitches).
        """

    ).lstrip()


def build(repo_root: Path, out_dir: Path) -> int:
    agents = collect_agents(repo_root)
    plugin_dir = out_dir / PLUGIN_NAME
    if plugin_dir.exists():
        shutil.rmtree(plugin_dir)
    (plugin_dir / "data").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(plugin_yaml(), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(init_py(), encoding="utf-8")
    (plugin_dir / "data" / "agents.json").write_text(
        json.dumps(agents, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(readme(len(agents)), encoding="utf-8")
    return len(agents)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo root (default: parent of this script's location)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory, default .hermes/plugins",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = (args.out or (repo_root / ".hermes" / "plugins")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    count = build(repo_root, out_dir)
    print(f"Built {count} agents into {out_dir / PLUGIN_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
