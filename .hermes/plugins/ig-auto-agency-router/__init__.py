\"\"\"Hermes plugin: lazy router for ig-autobot's social-auto-agency agents.\"\"\"
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
