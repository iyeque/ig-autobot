"""OpenRouter caption tailoring.

Replaces the per-platform AI Horde text calls (6-7 per bundle) with a single
cheap, fast LLM. Liquid LFM 2.6B answers in ~3s on the free tier, so a whole
bundle's captions cost a fraction of one AI Horde text request.

The deterministic editor in bot.py stays as the fallback: if OpenRouter is
unreachable, unconfigured, or returns unusable output, callers fall back to
local cleaning rather than failing the bundle.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import requests

DEFAULT_MODEL = "liquid/lfm-2.5-2.6b:free"
FALLBACK_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Per-platform character limits, mirrored from bot._PLATFORM_CHAR_LIMITS.
PLATFORM_LIMITS = {
    "instagram": 2200,
    "linkedin": 3000,
    "threads": 420,
    "bluesky": 300,
    "youtube": 5000,
    "pinterest": 500,
    "facebook": 63206,
}

# Platform-specific shaping rules. Kept terse so the model has room to write.
# Hashtags are deliberately NOT requested: the caller appends them via
# choose_hashtags(), and model-generated hashtags come out garbled
# ("#WabisicabEAesthetics").
PLATFORM_RULES = {
    "instagram": "Conversational, 2-4 short paragraphs, end with an engagement question.",
    "linkedin": "Professional but opinionated, first line is a hook, no emoji.",
    "threads": "Punchy, one idea per line, end with a single question.",
    "bluesky": "Short and direct, under 300 characters total.",
    "youtube": "Descriptive and searchable, 2-3 sentences.",
    "pinterest": "Keyword-rich title plus 2-3 short lines, under 500 characters.",
    "facebook": "Warm and community-focused, end with a question.",
}

# Meta-commentary the model sometimes emits instead of copy. Stripped on the
# way out so it can never reach a published post.
_META_OPENERS = [
    r"^here'?s a compelling",
    r"^here is a (linkedin|bluesky|threads?|instagram|youtube|post)",
    r"^this (caption|post|thread)\s*(does|is|has|will|captures|includes)",
    r"^this captures\s*:?\s*$",
    r"^captures\s*:?\s*$",
    r"^thread\s*:\s*",
    r"^[✓✔☑]\s",
    r"^\d+\.\s+(stays|references|includes|ends|maintains|uses|avoids)",
]


def _api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def configured() -> bool:
    """True when an OpenRouter key is available."""
    return bool(_api_key())


def _strip_meta(text: str) -> str:
    """Remove model process-notes that leaked into the output."""
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if any(re.search(p, s, re.IGNORECASE) for p in _META_OPENERS):
            continue
        kept.append(line)
    out = "\n".join(kept).strip()

    # Once the model starts explaining itself, drop everything after.
    m = re.search(r"(?im)^(this (caption|post|thread)|this captures|✓|captures\s*:)", out)
    if m:
        out = out[: m.start()].strip()
    return out


def _clamp(text: str, limit: int) -> str:
    """Trim to the limit at a sentence/line boundary."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 3]
    for punct in (". ", "? ", "! ", "\n"):
        idx = cut.rfind(punct)
        if idx > limit * 0.5:
            cut = cut[: idx + 1]
            break
    else:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip() + "..."


def tailor(
    master_reflection: str,
    platform: str,
    max_chars: Optional[int] = None,
    model: Optional[str] = None,
    timeout: float = 45.0,
    retries: int = 2,
) -> Optional[str]:
    """Tailor a master reflection for one platform.

    Returns the cleaned caption, or None when OpenRouter is unavailable or
    produced nothing usable — callers should fall back to the deterministic
    editor in that case.
    """
    if not master_reflection or not master_reflection.strip():
        return None
    if not configured():
        return None

    p = platform.lower()
    limit = max_chars or PLATFORM_LIMITS.get(p, 1800)
    # Leave headroom for the CTA/hashtag block appended downstream.
    budget = max(80, limit - 120)
    rules = PLATFORM_RULES.get(p, "Short, on-brand, end with a question.")

    system = (
        "You write social media captions for The Nine Stitches, a brand about "
        "digital wellbeing and intentional creation. Voice: direct, warm, "
        "anti-cliche, no corporate speak. Never use the word 'delve'.\n"
        f"Platform: {p}. Rules: {rules}\n"
        f"Return ONLY the caption text, under {budget} characters. "
        "No preamble, no labels, no markdown, no commentary about the caption. "
        "Write in natural sentences. Do not append labels, notes, or "
        "placeholders, and do not repeat a question after every line."
    )

    user = f"Master reflection:\n{master_reflection.strip()}\n\nWrite the {p} caption."

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {_api_key()}",
                    "HTTP-Referer": "https://github.com/iyeque/ig-autobot",
                    "X-Title": "ig-autobot",
                },
                json={
                    "model": model or DEFAULT_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    # This is a reasoning model: it spends tokens on a hidden
                    # `reasoning` field before answering. With a small budget it
                    # burns everything there and returns content:null with
                    # finish_reason "length". Ask the provider to exclude the
                    # reasoning block and give the answer room to exist.
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "reasoning": {"exclude": True},
                },
                timeout=timeout,
            )
            if resp.status_code == 402:
                # Out of credits on the free tier — don't retry, fall back.
                last_err = "openrouter 402 (no credits)"
                break
            if resp.status_code in (429, 500, 502, 503):
                last_err = f"openrouter {resp.status_code}"
                if attempt < retries:
                    time.sleep(5 * (attempt + 1))
                    continue
                break
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if not text:
                last_err = "empty response"
                break
            text = _strip_meta(text).strip()
            if not text:
                last_err = "empty response"
                break
            return _clamp(text, limit)
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(5 * (attempt + 1))
                continue
            break

    if last_err:
        print(f"  ⚠ OpenRouter unavailable ({last_err}) — using deterministic editor")
    return None


def tailor_all(
    master_reflection: str,
    platforms: list[str],
    limits: Optional[dict[str, int]] = None,
) -> dict[str, Optional[str]]:
    """Tailor for several platforms. Each call is independent."""
    limits = limits or {}
    out: dict[str, Optional[str]] = {}
    for p in platforms:
        out[p] = tailor(master_reflection, p, limits.get(p))
    return out
