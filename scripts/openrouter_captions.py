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

# Ordered preference list. Measured behaviour on the free tier:
#   - liquid/lfm-2.5-2.6b  : fastest (~3s) but aggressively rate-limited (429)
#   - nemotron-3-super-120b: reliable, ~5x fewer reasoning tokens, slower
# Neither `reasoning.exclude` nor `effort: none` disables hidden reasoning on
# this provider, so we just fail over between models instead of tuning it.
MODEL_CHAIN = [
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

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

    Tries each model in MODEL_CHAIN in order, so a rate-limited free-tier
    model (Liquid LFM returns 429 often) fails over to the next. Returns the
    cleaned caption, or None when every model is unavailable — callers should
    fall back to the deterministic editor in that case.
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

    chain = [model] if model else list(MODEL_CHAIN)
    last_err: Optional[str] = None

    for mdl in chain:
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
                        "model": mdl,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        # Generous budget: this model family reasons before
                        # answering, and a too-small cap makes it spend
                        # everything on hidden reasoning and return
                        # content:null (finish_reason "length").
                        "max_tokens": 1024,
                        "temperature": 0.7,
                    },
                    timeout=timeout,
                )
                if resp.status_code == 402:
                    # Out of credits on the free tier — don't retry, fall back.
                    last_err = "openrouter 402 (no credits)"
                    break
                if resp.status_code == 429:
                    last_err = f"{mdl} rate-limited (429)"
                    break  # try the next model rather than waiting
                if resp.status_code in (500, 502, 503):
                    last_err = f"{mdl} {resp.status_code}"
                    if attempt < retries:
                        time.sleep(5 * (attempt + 1))
                        continue
                    break
                resp.raise_for_status()
                data = resp.json()
                text = (data.get("choices") or [{}])[0].get("message", {}).get("content")
                if not text:
                    last_err = f"{mdl} empty response"
                    break
                text = _strip_meta(text).strip()
                if not text:
                    last_err = f"{mdl} empty after cleaning"
                    break
                return _clamp(text, limit)
            except requests.RequestException as e:
                last_err = f"{mdl} {type(e).__name__}: {e}"
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


# ── Carousel narrative (the "playground" route) ───────────────────────────
#
# The carousel narrative used to come from AI Horde (kudos-constrained, and
# when exhausted the deterministic fallback produced the same generic
# boilerplate for every topic). This route generates the 5 slides + post
# caption with a hosted model first, using the carousel anatomy that
# outperforms on LinkedIn/IG:
#   1. hook   — a specific promise/claim, not a topic label (≤10 words)
#   2. context — the stakes, one concrete sentence
#   3. reframe — one counterintuitive idea
#   4. action — one specific thing to do
#   5. CTA     — a single clear invitation
# The post caption mirrors slide 1's hook.

CAROUSEL_SYSTEM = (
    "You are a carousel editor for a digital-wellness brand about intentional "
    "technology use. Voice: direct, warm, anti-cliche, no corporate speak, "
    "never use the word 'delve'.\n\n"
    "Write a 5-slide carousel. Each slide does different narrative work:\n"
    "1. HOOK — a specific promise or claim, NOT a topic label. Max 10 words. "
    "It must make someone want to swipe.\n"
    "2. CONTEXT — the stakes or the problem. One concrete sentence.\n"
    "3. REFRAME — one counterintuitive idea that changes how they see it.\n"
    "4. ACTION — one specific thing they can do. No fluff.\n"
    "5. CTA — a single short invitation to comment, save, or share.\n\n"
    "Rules: one idea per slide, no recycled phrases, no hashtags, no "
    "marketing language, slide text under 12 words each. The post caption "
    "mirrors slide 1's hook, adds 1-2 sentences of substance, and ends with "
    "a question.\n\n"
    "Return exactly 6 lines, nothing else:\n"
    "slide1\nslide2\nslide3\nslide4\nslide5\npost_caption"
)


def _parse_carousel_response(text: str) -> Optional[dict]:
    """Parse a 6-line carousel response into slides + post_caption."""
    import re as _re
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Strip "Slide N:" / "1)" style prefixes the model may add
    cleaned = []
    for line in lines:
        m = _re.match(r"^\s*(?:slide\s*\d+|post_caption|\d+)\s*[:.\)\-]?\s*(.*)$", line, _re.IGNORECASE)
        cleaned.append((m.group(1) if m else line).strip())
    cleaned = [c for c in cleaned if c]
    if len(cleaned) < 6:
        return None
    return {"slides": cleaned[:5], "post_caption": cleaned[5]}


def carousel_narrative(
    topic: str,
    voice: str = "Max Wigman: grounded, slightly literary, reflective, occasionally wry.",
    model: Optional[str] = None,
    timeout: float = 60.0,
    retries: int = 2,
) -> Optional[dict]:
    """Generate a 5-slide carousel narrative + post caption via OpenRouter.

    Tries each model in MODEL_CHAIN in order (the 120B nemotron writes
    noticeably better copy than the 2.6B; it is the primary). Returns the
    parsed dict {"slides": [...5], "post_caption": str}, or None when every
    model is unavailable — callers fall back to AI Horde / deterministic.
    """
    if not topic or not topic.strip():
        return None
    if not configured():
        return None

    system = CAROUSEL_SYSTEM + f"\nBrand voice: {voice}"
    user = f"Topic: {topic.strip().rstrip('.')}"

    chain = [model] if model else list(MODEL_CHAIN)
    last_err: Optional[str] = None

    for mdl in chain:
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
                        "model": mdl,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        # Reasoning models spend small budgets on hidden
                        # reasoning and return nothing; 512 leaves room.
                        "max_tokens": 512,
                        "temperature": 0.8,
                    },
                    timeout=timeout,
                )
                if resp.status_code == 402:
                    last_err = "openrouter 402 (no credits)"
                    break
                if resp.status_code == 429:
                    last_err = f"{mdl} rate-limited (429)"
                    break
                if resp.status_code in (500, 502, 503):
                    last_err = f"{mdl} {resp.status_code}"
                    if attempt < retries:
                        time.sleep(5 * (attempt + 1))
                        continue
                    break
                resp.raise_for_status()
                text = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content")
                if not text:
                    last_err = f"{mdl} empty response"
                    break
                parsed = _parse_carousel_response(_strip_meta(text))
                if not parsed:
                    last_err = f"{mdl} unparseable response"
                    break
                return parsed
            except requests.RequestException as e:
                last_err = f"{mdl} {type(e).__name__}: {e}"
                if attempt < retries:
                    time.sleep(5 * (attempt + 1))
                    continue
                break

    if last_err:
        print(f"  ⚠ Carousel narrative unavailable ({last_err}) — falling back")
    return None
