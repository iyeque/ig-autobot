#!/usr/bin/env python3
"""
Brand Guardian — self-healing pre-publish context integrity gate.

Runs as a CI step between prepare_assets and publish. For each pending
caption, it:

1. Detects drift between caption.txt and state.json (source of truth)
2. Auto-heals fixable issues (trim length, add CTA, fix hashtags, fix drift)
3. Attempts recovery for unfixable issues via AI Horde regeneration or
   deterministic fallback before giving up
4. Writes back the healed caption.txt so the publish step gets clean assets

Exit 0 = all clear or healed. Exit 1 = unfixable issues found (platforms
may be skipped, but the pipeline continues).
"""

import os
import sys
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE_PATH = REPO / "state.json"
CAPTION_PATH = REPO / "caption.txt"
POSTS_PATH = REPO / "posts.json"

# Platform caption limits (characters)
PLATFORM_LIMITS = {
    "instagram": 2200,
    "linkedin": 3000,
    "threads": 420,
    "bluesky": 300,
    "youtube": 5000,
    "pinterest": 500,
}

# Platforms that benefit from a CTA / "read more" suffix
CTA_PLATFORMS = {"bluesky", "threads"}
BLUESKY_CTA = "Want to read more?... check out my LinkedIn"

# Platforms where hashtags in the first comment are preferred over inline
HASHTAG_INLINE = {"instagram", "youtube", "pinterest"}

# Text patterns that indicate un-generated or broken content
BAD_PATTERNS = [
    "[caption generation failed",
    "[Caption generation failed",
    "Traceback (most recent call last)",
    "<module>",
    "Error:",
    "Exception:",
]


# ── Content judge: the semantic gate ──────────────────────────────────────
#
# The pattern layer below is the FLOOR, not the gate. Everything observed
# leaking into published captions across Oct 7-9 2026:
#   "(Paste here)", "(Execute the final output...)", "(Generate Bluesky Post)"
#   "(The Professional Failure Expert Persona)", "(Thread 1/X)", "(Self-
#   Correction/Refinement on my part)", "My Analysis: Assembling this
#   Marketing Content Creator Agent's profile...", "## Building Guardd..."
#   "(This is where the prompt would be generated.)"
# A regex floor catches these cheaply, but the gate's real verdict comes from
# the LLM judge — patterns alone will always be one leak behind.

AGENT_LEAK_PATTERNS = [
    # Unfilled placeholders
    r"\(\s*paste\s+here\s*\)",
    r"\(\s*execute\b",
    r"\(\s*generate\s+\w+",
    r"\(\s*thread\s+\d+\s*/\s*[xX\d]+\s*\)",
    r"\(\s*this\s+is\s+where\b",
    r"\{\{[^}]{1,40}\}\}",
    r"\[(?:your|specific|insert|name|company|topic|product|milestone|title|placeholder)[^\]]{0,40}\]",
    # Persona / role headers
    r"^\**\s*\([^)]{0,50}(?:persona|expert|agent|generator|author)\b",
    # Self-correction / process notes
    r"self[-\s]?correction",
    r"note\s+to\s+(?:the\s+)?generator",
    r"my\s+analysis\s*:",
    r"assembling\s+this\s+\w+\s+agent",
    r"would\s+you\s+like\s+me\s+to",
    r"notes?\s*(?:&|and)\s*justification",
    r"character\s+count\s*:",
    r"kept\s+it\s+under\s+\d+\s+character",
    # Draft menus / template scaffolding
    r"here\s+are\s+(?:three|two|four|five|\d+)\s+caption",
    r"^\s*variant\s+\d+\s*[\(:.]",
    r"^\s*thread\s+start\s*:",
    r"^\s*part\s+\d+\s*:",
    r"a\s+well[-\s]crafted\s+\w+\s+post\s+should\s+have",
    r"^\s*for\s+best\s+results\b",
    r"feel\s+free\s+to\s+adjust",
    r"based\s+on\s+(?:tone|style|your)\s+preference",
    r"please\s+provide\s+those\s+details",
]

# Caption-only floors: a published caption must not carry markdown structure.
# A master_reflection legitimately contains markdown (the reflection writer
# uses headers and lists, and the caption builder strips them), so these are
# deliberately NOT part of the source check — bundle 321's reflection has a
# "### " header and a numbered list and is perfectly good source material.
CAPTION_LEAK_PATTERNS = AGENT_LEAK_PATTERNS + [
    r"^\s*#{1,6}\s+",
    r"^\s*[-*]?\s*\[[ xX]\]",
]

# Minimum real caption length per platform. Bundle 322 shipped pinterest at
# 12 chars ("(Paste here)"), youtube at 51, bluesky at 68 — none are captions.
MIN_CAPTION_LEN = {
    "bluesky": 40,
    "threads": 60,
    "pinterest": 80,
    "youtube": 80,
    "instagram": 80,
    "facebook": 80,
    "linkedin": 100,
}

# Minimum length for a master_reflection — the source every caption is built
# from. Below this it cannot carry a post's worth of content.
MIN_SOURCE_LEN = 100

# Judge model order. The 120B nemotron is the primary judge: it reliably
# answers PASS/FAIL (measured FAIL on bundle 322's leaked direction line,
# PASS on 321's clean copy). The 2.6B liquid is the fallback — it passes
# subtle leaks, so it only ever votes when the stronger model is down.
JUDGE_MODEL_CHAIN = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "liquid/lfm-2.5-2.6b:free",
]


def has_agent_leak(text: str, patterns: list[str] | None = None) -> list[str]:
    """Return the leak patterns a text matches. Empty list = no leak found.

    patterns defaults to the caption set (markdown floors included); pass
    AGENT_LEAK_PATTERNS explicitly for source text.
    """
    if not text:
        return []
    hits: list[str] = []
    for pat in (patterns if patterns is not None else CAPTION_LEAK_PATTERNS):
        if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
            hits.append(pat)
    return hits


def _llm_judge(
    text: str,
    platform: str,
    timeout: float = 90.0,
    system_override: str | None = None,
) -> bool | None:
    """Ask a small LLM whether text is a finished, publishable caption.

    Returns True (PASS), False (FAIL), or None when no judge is reachable —
    callers fall back to floors-only in that case. Never raises.

    Model notes (measured Oct 9 2026): the 120B nemotron reliably answers
    PASS/FAIL with max_tokens=512 — at 64 it spends the whole budget on
    hidden reasoning and returns nothing. The 2.6B liquid model is a weaker
    judge (it passed bundle 322's leaked direction line) and rate-limits
    hard, so it is the fallback, not the primary. A missing answer from one
    model means "try the next", never "accept".
    """
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        import openrouter_captions as orc
        if not orc.configured():
            return None

        system = system_override or (
            "You are a quality gate for a social media publishing pipeline. "
            "Decide whether the text is a finished, ready-to-publish "
            f"{platform} caption.\n\n"
            "REJECT if it contains ANY of:\n"
            "- agent process notes, reasoning, or self-commentary\n"
            '- style directions or briefs like "Professional, reflective, yet punchy." or "Tone: witty"\n'
            '- unfilled placeholders like "(Paste here)", "(Generate X Post)", "[Your Name]"\n'
            '- persona or role headers like "(The X Expert Persona)" or "Thread Start:"\n'
            "- template scaffolding, instructions, or draft menus\n"
            "- markdown headers, checkbox walls\n\n"
            "ACCEPT only if it reads like finished human-written social copy "
            "from the very first line.\n\n"
            "Answer with one word: PASS or FAIL."
        )
        user = f"TEXT:\n{text[:2000]}"

        import requests
        for mdl in JUDGE_MODEL_CHAIN:
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {orc._api_key()}",
                        "HTTP-Referer": "https://github.com/iyeque/ig-autobot",
                        "X-Title": "ig-autobot",
                    },
                    json={
                        "model": mdl,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        # 512, not 64: reasoning models spend small budgets on
                        # hidden reasoning and return content:null, which must
                        # not be read as a verdict.
                        "max_tokens": 512,
                        "temperature": 0.0,
                    },
                    timeout=timeout,
                )
                if resp.status_code != 200:
                    continue  # try next model
                msg = (resp.json().get("choices") or [{}])[0].get("message", {})
                content = msg.get("content") or msg.get("reasoning") or ""
                verdict, reason = _parse_verdict(content)
                if verdict is True:
                    return True
                if verdict is False:
                    print(f"    ⚖️  Judge FAIL for {platform}: {reason}")
                    return False
                # Unparseable/empty answer — next model, not a verdict
            except Exception:
                continue
        return None
    except Exception:
        return None


def _parse_verdict(content: str) -> tuple[bool | None, str]:
    """Extract a PASS/FAIL verdict from a judge answer.

    Some reasoning models emit the verdict in the last line of their
    reasoning, not in `content`. Scan the tail of the response for the
    first unambiguous PASS/FAIL token. Returns (verdict, reason).
    """
    if not content or not content.strip():
        return None, ""
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    for line in reversed(lines[-4:] if len(lines) >= 4 else lines):
        u = line.upper().strip(".:;! ")
        if u.startswith("FAIL"):
            return False, line.strip() or "judge rejected"
        if u.startswith("PASS"):
            return True, ""
    return None, ""


def judge_caption(caption: str, platform: str, use_llm: bool = True) -> tuple[bool, str]:
    """Judge one caption. Returns (ok, reason).

    Order: floors (free, certain) → LLM judge (semantic) when reachable.
    Floors fail = reject regardless. Judge unreachable = floors-only verdict.
    """
    p = platform.lower()
    text = (caption or "").strip()

    if is_empty_caption(text):
        return False, "empty caption"

    floor_hits = has_agent_leak(text)
    if floor_hits:
        return False, f"agent leak: {floor_hits[0]}"

    min_len = MIN_CAPTION_LEN.get(p, 80)
    if len(text) < min_len:
        return False, f"too short for {p}: {len(text)} < {min_len} chars"

    if use_llm:
        verdict = _llm_judge(text, p)
        if verdict is False:
            return False, "judge rejected"
        # True or None (unavailable) both fall through to accept

    return True, "ok"


def judge_source(master_reflection: str, use_llm: bool = True) -> tuple[bool, str]:
    """Judge a master_reflection — the source every caption is built from.

    A contaminated source can never produce clean captions (bundles 320 and
    322 both shipped the agent's own reasoning as the reflection), so this
    runs at generation time, before anything is built from it.
    """
    text = (master_reflection or "").strip()
    if not text:
        return False, "empty master_reflection"

    leak_hits = has_agent_leak(text, AGENT_LEAK_PATTERNS)
    if leak_hits:
        return False, f"source contaminated: {leak_hits[0]}"

    if len(text) < MIN_SOURCE_LEN:
        return False, f"source too short: {len(text)} < {MIN_SOURCE_LEN} chars"

    if use_llm:
        verdict = _llm_judge(
            text[:1500],
            "linkedin",
            system_override=(
                "You are a quality gate for a content pipeline. Below is a "
                "master reflection — the first-person source prose that a "
                "caption editor will build platform captions from. It is "
                "legitimately long-form prose and MAY contain first-person "
                "voice, rhetorical questions, headers, and lists — those are "
                "normal writing, not problems.\n\n"
                "REJECT it ONLY if it contains agent process contamination:\n"
                "- persona or role headers (\"(The X Expert Persona)\")\n"
                '- unfilled placeholders ("(Paste here)", "[Your Name]")\n'
                "- process notes, reasoning, self-commentary, draft menus\n"
                "- template scaffolding or instructions\n\n"
                "ACCEPT if it is genuine reflection prose usable as source "
                "material, even if informal.\n\n"
                "Answer with one word: PASS or FAIL."
            ),
        )
        if verdict is False:
            return False, "judge rejected source"

    return True, "ok"


# ── Normalization helpers ──────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Strip markdown and normalize whitespace + quotes for comparison.
    Must be idempotent with clean_caption so drift detection is stable."""
    text = text.replace("**", "").replace("*", "")
    text = text.replace("__", "").replace("_", "")
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("'", "'").replace("'", "'")  # normalize curly quotes
    text = " ".join(text.split())
    return text.lower()


def clean_caption(text: str) -> str:
    """Apply the same cleaning prepare_assets does: strip markdown formatting."""
    text = text.replace("**", "").replace("*", "")
    text = text.replace("__", "").replace("_", "")
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("'", "\u2019").replace("'", "\u2018")
    return text.strip()


# ── Heal helpers ────────────────────────────────────────────────────────────

def heal_drift(state_caption: str, platform: str, prepared_caption: str) -> tuple[str, list[str]]:
    """
    Detect and fix drift between state.json caption and caption.txt.

    Returns (healed_caption, list_of_fixes_applied).
    If no drift, returns (prepared_caption, []).
    If drift detected, regenerates caption.txt content from state.json
    and applies platform tailoring.
    """
    fixes = []

    # Normalize both for comparison
    state_norm = normalize(state_caption)
    prepared_norm = normalize(prepared_caption)

    if state_norm == prepared_norm:
        return prepared_caption, fixes  # No drift

    # Drift detected — re-derive from state.json (source of truth)
    fixes.append(f"  🔧  [{platform}] State drift detected — regenerating caption.txt from state.json")

    # Start from the state.json caption (the source of truth)
    healed = state_caption

    # Apply platform tailoring (same logic as prepare_assets._apply_platform_tailoring)
    healed = apply_platform_tailoring(healed, platform)

    return healed, fixes


def apply_platform_tailoring(caption: str, platform: str) -> str:
    """Apply platform-specific tailoring to a caption (same as prepare_assets)."""
    text = caption.strip()
    if not text:
        return text

    if platform == "threads":
        if len(text) > 420:
            text = text[:417].rstrip() + "..."
        return text

    if platform == "bluesky":
        suffix = BLUESKY_CTA
        if suffix not in text:
            text = text.rstrip() + "\n\n" + suffix
        if len(text) > 300:
            # Trim to fit CTA — remove from middle
            available = 300 - len(suffix) - 2  # 2 for \n\n
            if available < 50:
                text = text[:available].rstrip() + "..."
            else:
                # Keep first and last parts
                half = (available - 3) // 2  # 3 for "..."
                text = text[:half].rstrip() + "..." + text[-(available - half - 3):].lstrip()
            text = text.rstrip() + "\n\n" + suffix

        return text

    if platform == "youtube":
        if len(text) > 5000:
            text = text[:4997].rstrip() + "..."
        return text

    if platform == "linkedin":
        if len(text) > 3000:
            text = text[:2997].rstrip() + "..."
        return text

    # Instagram, pinterest: just clean
    return clean_caption(text)


def heal_length(caption: str, platform: str) -> tuple[str, list[str]]:
    """Trim caption if it exceeds platform limit. Returns (healed, fixes)."""
    fixes = []
    limit = PLATFORM_LIMITS.get(platform, 3000)
    length = len(caption)

    if length > limit:
        # Trim to limit
        healed = caption[:limit - 3].rstrip() + "..."
        fixes.append(
            f"  🔧  [{platform}] Trimmed caption from {length} → {limit} chars "
            f"(platform limit)"
        )
        return healed, fixes

    return caption, fixes


def heal_hashtags(caption: str, platform: str) -> tuple[str, list[str]]:
    """Fix double hashtags and excessive hashtag count. Returns (healed, fixes)."""
    fixes = []
    if platform not in HASHTAG_INLINE:
        return caption, fixes

    # Fix double hashtags (## → #)
    if "##" in caption:
        healed = caption.replace("##", "#")
        fixes.append(f"  🔧  [{platform}] Fixed double hashtag '##' → '#'")
        caption = healed

    # Trim excessive hashtags (>10)
    hashtags = re.findall(r"#\w+", caption)
    if len(hashtags) > 10:
        # Build a version with only first 10 hashtags
        result = []
        seen = 0
        i = 0
        while i < len(caption):
            if caption[i] == "#" and i == 0 or (i > 0 and caption[i-1] in " \t\n"):
                # Potential hashtag start
                end = i + 1
                while end < len(caption) and (caption[end].isalnum() or caption[end] == "_"):
                    end += 1
                if end > i + 1:  # It's a real hashtag
                    seen += 1
                    if seen <= 10:
                        result.append(caption[i:end])
                        i = end
                        continue
                    else:
                        # Skip this hashtag and everything until next whitespace
                        j = end
                        while j < len(caption) and caption[j] not in " \t\n":
                            j += 1
                        i = j
                        continue
            result.append(caption[i])
            i += 1
        healed = "".join(result).rstrip()
        fixes.append(
            f"  🔧  [{platform}] Trimmed {len(hashtags)} → 10 hashtags "
            f"(removed {len(hashtags) - 10})"
        )
        return healed, fixes

    return caption, fixes


def heal_bluesky_cta(caption: str) -> tuple[str, list[str]]:
    """Ensure Bluesky caption has the CTA suffix. Returns (healed, fixes)."""
    fixes = []
    if "bluesky" not in getattr(sys, "_brand_guardian_platform", ""):
        return caption, fixes

    suffix = BLUESKY_CTA
    if suffix not in caption:
        # Check if there's a similar suffix already
        if "check out my" in caption.lower() or "linkedin" in caption.lower():
            fixes.append(f"  ℹ️  [bluesky] CTA variant already present — not adding")
            return caption, fixes

        healed = caption.rstrip() + "\n\n" + suffix
        fixes.append(f"  🔧  [bluesky] Added missing CTA suffix")
        return healed, fixes

    return caption, fixes


# ── Critical issue detection ────────────────────────────────────────────────

def has_bad_patterns(caption: str) -> bool:
    """Return True if caption contains generation error text."""
    stripped = caption.strip().lower()
    for pattern in BAD_PATTERNS:
        if pattern.lower() in stripped:
            return True
    return False


def is_empty_caption(caption: str) -> bool:
    """Return True if caption is empty or whitespace only."""
    return not caption or not caption.strip()


def is_suspiciously_short(caption: str, platform: str) -> bool:
    """Return True if caption is suspiciously short for the platform."""
    if platform == "pinterest":
        return len(caption.strip()) < 10  # Pinterest can be short
    return len(caption.strip()) < 20


def is_cropped_or_truncated(caption: str, platform: str) -> bool:
    """Return True if caption appears cut off mid-thought (AI Horde partial response)."""
    stripped = caption.strip()
    if len(stripped) < 30:
        return False  # Too short to judge — handled by is_suspiciously_short

    # Abrupt ending: no ending punctuation, no hashtag, no CTA marker, ends mid-word
    last_char = stripped[-1] if stripped else ""
    has_ending_punct = last_char in ".!?\n"
    has_hashtag_end = "#" in stripped[-30:]
    ends_with_comma = last_char == ","

    # If it ends with a comma or has no ending punctuation and is < 50% of limit,
    # it's likely truncated
    limit = PLATFORM_LIMITS.get(platform, 3000)
    if ends_with_comma:
        return True
    if not has_ending_punct and not has_hashtag_end and len(stripped) < limit * 0.5:
        return True

    # Check for abrupt mid-sentence cut: last "word" is incomplete
    words = stripped.split()
    if words and len(words[-1]) < 3 and len(stripped) > 50:
        return True

    return False


# ── Recovery: AI Horde regeneration + fallback ─────────────────────────────

def _load_posts() -> list[dict]:
    """Load posts.json and return the posts list."""
    try:
        with open(POSTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("posts", [])
    except Exception:
        return []


def _get_caption_prompt(post_id: int) -> str | None:
    """Get the caption_prompt for a given post_id from posts.json."""
    for p in _load_posts():
        if isinstance(p, dict) and p.get("id") == post_id:
            return p.get("caption_prompt", "")
    return None


def _generate_caption_for_platform(
    platform: str,
    post_id: int,
    caption_prompt: str,
) -> str | None:
    """
    Try to regenerate a caption via AI Horde for the given platform.

    Uses the caption_prompt from posts.json as the seed topic.
    Falls back to None if AI Horde is unavailable or returns bad output.
    """
    try:
        # Lazy import — only when regeneration is needed
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "scripts"))
        from agent_orchestrator import generate_caption_for_platform as gen_cap

        topic = {
            "title": f"Post {post_id}",
            "topic": caption_prompt,
        }

        brand = {
            "name": "The Nine Stitches",
            "hashtag": "#TheNineStitches",
            "persona": "reflective storyteller",
            "book_title": "The Nine Stitches",
        }

        caption = gen_cap(platform, topic, caption_prompt, brand)
        if caption and not has_bad_patterns(caption) and not is_empty_caption(caption):
            return caption
    except Exception as e:
        print(f"  ⚠ AI Horde regeneration failed for {platform}: {e}")

    return None


def _build_fallback_caption(
    platform: str,
    caption_prompt: str,
    post_id: int,
) -> str:
    """Build a deterministic fallback caption from caption_prompt when AI Horde fails."""
    title = f"Post {post_id}"

    if platform == "instagram":
        return f"{title}\n\n{caption_prompt}"
    elif platform == "linkedin":
        # LinkedIn: title + prompt + hashtag, formatted for professional audience
        return f"{title}\n\n{caption_prompt}\n\n#TheNineStitches"
    elif platform == "threads":
        preview = caption_prompt[:300].rstrip()
        return f"{title}: {preview}..."
    elif platform == "bluesky":
        return f"{title} — {caption_prompt[:280]}"
    elif platform == "youtube":
        return f"{title}\n\n{caption_prompt}\n\n#TheNineStitches"
    elif platform == "pinterest":
        return f"{title}: {caption_prompt}"
    else:
        return f"{title}\n\n{caption_prompt}"


def recover_caption(
    platform: str,
    post_id: int,
    existing_caption: str,
) -> tuple[str | None, str]:
    """
    Attempt to recover a problematic caption.

    Tries AI Horde regeneration first, then falls back to a deterministic
    caption built from caption_prompt.

    Returns (caption, recovery_method) where recovery_method is one of:
    - "regenerated": AI Horde produced a fresh caption
    - "fallback": Used deterministic fallback caption
    - "none": All recovery attempts failed
    """
    caption_prompt = _get_caption_prompt(post_id)
    if not caption_prompt:
        return None, "none"

    # Try AI Horde regeneration
    regenerated = _generate_caption_for_platform(platform, post_id, caption_prompt)
    if regenerated:
        return regenerated, "regenerated"

    # Fallback: deterministic caption from caption_prompt
    fallback = _build_fallback_caption(platform, caption_prompt, post_id)
    if fallback and not has_bad_patterns(fallback) and not is_empty_caption(fallback):
        return fallback, "fallback"

    return None, "none"


# ── Main heal loop ──────────────────────────────────────────────────────────

def heal_platform(platform: str, state: dict) -> dict:
    """
    Self-heal a single platform's caption.

    Returns a report dict with:
      - platform: str
      - status: "healed" | "clean" | "skipped" | "flagged"
      - fixes: list of strings describing what was fixed
      - warnings: list of strings describing unfixable issues
      - caption: the (possibly healed) caption to use
    """
    report = {
        "platform": platform,
        "status": "clean",
        "fixes": [],
        "warnings": [],
        "caption": None,
    }

    active = state.get("active_bundle")
    if not isinstance(active, dict):
        report["warnings"].append(f"  ℹ️  [{platform}] No active bundle — skipping")
        report["status"] = "skipped"
        return report

    post_id = active.get("post_id")
    if post_id is None:
        report["warnings"].append(f"  ❌ [{platform}] No post_id in active bundle — cannot recover")
        report["status"] = "skipped"
        return report
    post_id = int(post_id)

    state_caption = active.get("captions", {}).get(platform, "")

    # Read current caption.txt (or derive from state.json if missing)
    if not CAPTION_PATH.exists():
        # caption.txt doesn't exist yet — state.json is the only source
        if not state_caption:
            report["warnings"].append(
                f"  ❌ [{platform}] No caption in state.json for bundle {post_id} — cannot post"
            )
            report["status"] = "skipped"
            return report

        # Derive from state.json, apply tailoring, WRITE caption.txt
        report["fixes"].append(f"  🔧  [{platform}] No caption.txt — derived from state.json")
        prepared_caption = apply_platform_tailoring(state_caption, platform)
        report["caption"] = prepared_caption
        report["status"] = "healed"

        # Write caption.txt (first-time creation)
        with open(CAPTION_PATH, "w", encoding="utf-8") as f:
            f.write(prepared_caption)
        print(f"  ✅ [{platform}] caption.txt created ({len(prepared_caption)} chars)")
    else:
        with open(CAPTION_PATH, encoding="utf-8") as f:
            prepared_caption = f.read().strip()

    # ── Critical checks (can't auto-fix) ──

    if is_empty_caption(prepared_caption):
        # Attempt recovery before skipping
        recovered, method = recover_caption(platform, post_id, prepared_caption)
        if recovered:
            report["fixes"].append(
                f"  🔧  [{platform}] Empty caption recovered via {method}"
            )
            prepared_caption = recovered
            report["status"] = "healed"
            # Write recovered caption
            with open(CAPTION_PATH, "w", encoding="utf-8") as f:
                f.write(prepared_caption)
            print(f"  ✅ [{platform}] caption.txt recovered ({len(prepared_caption)} chars)")
        else:
            report["warnings"].append(
                f"  ❌ [{platform}] caption.txt is empty and recovery failed — "
                f"skipping this platform"
            )
            report["status"] = "skipped"
            return report

    if has_bad_patterns(prepared_caption):
        # Attempt recovery before skipping
        recovered, method = recover_caption(platform, post_id, prepared_caption)
        if recovered:
            report["fixes"].append(
                f"  🔧  [{platform}] Broken caption recovered via {method}"
            )
            prepared_caption = recovered
            report["status"] = "healed"
            # Write recovered caption
            with open(CAPTION_PATH, "w", encoding="utf-8") as f:
                f.write(prepared_caption)
            print(f"  ✅ [{platform}] caption.txt recovered ({len(prepared_caption)} chars)")
        else:
            report["warnings"].append(
                f"  ❌ [{platform}] caption.txt contains generation error text and "
                f"recovery failed — skipping this platform"
            )
            report["status"] = "skipped"
            return report

    if is_suspiciously_short(prepared_caption, platform):
        # Attempt recovery for suspiciously short captions
        recovered, method = recover_caption(platform, post_id, prepared_caption)
        if recovered:
            report["fixes"].append(
                f"  🔧  [{platform}] Short caption recovered via {method} "
                f"({len(prepared_caption.strip())} → {len(recovered.strip())} chars)"
            )
            prepared_caption = recovered
            report["status"] = "healed"
            with open(CAPTION_PATH, "w", encoding="utf-8") as f:
                f.write(prepared_caption)
            print(f"  ✅ [{platform}] caption.txt recovered ({len(prepared_caption)} chars)")
        else:
            report["warnings"].append(
                f"  ⚠️  [{platform}] caption.txt is suspiciously short "
                f"({len(prepared_caption.strip())} chars) and recovery failed — "
                f"flagged for review"
            )
            report["status"] = "flagged"

    if is_cropped_or_truncated(prepared_caption, platform):
        # Attempt recovery for cropped/truncated captions
        recovered, method = recover_caption(platform, post_id, prepared_caption)
        if recovered:
            report["fixes"].append(
                f"  🔧  [{platform}] Truncated caption recovered via {method}"
            )
            prepared_caption = recovered
            report["status"] = "healed"
            with open(CAPTION_PATH, "w", encoding="utf-8") as f:
                f.write(prepared_caption)
            print(f"  ✅ [{platform}] caption.txt recovered ({len(prepared_caption)} chars)")
        else:
            report["warnings"].append(
                f"  ⚠️  [{platform}] caption.txt appears truncated and recovery failed — "
                f"flagged for review"
            )
            report["status"] = "flagged"

    # ── Auto-heal: state drift ──

    healed, drift_fixes = heal_drift(state_caption, platform, prepared_caption)
    report["fixes"].extend(drift_fixes)
    if drift_fixes:
        report["status"] = "healed"
        prepared_caption = healed

    # ── Auto-heal: length ──

    healed, length_fixes = heal_length(prepared_caption, platform)
    report["fixes"].extend(length_fixes)
    if length_fixes:
        report["status"] = "healed"
        prepared_caption = healed

    # ── Auto-heal: hashtags ──

    healed, hashtag_fixes = heal_hashtags(prepared_caption, platform)
    report["fixes"].extend(hashtag_fixes)
    if hashtag_fixes:
        report["status"] = "healed"
        prepared_caption = healed

    # ── Auto-heal: Bluesky CTA ──

    if platform == "bluesky":
        healed, cta_fixes = heal_bluesky_cta(prepared_caption)
        report["fixes"].extend(cta_fixes)
        if cta_fixes:
            report["status"] = "healed"
            prepared_caption = healed

    # ── Write back healed caption.txt ──

    if report["fixes"]:
        with open(CAPTION_PATH, "w", encoding="utf-8") as f:
            f.write(prepared_caption)
        print(f"  ✅ [{platform}] caption.txt updated ({len(prepared_caption)} chars)")

    report["caption"] = prepared_caption

    # ── Soft checks (informational, don't affect status) ──

    limit = PLATFORM_LIMITS.get(platform, 3000)
    length = len(prepared_caption)

    if length > limit * 0.9 and length <= limit:
        report["warnings"].append(
            f"  ℹ️  [{platform}] Caption ({length} chars) near limit ({limit}) — "
            f"may truncate on some clients"
        )

    if platform == "instagram" and length > 125:
        preview = prepared_caption[:125].rsplit(" ", 1)[0]
        report["warnings"].append(
            f"  ℹ️  [{platform}] Feed preview (~125 chars): '{preview}...' — "
            f"key message should lead"
        )

    if platform == "linkedin" and length > 1500:
        report["warnings"].append(
            f"  ℹ️  [{platform}] Long caption ({length} chars) — 'see more' fold "
            f"will hide most text"
        )

    # Hashtag count warning (after healing)
    hashtags = re.findall(r"#\w+", prepared_caption)
    if len(hashtags) > 8:
        report["warnings"].append(
            f"  ℹ️  [{platform}] {len(hashtags)} hashtags — approaching spam threshold"
        )

    if not report["warnings"] and not report["fixes"]:
        report["status"] = "clean"

    return report


def run(platform: str | None = None, state_path: str | None = None):
    """Run Brand Guardian self-healing. Exit 0 = OK, 1 = unfixable issues."""
    state_path = state_path or str(STATE_PATH)

    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        print(f"❌ Cannot load state ({state_path}): {e}")
        return 1

    active = state.get("active_bundle")
    if not isinstance(active, dict):
        print("✅ No active bundle — nothing to heal")
        return 0

    post_id = active.get("post_id")
    if post_id is None:
        print("✅ No post_id in active bundle — nothing to heal")
        return 0
    post_id = int(post_id)
    print(f"🔍 Brand Guardian self-healing for bundle {post_id}")
    print(f"   Image: {active.get('image', 'N/A')}")
    print(f"   Reel:  {active.get('reel', 'N/A')}")
    print()

    platforms_to_check = [platform] if platform else list(active.get("captions", {}).keys())
    all_fixes = []
    all_warnings = []
    any_skipped = False

    for plat in platforms_to_check:
        print(f"── {plat.upper()} ──")

        caption = active.get("captions", {}).get(plat, "")
        if not caption:
            print(f"  ⏭  No caption for {plat} — skipping")
            continue

        report = heal_platform(plat, state)

        if report["fixes"]:
            print(f"  🔧  Healed ({len(report['fixes'])} fix(es)):")
            for fix in report["fixes"]:
                print(fix)

        if report["warnings"]:
            print(f"  ⚠️  Warnings ({len(report['warnings'])} issue(s)):")
            for warning in report["warnings"]:
                print(warning)

        if report["status"] == "skipped":
            any_skipped = True
            print(f"  ❌ {plat} SKIPPED — unfixable issue after recovery attempts")

        if report["status"] == "flagged":
            print(f"  ⚠️  {plat} FLAGGED — review recommended")

        if report["status"] == "clean":
            print(f"  ✅ {plat} clean — no issues")

        if report["status"] == "healed":
            print(f"  ✅ {plat} healed — issues auto-fixed")

        if report["caption"] is not None:
            all_fixes.append((plat, report["caption"]))

        print()

    # Summary
    print("── Summary ──")
    if all_fixes:
        print(f"  🔧  Processed {len(all_fixes)} platform(s):")
        for plat, cap in all_fixes:
            print(f"    {plat}: {len(cap)} chars")

    if any_skipped:
        print(f"  ❌ Platform(s) skipped after all recovery attempts exhausted")
        print(f"      Pipeline will continue — skipped platforms won't be posted")

    if all_warnings:
        print(f"  ℹ️  {len(all_warnings)} soft warning(s) — review recommended but pipeline continues")

    if not all_fixes and not all_warnings and not any_skipped:
        print("  ✅ All platforms clean — no issues")

    if any_skipped:
        print()
        print("❌ Brand Guardian: some platforms skipped — check warnings above")
        return 1

    print()
    print("✅ Brand Guardian: context integrity maintained")
    return 0


if __name__ == "__main__":
    platform = sys.argv[1] if len(sys.argv) > 1 else None
    state_path = sys.argv[2] if len(sys.argv) > 2 else None
    exit_code = run(platform, state_path)
    sys.exit(exit_code)
