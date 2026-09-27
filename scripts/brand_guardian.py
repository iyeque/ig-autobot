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
