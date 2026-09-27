#!/usr/bin/env python3
"""
Brand Guardian — pre-publish quality gate for ig-autobot captions.

Runs as a CI step between prepare_assets and publish. For each pending
caption, it checks:

1. Platform-specificity — caption length, formatting, CTA presence
2. Cropping safety — text near edges that platform UIs may cover
3. State drift — caption.txt matches state.json master_reflection
4. Content sanity — no placeholders, no double hashtags, no blank captions

Exit 0 = all clear. Exit 1 = issues found (prints them, does NOT block).
"""

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE_PATH = REPO / "state.json"

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


def load_state() -> dict:
    if not STATE_PATH.exists():
        print("⚠ state.json not found — cannot check state drift")
        return {}
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def check_caption_length(caption: str, platform: str) -> list[str]:
    """Warn if caption exceeds platform limit or is suspiciously short."""
    issues = []
    limit = PLATFORM_LIMITS.get(platform, 3000)
    length = len(caption)

    if length > limit:
        issues.append(
            f"  ⚠️  [{platform}] Caption length {length} exceeds "
            f"platform limit of {limit} chars (overflow by {length - limit})"
        )
    elif length > limit * 0.9:
        issues.append(
            f"  ⚡ [{platform}] Caption length {length} is close to "
            f"limit ({limit}) — may get truncated on some clients"
        )

    # Flag suspiciously short captions (likely generation failure)
    if length < 20 and platform != "pinterest":
        issues.append(
            f"  ⚠️  [{platform}] Caption too short ({length} chars) — "
            f"may be a generation failure or placeholder"
        )
    return issues


def check_platform_specificity(caption: str, platform: str) -> list[str]:
    """Check that the caption looks tailored for the target platform."""
    issues = []
    stripped = caption.strip()

    if not stripped:
        issues.append(f"  ❌ [{platform}] Empty caption — will post blank")
        return issues

    # Check for bad patterns (generation failures)
    for pattern in BAD_PATTERNS:
        if pattern.lower() in stripped.lower():
            issues.append(
                f"  ❌ [{platform}] Caption contains generation error pattern: "
                f"'{pattern}' — do not post"
            )
            return issues

    # Platform-specific checks
    if platform == "bluesky":
        suffix = "Want to read more?... check out my LinkedIn"
        if suffix not in stripped:
            issues.append(
                f"  ⚠️  [{platform}] Caption missing Bluesky CTA suffix — "
                f"readers may not discover the LinkedIn post"
            )

    if platform == "threads":
        if len(stripped) > 420:
            issues.append(
                f"  ⚠️  [{platform}] Caption ({len(stripped)} chars) exceeds "
                f"Threads 420-char limit — need trimming"
            )
        # Threads captions should be punchier — flag if too formal/standalone
        # (this is a soft check, not a hard block)

    if platform == "youtube":
        if "#" not in stripped:
            issues.append(
                f"  ⚠️  [{platform}] YouTube caption has no hashtags — "
                f"reduces discoverability"
            )

    if platform == "linkedin":
        # LinkedIn likes spacing between paragraphs
        lines = stripped.split("\n")
        short_lines = [l for l in lines if len(l.strip()) < 10 and l.strip()]
        if len(short_lines) > len(lines) * 0.4:
            issues.append(
                f"  ⚠️  [{platform}] Caption looks dense ({len(short_lines)} "
                f"short lines out of {len(lines)}) — may reduce engagement"
            )

    return issues


def check_hashtag_safety(caption: str, platform: str) -> list[str]:
    """Warn about hashtag issues that could look spammy or break formatting."""
    issues = []
    if platform not in HASHTAG_INLINE:
        return issues

    import re
    hashtags = re.findall(r"#\w+", caption)
    if len(hashtags) > 10:
        issues.append(
            f"  ⚠️  [{platform}] {len(hashtags)} hashtags detected — "
            f"may look spammy on {platform}"
        )

    # Check for double hashtags (##) which break IG parsing
    if "##" in caption:
        issues.append(
            f"  ❌ [{platform}] Double hashtag '##' found — breaks IG hashtag parsing"
        )

    return issues


def check_cropping_safety(caption: str, platform: str) -> list[str]:
    """
    Check if caption content is at risk of being cropped by platform UI.

    Different platforms overlay UI elements on different parts of the caption:
    - Instagram: truncates in feed preview (shows ~125 chars, '… more')
    - LinkedIn: may hide text behind 'see more' after ~3 lines
    - Threads: hard limit at 420 chars
    - Bluesky: hard limit at 300 chars
    """
    issues = []
    stripped = caption.strip()
    length = len(stripped)

    if platform == "instagram" and length > 125:
        # Show what gets cut off
        preview = stripped[:125].rsplit(" ", 1)[0]
        issues.append(
            f"  ℹ️  [{platform}] Caption ({length} chars) will show ~125 chars "
            f"in feed ('{preview}...') — key message should be in first 125 chars"
        )

    if platform == "linkedin" and length > 1500:
        issues.append(
            f"  ℹ️  [{platform}] Long caption ({length} chars) — 'see more' "
            f"fold will hide most text; lead with the hook"
        )

    return issues


def check_state_drift(state: dict, platform: str) -> list[str]:
    """
    Verify caption.txt matches what state.json says for the active bundle.

    Drift happens when:
    - caption.txt was edited manually after prepare_assets ran
    - The active bundle changed but caption.txt wasn't regenerated
    - prepare_assets wrote a different caption than what's in state.json
    """
    issues = []

    active = state.get("active_bundle")
    if not isinstance(active, dict):
        return issues  # No active bundle — nothing to check

    post_id = active.get("post_id")
    state_caption = active.get("captions", {}).get(platform, "")

    if not state_caption:
        issues.append(
            f"  ⚠️  [{platform}] No caption found in state.json for "
            f"bundle {post_id} — caption.txt may be stale"
        )
        return issues

    # Read prepared caption.txt
    caption_path = REPO / "caption.txt"
    if not caption_path.exists():
        issues.append(
            f"  ❌ [{platform}] caption.txt missing — prepare_assets did not run?"
        )
        return issues

    with open(caption_path, encoding="utf-8") as f:
        prepared_caption = f.read().strip()

    if not prepared_caption:
        issues.append(
            f"  ❌ [{platform}] caption.txt is empty but state.json has "
            f"a caption for bundle {post_id} — mismatch"
        )
        return issues

    # Normalize both for comparison (strip markdown, normalize whitespace)
    def normalize(text: str) -> str:
        text = text.replace("**", "").replace("*", "")
        text = " ".join(text.split())
        return text.lower()

    state_norm = normalize(state_caption)
    prepared_norm = normalize(prepared_caption)

    if state_norm != prepared_norm:
        # Check if it's just a trailing suffix difference (normal for tailoring)
        state_core = state_norm
        prepared_core = prepared_norm

        # For bluesky, the suffix is added by prepare_assets — allow it
        if platform == "bluesky":
            suffix = "want to read more?... check out my linkedin"
            if prepared_norm.endswith(suffix) and state_norm in prepared_norm:
                return issues  # Tailoring added suffix — OK
            if state_norm.endswith(suffix) and prepared_norm in state_norm:
                return issues

        # For threads, trimming happens — allow if core matches
        if platform == "threads":
            if state_core[:400] == prepared_core[:400]:
                return issues  # Trimming — OK

        issues.append(
            f"  ⚠️  [{platform}] STATE DRIFT detected for bundle {post_id}:\n"
            f"       state.json caption: {state_caption[:100]}...\n"
            f"       caption.txt:        {prepared_caption[:100]}...\n"
            f"       These don't match — caption.txt may be from a different bundle"
        )

    return issues


def check_image_exists(state: dict, platform: str) -> list[str]:
    """Verify the output image exists for platforms that need it."""
    issues = []
    active = state.get("active_bundle")
    if not isinstance(active, dict):
        return issues

    image_field = active.get("image", "")
    if not image_field:
        issues.append(
            f"  ⚠️  [{platform}] No image field in active bundle — "
            f"post may be text-only"
        )
        return issues

    # Check local copy
    output_path = REPO / "output.jpg"
    if not output_path.exists():
        issues.append(
            f"  ❌ [{platform}] output.jpg not found — "
            f"prepare_assets did not copy the image"
        )
        return issues

    size = output_path.stat().st_size
    if size < 10000:
        issues.append(
            f"  ⚠️  [{platform}] output.jpg is only {size} bytes — "
            f"may be a corrupted or placeholder image"
        )

    return issues


def run(platform: str | None = None, state_path: str | None = None):
    """Run Brand Guardian checks. Exit 0 = pass, 1 = issues found."""
    state_path = state_path or str(STATE_PATH)
    state = load_state() if not state_path else {}

    if not state:
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            print(f"❌ Cannot load state: {e}")
            return 1

    active = state.get("active_bundle")
    if not isinstance(active, dict):
        print("✅ No active bundle — nothing to check")
        return 0

    post_id = active.get("post_id")
    print(f"🔍 Brand Guardian check for bundle {post_id}")
    print(f"   Image: {active.get('image', 'N/A')}")
    print(f"   Reel:  {active.get('reel', 'N/A')}")
    print()

    platforms_to_check = [platform] if platform else list(active.get("captions", {}).keys())
    all_issues = []

    for plat in platforms_to_check:
        print(f"── {plat.upper()} ──")

        caption = active.get("captions", {}).get(plat, "")
        if not caption:
            print(f"  ⏭  No caption for {plat} — skipping")
            continue

        issues = []
        issues.extend(check_caption_length(caption, plat))
        issues.extend(check_platform_specificity(caption, plat))
        issues.extend(check_hashtag_safety(caption, plat))
        issues.extend(check_cropping_safety(caption, plat))
        issues.extend(check_state_drift(state, plat))
        issues.extend(check_image_exists(state, plat))

        if issues:
            for issue in issues:
                print(issue)
            all_issues.extend(issues)
        else:
            print(f"  ✅ {plat} caption looks good")

        print()

    if all_issues:
        print(f"⚠️  Brand Guardian found {len(all_issues)} issue(s) — review before publishing")
        return 1
    else:
        print("✅ Brand Guardian: all checks passed")
        return 0


if __name__ == "__main__":
    platform = sys.argv[1] if len(sys.argv) > 1 else None
    state_path = sys.argv[2] if len(sys.argv) > 2 else None
    exit_code = run(platform, state_path)
    sys.exit(exit_code)
