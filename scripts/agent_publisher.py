#!/usr/bin/env python3
"""
Publisher Agent for ig-autobot.

Reads the active bundle from state.json and publishes to all prepared platforms.
Wraps the per-platform publish functions with retry logic, state tracking,
and error recovery.

Usage:
    python scripts/agent_publisher.py --brand main
    python scripts/agent_publisher.py --brand main --platforms linkedin,bluesky
    python scripts/agent_publisher.py --brand main --dry-run
    python scripts/agent_publisher.py --brand wilma --day 12
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# ── Paths ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO / "scripts"
FORWILMA_DIR = REPO / "forwilma"
STATE_PATH = REPO / "state.json"
WILMA_STATE_PATH = FORWILMA_DIR / "state.json"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS_DIR))

# ── Load shared helpers ──────────────────────────────────────────────────
from shared_utils import (  # noqa: E402
    load_state,
    save_state,
    is_platform_posted,
    update_state_after_post,
    get_active_bundle,
    advance_stale_active_bundle,
)


# ── Brand configuration ──────────────────────────────────────────────────
BRAND = {
    "main": {
        "name": "WP WIGMAN",
        "state_path": STATE_PATH,
        "platforms": ["instagram", "linkedin", "bluesky", "threads", "pinterest", "youtube"],
        "publish_functions": {
            "linkedin": "publish_linkedin.publish_to_linkedin_rest",
            "bluesky": "publish_bluesky.publish_to_bluesky",
            "threads": "publish_threads.publish_to_threads",
            "pinterest": "publish_pinterest.publish_to_pinterest",
            "youtube": "publish_youtube.publish_to_youtube",
            "instagram": "publish.main",
        },
        "requires_media": ["instagram", "linkedin", "bluesky", "threads", "pinterest"],
    },
    "wilma": {
        "name": "DigitalGuard",
        "state_path": WILMA_STATE_PATH,
        "platforms": ["linkedin", "bluesky"],
        "publish_functions": {
            "linkedin": "publish_wilma_linkedin.publish_to_linkedin_rest",
            "bluesky": "publish_wilma_bluesky.publish_wilma_to_bluesky",
        },
        "requires_media": ["linkedin", "bluesky"],
    },
}


# ── Utility ──────────────────────────────────────────────────────────────
def load_publish_function(ref: str) -> Callable:
    """Load a publish function from a module path like 'publish_linkedin.publish_to_linkedin_rest'."""
    module_path, func_name = ref.rsplit(".", 1)
    # Try scripts/ first, then forwilma/
    for base in [SCRIPTS_DIR, FORWILMA_DIR]:
        mod_file = base / f"{module_path}.py"
        if mod_file.exists():
            # Use importlib with full path
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_path, mod_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_path] = module
                spec.loader.exec_module(module)
                return getattr(module, func_name)
    raise ImportError(f"Cannot find publish function: {ref}")


def publish_with_retry(
    func: Callable,
    platform: str,
    max_retries: int = 2,
    delay: int = 5,
) -> tuple[bool, str]:
    """Run a publish function with retry. Returns (success, message)."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Attempt {attempt}/{max_retries}...")
            func()
            return True, "posted"
        except Exception as e:
            msg = str(e)[:100]
            print(f"    ✗ Attempt {attempt} failed: {msg}")
            if attempt < max_retries:
                print(f"    Retrying in {delay}s...")
                time.sleep(delay)
    return False, msg


# ── Main publisher ────────────────────────────────────────────────────────
def publish_main(brand: dict, state: dict, platforms: list[str] | None = None, dry_run: bool = False) -> dict:
    """Publish main brand active bundle to platforms."""
    state_path = str(brand["state_path"])
    
    active = get_active_bundle(state_path)
    if not active:
        # Try to advance stale pending
        print("[publisher] No active bundle, attempting advance...")
        advance_stale_active_bundle(state_path)
        state = load_state(state_path)
        active = get_active_bundle(state_path)
        if not active:
            print("[publisher] ✗ No active bundle found. Nothing to publish.")
            return {"status": "no_bundle", "platforms": {}}

    post_id = active.get("post_id")
    captions = active.get("captions", {})
    image = active.get("image", "")
    platforms_to_publish = platforms or brand["platforms"]

    print(f"[publisher] Publishing main bundle post_id={post_id}")
    print(f"[publisher] Image: {image}")
    print(f"[publisher] Platforms: {', '.join(platforms_to_publish)}")

    results = {}
    for platform in platforms_to_publish:
        print(f"\n── {platform.upper()} ──")

        # Skip if already posted
        if is_platform_posted(platform, state_path):
            print(f"  ⏭ Already posted to {platform}. Skipping.")
            results[platform] = "already_posted"
            continue

        # Skip if no caption for this platform
        if not captions.get(platform):
            print(f"  ⏭ No caption for {platform}. Skipping.")
            results[platform] = "no_caption"
            continue

        if dry_run:
            print(f"  ⏭ DRY RUN — would publish to {platform}")
            results[platform] = "dry_run"
            continue

        # Load and run publish function
        func_ref = brand["publish_functions"].get(platform)
        if not func_ref:
            print(f"  ✗ No publish function for {platform}")
            results[platform] = "no_function"
            continue

        try:
            func = load_publish_function(func_ref)
            success, msg = publish_with_retry(func, platform)
            if success:
                print(f"  ✓ Published to {platform}")
                results[platform] = "posted"
                # Update state
                try:
                    update_state_after_post(platform, state_path)
                    state = load_state(state_path)
                except Exception as e:
                    print(f"  ⚠ State update failed: {e}")
            else:
                print(f"  ✗ Failed: {msg}")
                results[platform] = f"failed: {msg}"
        except ImportError as e:
            print(f"  ✗ Import error: {e}")
            results[platform] = f"import_error: {e}"
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results[platform] = f"error: {e}"

    return {"status": "done", "post_id": post_id, "platforms": results}


def publish_wilma(brand: dict, state: dict, day: int | None = None, platforms: list[str] | None = None, dry_run: bool = False) -> dict:
    """Publish Wilma bundle for given day."""
    queue = state.get("content_queue", [])
    if not queue:
        print("[publisher] ✗ No content_queue found.")
        return {"status": "no_queue", "platforms": {}}

    # Find the bundle for the target day
    target = None
    if day:
        for item in queue:
            if item.get("post_id") == f"day_{day}" or item.get("day") == day:
                target = item
                break
    else:
        target = queue[-1]  # Latest

    if not target:
        print(f"[publisher] ✗ No bundle found for day {day}")
        return {"status": "no_bundle", "platforms": {}}

    post_id = target.get("post_id")
    print(f"[publisher] Publishing Wilma bundle {post_id}")

    results = {}
    platforms_to_publish = platforms or brand["platforms"]

    for platform in platforms_to_publish:
        print(f"\n── {platform.upper()} ──")
        func_ref = brand["publish_functions"].get(platform)
        if not func_ref:
            print(f"  ✗ No publish function")
            results[platform] = "no_function"
            continue

        if dry_run:
            print(f"  ⏭ DRY RUN")
            results[platform] = "dry_run"
            continue

        try:
            func = load_publish_function(func_ref)
            success, msg = publish_with_retry(func, platform)
            results[platform] = "posted" if success else f"failed: {msg}"
        except Exception as e:
            results[platform] = f"error: {e}"

    return {"status": "done", "post_id": post_id, "platforms": results}


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-platform publisher agent for ig-autobot")
    parser.add_argument("--brand", choices=["main", "wilma"], default="main")
    parser.add_argument("--platforms", type=str, default=None, help="Comma-separated list (e.g. linkedin,bluesky)")
    parser.add_argument("--day", type=int, default=None, help="Wilma day number")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual publishing")
    args = parser.parse_args()

    brand = BRAND[args.brand]
    state = load_state(str(brand["state_path"]))
    platforms = [p.strip() for p in args.platforms.split(",")] if args.platforms else None

    if args.brand == "main":
        result = publish_main(brand, state, platforms=platforms, dry_run=args.dry_run)
    else:
        result = publish_wilma(brand, state, day=args.day, platforms=platforms, dry_run=args.dry_run)

    print(f"\n[publisher] Result: {result['status']}")
    for p, r in result.get("platforms", {}).items():
        print(f"  {p}: {r}")


if __name__ == "__main__":
    main()
