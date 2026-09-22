#!/usr/bin/env python3
"""
Publisher Agent for ig-autobot.

End-to-end per-platform pipeline: prepare assets -> (brand) -> publish -> VERIFY.
A platform is only reported "posted" if state.json actually records it.

Usage:
    python scripts/agent_publisher.py --brand main
    python scripts/agent_publisher.py --brand main --platforms linkedin,bluesky
    python scripts/agent_publisher.py --brand main --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
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

from shared_utils import (  # noqa: E402
    load_state,
    is_platform_posted,
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
            "instagram": "publish.main",
            "linkedin": "publish_linkedin.publish_to_linkedin_rest",
            "bluesky": "publish_bluesky.publish_to_bluesky",
            "threads": "publish_threads.publish_to_threads",
            "pinterest": "publish_pinterest.publish_to_pinterest",
            "youtube": "publish_youtube.publish_to_youtube",
        },
        # Platforms whose CI workflow brands output.jpg before publishing
        "brand_step": {"linkedin"},
    },
    "wilma": {
        "name": "DigitalGuard",
        "state_path": WILMA_STATE_PATH,
        "platforms": ["linkedin", "bluesky"],
        "publish_functions": {
            "linkedin": "publish_wilma_linkedin.publish_to_linkedin_rest",
            "bluesky": "publish_wilma_bluesky.publish_wilma_to_bluesky",
        },
        "brand_step": set(),
    },
}


# ── Utility ──────────────────────────────────────────────────────────────
def load_publish_function(ref: str) -> Callable:
    """Load a publish function from 'module.function' searching scripts/ then forwilma/."""
    module_path, func_name = ref.rsplit(".", 1)
    for base in [SCRIPTS_DIR, FORWILMA_DIR]:
        mod_file = base / f"{module_path}.py"
        if mod_file.exists():
            spec = importlib.util.spec_from_file_location(module_path, mod_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_path] = module
                spec.loader.exec_module(module)
                return getattr(module, func_name)
    raise ImportError(f"Cannot find publish function: {ref}")


def run_step(cmd: list[str], cwd: Path, label: str) -> bool:
    """Run a subprocess step, streaming output. Returns True on exit 0."""
    print(f"    $ {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"    ✗ {label} timed out after 300s")
        return False
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.strip().splitlines():
        print(f"    | {line}")
    if r.returncode != 0:
        print(f"    ✗ {label} exited {r.returncode}")
        return False
    return True


def verify_posted(platform: str, state_path: str, post_id: Any) -> bool:
    """True only if state.json actually records this post_id for the platform."""
    state = load_state(state_path)
    return bool(post_id in state.get("platform_posted_bundles", {}).get(platform, []))


def publish_with_retry(func: Callable, max_retries: int = 2, delay: int = 5) -> tuple[bool, str]:
    """Run a publish function with retry. Scripts sys.exit(1) on failure — catch that too."""
    msg = ""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Attempt {attempt}/{max_retries}...")
            func()
            return True, "returned"
        except SystemExit as e:
            msg = f"exited {e.code}"
            print(f"    ✗ Attempt {attempt}: script exited {e.code}")
        except Exception as e:  # noqa: BLE001
            msg = str(e)[:120]
            print(f"    ✗ Attempt {attempt} failed: {msg}")
        if attempt < max_retries:
            print(f"    Retrying in {delay}s...")
            time.sleep(delay)
    return False, msg


# ── Main publisher ────────────────────────────────────────────────────────
def publish_main(brand: dict, platforms: list[str] | None = None, dry_run: bool = False) -> dict:
    """Per platform: prepare -> (brand) -> publish -> verify. Sequential, no concurrent state writes."""
    state_path = str(brand["state_path"])
    active = get_active_bundle(state_path)
    if not active:
        print("[publisher] No active bundle, attempting advance...")
        advance_stale_active_bundle(state_path)
        active = get_active_bundle(state_path)
        if not active:
            print("[publisher] ✗ No active bundle found. Nothing to publish.")
            return {"status": "no_bundle", "platforms": {}}

    post_id = active.get("post_id")
    print(f"[publisher] Bundle {post_id} | image: {active.get('image')}")
    print(f"[publisher] Platforms: {', '.join(platforms or brand['platforms'])}\n")

    results = {}
    for platform in (platforms or brand["platforms"]):
        print(f"── {platform.upper()} ──")

        if is_platform_posted(platform, state_path):
            print("  ⏭ Already posted (state.json). Skipping.\n")
            results[platform] = "already_posted"
            continue

        if dry_run:
            print("  ⏭ DRY RUN — would prepare + publish\n")
            results[platform] = "dry_run"
            continue

        # 1. Prepare per-platform assets (caption.txt, ready flag, output.jpg)
        if not run_step([sys.executable, "scripts/prepare_assets.py", "--platform", platform], REPO, "prepare"):
            results[platform] = "prepare_failed"
            continue

        # 2. Brand step (LinkedIn: logo + text overlay on output.jpg)
        if platform in brand.get("brand_step", set()):
            if not run_step([sys.executable, "scripts/brand_linkedin_image.py"], REPO, "brand"):
                results[platform] = "brand_failed"
                continue

        # 3. Publish
        try:
            func = load_publish_function(brand["publish_functions"][platform])
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ Load failed: {e}\n")
            results[platform] = f"load_error: {e}"
            continue

        ok, msg = publish_with_retry(func)

        # 4. VERIFY against state.json — never trust exit status alone
        if verify_posted(platform, state_path, post_id):
            print(f"  ✓ VERIFIED posted to {platform} (recorded in state.json)\n")
            results[platform] = "posted"
        elif ok:
            print(f"  ⚠ Script returned OK but state does NOT record {platform} — treated as NO-OP\n")
            results[platform] = "no_op_skipped"
        else:
            print(f"  ✗ FAILED: {msg}\n")
            results[platform] = f"failed: {msg}"

    return {"status": "done", "post_id": post_id, "platforms": results}


def publish_wilma(brand: dict, day: int | None = None, platforms: list[str] | None = None, dry_run: bool = False) -> dict:
    """Publish Wilma bundle (prepare runs from forwilma/ cwd)."""
    state_path = str(brand["state_path"])
    state = load_state(state_path)
    queue = state.get("content_queue", [])
    target = None
    if day:
        for item in queue:
            if item.get("post_id") == f"day_{day}" or item.get("day") == day:
                target = item
                break
    else:
        target = state.get("active_bundle") or (queue[0] if queue else None)

    if not target:
        print("[publisher] ✗ No Wilma bundle found.")
        return {"status": "no_bundle", "platforms": {}}

    print(f"[publisher] Wilma bundle {target.get('post_id')}\n")
    results = {}
    for platform in (platforms or brand["platforms"]):
        print(f"── {platform.upper()} ──")
        if dry_run:
            print("  ⏭ DRY RUN\n")
            results[platform] = "dry_run"
            continue
        if not run_step([sys.executable, "prepare_assets.py", "--platform", platform], FORWILMA_DIR, "prepare"):
            results[platform] = "prepare_failed"
            continue
        try:
            func = load_publish_function(brand["publish_functions"][platform])
        except Exception as e:  # noqa: BLE001
            results[platform] = f"load_error: {e}"
            continue
        ok, msg = publish_with_retry(func)
        results[platform] = "posted" if ok else f"failed: {msg}"
    return {"status": "done", "post_id": target.get("post_id"), "platforms": results}


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-platform publisher agent for ig-autobot")
    parser.add_argument("--brand", choices=["main", "wilma"], default="main")
    parser.add_argument("--platforms", type=str, default=None, help="Comma-separated list (e.g. linkedin,bluesky)")
    parser.add_argument("--day", type=int, default=None, help="Wilma day number")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual publishing")
    args = parser.parse_args()

    brand = BRAND[args.brand]
    platforms = [p.strip() for p in args.platforms.split(",")] if args.platforms else None

    if args.brand == "main":
        result = publish_main(brand, platforms=platforms, dry_run=args.dry_run)
    else:
        result = publish_wilma(brand, day=args.day, platforms=platforms, dry_run=args.dry_run)

    print(f"\n[publisher] Result: {result['status']}")
    for p, r in result.get("platforms", {}).items():
        print(f"  {p}: {r}")


if __name__ == "__main__":
    main()
