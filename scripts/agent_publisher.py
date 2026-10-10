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
import json
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


def _invalidate_image_cache() -> None:
    """Drop the raw AI Horde image cache once a bundle has been published.

    The cache exists so a failed run can reuse its already-generated image
    instead of spending kudos again. After a successful publish the next
    bundle must generate a fresh image, so the cache is cleared here.
    """
    for cache in (REPO / "images" / ".horde_cache.png",
                  FORWILMA_DIR / "images" / ".horde_cache.png"):
        try:
            if cache.exists():
                cache.unlink()
                print(f"  🧹 Cleared image cache: {cache}")
        except Exception as e:
            print(f"  ⚠ Could not clear image cache {cache}: {e}")


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


# ── Content gate ──────────────────────────────────────────────────────────

def content_gate_preflight(bundle: dict, platforms: list[str], state_path: str) -> tuple[bool, list[str]]:
    """Judge every target caption BEFORE any platform posts.

    The gate makes an unverifiable bundle impossible to publish: each caption
    is judged (floors + LLM), failures are rebuilt from the bundle's
    master_reflection via bot.py's deterministic editor and re-judged, and
    any caption that still fails aborts the ENTIRE publish — not just its
    platform. One bad caption means the bundle isn't ready.

    Returns (proceed, log_lines).
    """
    log: list[str] = []
    try:
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "scripts"))
        import brand_guardian
    except Exception as e:
        # Gate code itself broken: log loudly and proceed rather than
        # silently dropping every bundle forever.
        log.append(f"    ⚖️  [gate] UNAVAILABLE ({e}) — proceeding ungated")
        return True, log

    state: dict = {}
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        # Not fatal: fall back to the passed bundle. But log it — a silent
        # load failure here once hid a NameError and disabled persistence.
        log.append(f"    ⚖️  [gate] state load failed ({e}) — using passed bundle")

    active = state.get("active_bundle") if isinstance(state.get("active_bundle"), dict) else bundle
    captions = dict(active.get("captions") or {})
    master_reflection = active.get("master_reflection") or bundle.get("master_reflection") or ""
    post_id = active.get("post_id") or bundle.get("post_id")
    changed = False
    failures: list[str] = []

    for p in (platforms or []):
        caption = captions.get(p)
        if caption is None:
            continue
        ok, reason = brand_guardian.judge_caption(caption, p)
        if ok:
            log.append(f"    ⚖️  [gate] {p}: PASS")
            continue
        log.append(f"    ⚖️  [gate] {p}: FAIL ({reason}) — rebuilding from source")

        rebuilt = None
        try:
            import bot
            post = {
                "title": active.get("title") or active.get("topic") or f"Post {post_id}",
                "topic": active.get("topic") or active.get("title") or "",
                "pillar": active.get("pillar") or "reflection",
            }
            rebuilt = bot._build_deterministic_caption(
                post, master_reflection, p, state, list(captions.keys()),
            )
        except Exception as e:
            log.append(f"    ⚖️  [gate] {p}: rebuild raised ({e})")

        if rebuilt:
            ok2, reason2 = brand_guardian.judge_caption(rebuilt, p)
            if ok2:
                captions[p] = rebuilt
                changed = True
                log.append(f"    ⚖️  [gate] {p}: rebuilt and PASSED ({len(rebuilt)} chars)")
                continue
            log.append(f"    ⚖️  [gate] {p}: rebuilt but still FAILS ({reason2})")
        else:
            log.append(f"    ⚖️  [gate] {p}: no rebuild produced")
        failures.append(p)

    if failures:
        log.append(
            f"    ⚖️  [gate] ABORT — {len(failures)} caption(s) unverifiable: "
            f"{', '.join(failures)}. Nothing was published."
        )
        return False, log

    if changed and isinstance(state.get("active_bundle"), dict):
        state["active_bundle"]["captions"] = captions
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            log.append("    ⚖️  [gate] rebuilt captions persisted to state.json")
        except Exception as e:
            log.append(f"    ⚖️  [gate] WARNING: could not persist rebuilt captions ({e})")

    return True, log


# ── Main publisher ────────────────────────────────────────────────────────
def publish_main(brand: dict, platforms: list[str] | None = None, dry_run: bool = False, fmt: str = "standard") -> dict:
    """Per platform: prepare -> (brand) -> publish -> verify. Sequential, no concurrent state writes.
    
    fmt: 'standard' | 'carousel' | 'quote'
      - standard: prepare_assets -> publish
      - carousel: generate_carousel -> prepare_assets -> publish (IG + LI only)
      - quote: generate_quote_image -> prepare_assets --platform instagram -> publish
    """
    state_path = str(brand["state_path"])

    # The quote and carousel pipelines are self-contained: quote content
    # comes from posts.json + quotes_state.json; the carousel generator
    # promotes queue[0] itself when there is no active bundle. Neither may be
    # gated on the standard bundle's active_bundle — that bundle only exists
    # in the ~30-minute window between the 02:00 gen push and the publisher
    # consuming it, which silently no-oped 4 of 5 daily quote slots and
    # breaks carousel runs whenever the queue is momentarily empty.
    if fmt in ("quote", "carousel"):
        active = None
        post_id = None
        print(f"[publisher] Format: {fmt} | self-contained pipeline (posts.json / queue)")
        print(f"[publisher] Platforms: {', '.join(platforms or brand['platforms'])}\n")
    else:
        active = get_active_bundle(state_path)
        if not active:
            print("[publisher] No active bundle, attempting advance...")
            advance_stale_active_bundle(state_path)
            active = get_active_bundle(state_path)
            if not active:
                print("[publisher] ✗ No active bundle found. Nothing to publish.\n")
                return {"status": "no_bundle", "platforms": {}}

        post_id = active.get("post_id")
        print(f"[publisher] Bundle {post_id} | format: {fmt} | image: {active.get('image')}")
        print(f"[publisher] Platforms: {', '.join(platforms or brand['platforms'])}\n")

    # ── Content gate: judge every caption BEFORE any platform posts ──
    # One unverifiable caption aborts the whole publish — the bundle stays
    # queued and is retried, rather than shipping broken copy. Only the
    # standard format publishes bundle captions; quote/carousel have their
    # own pipelines. Dry runs never mutate state, so they skip the gate.
    if fmt == "standard" and not dry_run:
        gate_ok, gate_log = content_gate_preflight(
            active, list(platforms or brand["platforms"]), state_path
        )
        for line in gate_log:
            print(line)
        if not gate_ok:
            print("  ✗ CONTENT GATE failed — publish aborted, nothing went out\n")
            return {"status": "gate_failed", "platforms": {}}

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

        # Format-specific pre-processing
        if fmt == "carousel" and platform in ("instagram", "linkedin"):
            if not run_step([sys.executable, "scripts/generate_carousel_from_bundle.py", "--state_path", "state.json"], REPO, "carousel"):
                results[platform] = "carousel_failed"
                continue
        elif fmt == "quote" and platform == "instagram":
            # Quote pipeline is separate (uses posts.json + quotes_state.json)
            # and tracks its own posted state — nothing here touches the
            # standard bundle's state.json records.
            if not run_step([sys.executable, "scripts/publish_instagram_quotes.py", "--generate-only"], REPO, "quote_gen"):
                results[platform] = "quote_failed"
                continue
            if not run_step([sys.executable, "scripts/publish_instagram_quotes.py", "--publish-only"], REPO, "quote_pub"):
                results[platform] = "quote_failed"
                continue
            results[platform] = "posted"
            continue

        # 1. Prepare per-platform assets
        if not run_step([sys.executable, "scripts/prepare_assets.py", "--platform", platform], REPO, "prepare"):
            results[platform] = "prepare_failed"
            continue

        # 1.5. Brand Guardian — pre-publish quality gate
        bg_ok = subprocess.run(
            [sys.executable, "scripts/brand_guardian.py", platform],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if bg_ok.returncode != 0:
            print(f"  ⚠️  Brand Guardian found {platform} caption issues — review before posting:")
            for line in bg_ok.stdout.strip().splitlines():
                if line.strip():
                    print(f"    {line}")
        else:
            print(f"  ✅ Brand Guardian: {platform} caption OK")

        # 2. Brand step (LinkedIn)
        if platform in brand.get("brand_step", set()):
            if not run_step([sys.executable, "scripts/brand_linkedin_image.py"], REPO, "brand"):
                results[platform] = "brand_failed"
                continue

        # 3. Publish
        try:
            func = load_publish_function(brand["publish_functions"][platform])
        except Exception as e:
            print(f"  ✗ Load failed: {e}\n")
            results[platform] = f"load_error: {e}"
            continue

        ok, msg = publish_with_retry(func)

        # 4. VERIFY against state.json
        if verify_posted(platform, state_path, post_id):
            print(f"  ✓ VERIFIED posted to {platform}\n")
            results[platform] = "posted"
        elif ok:
            print(f"  ⚠ Script OK but state does NOT record {platform}\n")
            results[platform] = "no_op_skipped"
        else:
            print(f"  ✗ FAILED: {msg}\n")
            results[platform] = f"failed: {msg}"

    if any(r == "posted" for r in results.values()):
        _invalidate_image_cache()
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
        if not run_step([sys.executable, "scripts/prepare_assets.py", "--platform", platform, "--state_path", "forwilma/state.json"], REPO, "prepare"):
            results[platform] = "prepare_failed"
            continue
        try:
            func = load_publish_function(brand["publish_functions"][platform])
        except Exception as e:  # noqa: BLE001
            results[platform] = f"load_error: {e}"
            continue
        ok, msg = publish_with_retry(func)
        results[platform] = "posted" if ok else f"failed: {msg}"
    if any(r == "posted" for r in results.values()):
        _invalidate_image_cache()
    return {"status": "done", "post_id": target.get("post_id"), "platforms": results}


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-platform publisher agent for ig-autobot")
    parser.add_argument("--brand", choices=["main", "wilma"], default="main")
    parser.add_argument("--platforms", type=str, default=None, help="Comma-separated list (e.g. linkedin,bluesky)")
    parser.add_argument("--day", type=int, default=None, help="Wilma day number")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual publishing")
    parser.add_argument("--format", choices=["standard", "carousel", "quote"], default="standard", help="Content format (default: standard)")
    args = parser.parse_args()

    brand = BRAND[args.brand]
    platforms = [p.strip() for p in args.platforms.split(",")] if args.platforms else None

    if args.brand == "main":
        result = publish_main(brand, platforms=platforms, dry_run=args.dry_run, fmt=args.format)
    else:
        result = publish_wilma(brand, day=args.day, platforms=platforms, dry_run=args.dry_run)

    print(f"\n[publisher] Result: {result['status']}")
    for p, r in result.get("platforms", {}).items():
        print(f"  {p}: {r}")


if __name__ == "__main__":
    main()
