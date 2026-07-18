#!/usr/bin/env python3
import os
import json
import sys
import argparse
import shutil
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared_utils import load_state, save_state, is_platform_posted, required_platforms, clean_caption_formatting


def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--bundle", help="Legacy support (ignored in queue mode)")
    parser.add_argument("--state_path", default="state.json", help="Path to state.json")
    args = parser.parse_args()
    platform = args.platform.lower()
    state_path = args.state_path
    state_dir = os.path.dirname(state_path) or "."

    if not os.path.exists(state_path):
        print(f"Error: {state_path} not found.")
        sys.exit(1)

    # Normalize any Windows-style backslashes in paths to forward slashes.
    # This prevents stale GitHub Actions states from failing on Linux runners.
    _fixed_paths = False
    def _normalize_paths(obj):
        global _fixed_paths
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, str) and chr(92) in v:
                    obj[k] = v.replace(chr(92), "/")
                    _fixed_paths = True
                else:
                    _normalize_paths(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str) and chr(92) in v:
                    obj[i] = v.replace(chr(92), "/")
                    _fixed_paths = True
                else:
                    _normalize_paths(v)
    _normalize_paths(state := load_state(state_path))
    if _fixed_paths:
        save_state(state, state_path)

    # Ensure we are operating on the freshest remote state to avoid stale skips.
    if os.environ.get("GITHUB_ACTIONS") == "true":
        try:
            result = subprocess.run(
                ["git", "pull", "--rebase", "origin", "master"],
                check=False,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"Git pull rebase failed with code {result.returncode}, trying merge fallback...")
                print(result.stderr)
                # Fallback: try a non-rebase pull
                result2 = subprocess.run(
                    ["git", "pull", "origin", "master"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                print(result2.stdout)
                print(result2.stderr)
                if result2.returncode != 0:
                    print(f"Git pull fallback also failed with code {result2.returncode}. Proceeding with local state.")
        except Exception as pull_exc:
            print(f"Git pull exception: {pull_exc}")

    state = load_state(state_path)

    # --- Self-healing: clear stale active bundle when all required platforms are already posted ---
    active = state.get("active_bundle")
    if active and isinstance(active, dict):
        required = required_platforms(state_path)
        posted = active.get("platforms_posted", [])
        if all(p in posted for p in required):
            print(f"Active bundle {active.get('post_id')} already fully posted. Clearing.")
            state["active_bundle"] = None
            active = None
            save_state(state, state_path)

    # --- Queue Management ---
    if not active:
        queue = state.get("content_queue", [])
        if not queue:
            print(f"Content queue in {state_path} is empty. Nothing to prepare.")
            sys.exit(0)

        # Pop until we find a bundle that still needs this platform
        required = required_platforms(state_path)
        seen_ids = set()
        while queue:
            candidate = queue.pop(0)
            cid = candidate.get("post_id")
            if cid in seen_ids:
                print(f"Skipping duplicate {cid}. Remaining: {len(queue)}")
                continue
            already_posted = all(p in candidate.get("platforms_posted", []) for p in required)
            if already_posted:
                print(f"Skipping {candidate.get('post_id')}: already posted. Remaining queue: {len(queue)}")
                continue
            seen_ids.add(cid)
            active = candidate
            break

        if not active:
            print(f"Content queue in {state_path} has only already-posted bundles. Clearing.")
            state["content_queue"] = []
            save_state(state, state_path)
            sys.exit(0)

        state["active_bundle"] = active
        state["content_queue"] = queue
        print(f"Pulled bundle {active.get('post_id')} from queue. Remaining: {len(queue)}")

    if not isinstance(active, dict):
        print("Error: active_bundle is not a valid dict.")
        sys.exit(1)

    # Never re-prepare a platform that already posted this bundle.
    if is_platform_posted(platform, state_path):
        print(f"{platform.upper()} already POSTED for active bundle {active.get('post_id')}. Skipping.")
        save_state(state, state_path)
        sys.exit(0)

    print(f"[{state_path}] Preparing assets from bundle: {active.get('post_id')}")

    # --- Prepare Media Files ---
    media_map = {
        "image": "output.jpg",
        "reel": "reel.mp4",
        "story": "story.jpg",
    }

    for key, local_name in media_map.items():
        src = active.get(key)
        if not src:
            continue

        target_path = os.path.join(state_dir, local_name)
        copied = False
        candidates = [src, os.path.join(state_dir, src), os.path.join(state_dir, os.path.basename(src))]
        for cand in candidates:
            if not copied and os.path.exists(cand):
                shutil.copy(cand, target_path)
                print(f"Copied {cand} -> {target_path}")
                copied = True
                break
        if not copied:
            print(f"❌ Critical: Required media '{key}' ({src}) not found for bundle {active.get('post_id') or state.get('active_bundle', {}).get('post_id')}.")
            sys.exit(1)

    # --- Prepare Carousel (if present) ---
    carousel_paths = active.get("carousel") or []

    # Hard carousel-day guard: carousels may only be prepared/posted on UTC Wednesdays
    is_carousel_day = datetime.utcnow().weekday() == 2
    if carousel_paths and not is_carousel_day:
        print(f"⏭️ Carousel skipped: bundle {active.get('post_id')} has carousel slides, but today is not UTC Wednesday. Falling back to single image.")
        carousel_paths = []

    if carousel_paths:
        carousel_dir = os.path.join(state_dir, "carousel")
        os.makedirs(carousel_dir, exist_ok=True)
        prepared_paths = []
        for idx, src in enumerate(carousel_paths, start=1):
            target_path = os.path.join(carousel_dir, f"slide_{idx}.jpg")
            if os.path.exists(src):
                shutil.copy(src, target_path)
                prepared_paths.append(target_path)
            else:
                alt_src = os.path.join(state_dir, os.path.basename(src))
                if os.path.exists(alt_src):
                    shutil.copy(alt_src, target_path)
                    prepared_paths.append(target_path)
                else:
                    print(f"Warning: Carousel slide {idx} ({src}) not found.")
        if prepared_paths:
            carousel_json = os.path.join(state_dir, "carousel.json")
            url_paths = ["carousel/" + os.path.basename(p) for p in prepared_paths]
            with open(carousel_json, "w", encoding="utf-8") as f:
                json.dump(url_paths, f, indent=2)
            print(f"Prepared carousel.json with {len(url_paths)} slides")

    # --- Prepare Caption ---
    captions = active.get("captions", {})
    if platform not in captions:
        print(f"Error: Caption for platform '{platform}' not found in active bundle.")
        sys.exit(1)

    raw_caption = clean_caption_formatting(captions.get(platform) or "")
    if not raw_caption.strip():
        print(f"⚠ Caption for {platform.upper()} is empty — skipping ready flag to prevent blank post.")
        sys.exit(1)
    if "[Caption generation failed" in raw_caption or "Traceback" in raw_caption:
        print(f"⚠ Caption for {platform.upper()} looks like an error payload — aborting prepare.")
        sys.exit(1)

    caption_path = os.path.join(state_dir, "caption.txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(raw_caption)
    print(f"Prepared caption.txt for {platform.upper()} at {caption_path}")

    # --- Create Ready Flag ---
    is_wilma = "forwilma" in state_path or "wilma" in platform
    flag_prefix = "wilma_" if is_wilma else ""
    flag_name = f"{flag_prefix}{platform}_ready.flag"
    flag_path = os.path.join(state_dir, flag_name)

    with open(flag_path, "w", encoding="utf-8") as f:
        f.write(str(active.get("timestamp", "")))
    print(f"Created {flag_path}")

    # Track prep attempts (informational only; posting guard uses platforms_posted).
    if "platforms_prepared" not in state["active_bundle"]:
        state["active_bundle"]["platforms_prepared"] = []
    if platform not in state["active_bundle"]["platforms_prepared"]:
        state["active_bundle"]["platforms_prepared"].append(platform)

    save_state(state, state_path)
    print(f"Assets ready for {platform.upper()}.")


if __name__ == "__main__":
    prepare()
