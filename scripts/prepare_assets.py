#!/usr/bin/env python3
import os
import json
import sys
import argparse
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared_utils import load_state, save_state, is_platform_posted, required_platforms


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

        active = queue.pop(0)
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

        if os.path.exists(src):
            shutil.copy(src, target_path)
            print(f"Copied {src} -> {target_path}")
        else:
            alt_src = os.path.join(state_dir, os.path.basename(src))
            if os.path.exists(alt_src):
                shutil.copy(alt_src, target_path)
                print(f"Copied {alt_src} -> {target_path} (alt path)")
            else:
                print(f"Warning: Media {key} ({src}) not found.")

    # --- Prepare Caption ---
    captions = active.get("captions", {})
    if platform not in captions:
        print(f"Error: Caption for platform '{platform}' not found in active bundle.")
        sys.exit(1)

    caption_path = os.path.join(state_dir, "caption.txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(captions[platform])
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
