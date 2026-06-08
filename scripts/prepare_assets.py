#!/usr/bin/env python3
import os
import json
import sys
import argparse
import shutil

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--bundle", help="Legacy support (ignored in queue mode)")
    parser.add_argument("--state_path", default="state.json", help="Path to state.json")
    args = parser.parse_args()
    platform = args.platform.lower()
    state_path = args.state_path

    # --- 1. Load State ---
    if not os.path.exists(state_path):
        print(f"❌ Error: {state_path} not found.")
        sys.exit(1)
        
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    # --- 2. Queue Management ---
    active = state.get("active_bundle")
    
    if not active:
        queue = state.get("content_queue", [])
        if not queue:
            print(f"⏭️ Content queue in {state_path} is empty. Nothing to prepare.")
            sys.exit(0)
            
        active = queue.pop(0)
        state["active_bundle"] = active
    
    # --- IDEMPOTENCY CHECK ---
    # If this platform has already been prepared for THIS active bundle, skip.
    prepared_list = active.get("platforms_prepared", [])
    if platform in prepared_list:
        # Check if the flag also exists. If not, maybe it was consumed but we're re-running?
        # To be safe, if it's in prepared_list, we assume it's already DONE.
        print(f"⏭️ {platform.upper()} already marked as PREPARED in active bundle. Skipping to prevent duplicates.")
        sys.exit(0)

    print(f"📦 [{state_path}] Preparing assets from bundle: {active.get('post_id')}")

    # --- 3. Prepare Media Files ---
    media_map = {
        "image": "output.jpg",
        "reel": "reel.mp4",
        "story": "story.jpg"
    }
    
    # If using Wilma's state, we might need to adjust paths if they were saved relative to FORWILMA_DIR
    for key, local_name in media_map.items():
        src = active.get(key)
        if not src: continue
        
        # Absolute path check or relative to current working dir (root)
        if os.path.exists(src):
            shutil.copy(src, local_name)
            print(f"✓ Copied {src} -> {local_name}")
        else:
            # Try relative to the state file's directory
            alt_src = os.path.join(os.path.dirname(state_path), os.path.basename(src))
            if os.path.exists(alt_src):
                shutil.copy(alt_src, local_name)
                print(f"✓ Copied {alt_src} -> {local_name} (alt path)")
            else:
                print(f"⚠ Warning: Media {key} ({src}) not found.")

    # --- 4. Prepare Caption ---
    captions = active.get("captions", {})
    if platform not in captions:
        print(f"❌ Error: Caption for platform '{platform}' not found in active bundle.")
        sys.exit(1)
        
    with open("caption.txt", "w", encoding="utf-8") as f:
        f.write(captions[platform])
    print(f"✓ Prepared caption.txt for {platform.upper()}")

    # --- 5. Create Ready Flag ---
    # For Wilma, flags are named wilma_[platform]_ready.flag
    is_wilma = "forwilma" in state_path or "wilma" in platform
    flag_prefix = "wilma_" if is_wilma else ""
    flag_name = f"{flag_prefix}{platform}_ready.flag"
    
    if os.path.exists(flag_name):
        print(f"ℹ️ Flag {flag_name} already exists. Skipping prep.")
    else:
        with open(flag_name, "w") as f:
            f.write(active.get("timestamp", ""))
        print(f"🚩 Created {flag_name}")

    # --- 6. Finalize State ---
    if "platforms_prepared" not in state["active_bundle"]:
        state["active_bundle"]["platforms_prepared"] = []
    
    if platform not in state["active_bundle"]["platforms_prepared"]:
        state["active_bundle"]["platforms_prepared"].append(platform)

    # Required platforms detection
    if is_wilma:
        required = ["linkedin", "bluesky"]
    else:
        required = ["instagram", "linkedin", "pinterest", "youtube", "threads", "bluesky"]
        
    prepared = state["active_bundle"].get("platforms_prepared", [])
    if all(p in prepared for p in required):
        print(f"🎊 Bundle fully consumed for {'Wilma' if is_wilma else 'Trilogy'}. Clearing active_bundle.")
        state["active_bundle"] = None
    
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

    print(f"✅ Assets ready for {platform.upper()}.")

if __name__ == "__main__":
    prepare()
