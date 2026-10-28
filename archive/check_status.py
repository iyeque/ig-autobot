#!/usr/bin/env python3
import json
import os

def summarize_state(state_path, label):
    if not os.path.exists(state_path):
        print(f"{label}: State file not found")
        return
    
    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    active = state.get('active_bundle')
    content_queue = state.get('content_queue', [])
    
    print(f"\n=== {label} ===")
    print(f"Total bundles in queue: {len(content_queue)}")
    
    if active and isinstance(active, dict):
        post_id = active.get("post_id", "N/A")
        print(f"Active bundle (currently publishing): Post ID {post_id}, Image: {active.get('image', 'N/A')}")
        print(f"Platforms already posted for active bundle: {active.get('platforms_posted', [])}")
    
    print(f"\nBundles in queue:")
    for i, bundle in enumerate(content_queue):
        if not isinstance(bundle, dict):
            print(f"  [{i+1}] Invalid bundle entry (expected dict): {bundle!r}")
            continue
        post_id = bundle.get("post_id", "N/A")
        captions = bundle.get("captions")
        if not isinstance(captions, dict):
            captions = {}
        snippet = str(captions.get("instagram", ""))[:50]
        print(f"  [{i+1}] Post ID: {post_id}, Caption snippet: {snippet}...")
    
    print(f"\n--- End of {label} ---")


if __name__ == "__main__":
    summarize_state('state.json', 'The Nine Stitches (Main)')
    summarize_state('forwilma/state.json', 'Digital Guardian (Wilma)')
