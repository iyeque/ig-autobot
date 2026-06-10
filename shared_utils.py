
import os
import json


def update_state_after_post(platform, state_path="state.json"):
    """Update state.json to mark the platform as posted in the active bundle."""
    if not os.path.exists(state_path):
        print(f"{state_path} not found, skipping state update.")
        return
        
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        active = state.get("active_bundle")
        if not active:
            print("No active_bundle in state, skipping state update.")
            return
            
        if "platforms_posted" not in active:
            active["platforms_posted"] = []
            
        if platform not in active["platforms_posted"]:
            active["platforms_posted"].append(platform)
            print(f"Marked {platform} as posted in active bundle.")
            
        # Check if all required platforms are posted, and if so, clear active_bundle
        is_wilma = "forwilma" in state_path or "wilma" in platform
        if is_wilma:
            required = ["linkedin", "bluesky"]
        else:
            required = ["instagram", "linkedin", "pinterest", "youtube", "threads", "bluesky", "facebook"]
            
        posted = active.get("platforms_posted", [])
        if all(p in posted for p in required):
            print("All platforms posted, clearing active_bundle.")
            state["active_bundle"] = None
            
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
            
    except Exception as e:
        print(f"Failed to update state: {e}")

