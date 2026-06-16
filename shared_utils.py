
import os
import json
from typing import Any, Dict, List, Optional


MAIN_REQUIRED_PLATFORMS = [
    "instagram", "linkedin", "youtube", "threads", "bluesky", "facebook"
]
WILMA_REQUIRED_PLATFORMS = ["linkedin", "bluesky"]


def load_state(state_path: str = "state.json") -> Dict[str, Any]:
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any], state_path: str = "state.json") -> None:
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)


def get_active_bundle(state_path: str = "state.json") -> Optional[Dict[str, Any]]:
    state = load_state(state_path)
    active = state.get("active_bundle")
    return active if isinstance(active, dict) else None


def is_platform_posted(platform: str, state_path: str = "state.json") -> bool:
    active = get_active_bundle(state_path)
    if not active:
        return False
    return platform in active.get("platforms_posted", [])


def required_platforms(state_path: str = "state.json") -> List[str]:
    if "forwilma" in state_path.replace("\\", "/"):
        return list(WILMA_REQUIRED_PLATFORMS)
    return list(MAIN_REQUIRED_PLATFORMS)


def resolve_bundle_media(
    active: Dict[str, Any],
    base_url: str = "https://iyeque.github.io/ig-autobot/",
    state_dir: str = ".",
) -> Dict[str, str]:
    """
    Resolve public URLs for the active bundle's media.
    Prefers prepared local copies (output.jpg / reel.mp4), then bundle paths.
    """
    image_local = os.path.join(state_dir, "output.jpg")
    reel_local = os.path.join(state_dir, "reel.mp4")
    story_local = os.path.join(state_dir, "story.jpg")

    image_path = active.get("image", "")
    reel_path = active.get("reel", "")
    story_path = active.get("story", "")

    if os.path.exists(image_local):
        image_path = image_local.replace("\\", "/")
    if os.path.exists(reel_local):
        reel_path = reel_local.replace("\\", "/")
    if os.path.exists(story_local):
        story_path = story_local.replace("\\", "/")

    def _to_url(path: str) -> str:
        if not path:
            return ""
        if path.startswith("http"):
            return path
        return base_url + path.lstrip("/").replace("\\", "/")

    return {
        "image": _to_url(image_path),
        "reel": _to_url(reel_path),
        "story": _to_url(story_path),
        "image_local": image_local if os.path.exists(image_local) else image_path,
        "reel_local": reel_local if os.path.exists(reel_local) else reel_path,
    }


def update_state_after_post(platform, state_path="state.json"):
    """Update state.json to mark the platform as posted in the active bundle."""
    if not os.path.exists(state_path):
        print(f"{state_path} not found, skipping state update.")
        return

    try:
        state = load_state(state_path)
        active = state.get("active_bundle")
        if not active or not isinstance(active, dict):
            print("No active_bundle in state, skipping state update.")
            return

        if "platforms_posted" not in active:
            active["platforms_posted"] = []

        if platform not in active["platforms_posted"]:
            active["platforms_posted"].append(platform)
            print(f"Marked {platform} as posted in active bundle.")

        posted = active.get("platforms_posted", [])
        required = required_platforms(state_path)
        if all(p in posted for p in required):
            print("All platforms posted, clearing active_bundle.")
            state["active_bundle"] = None

        save_state(state, state_path)

    except Exception as e:
        print(f"Failed to update state: {e}")
