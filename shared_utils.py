
import os
import json
import tempfile
from typing import Any, Dict, List, Optional


MAIN_REQUIRED_PLATFORMS = [
    "instagram", "youtube", "threads", "bluesky", "facebook"
]
WILMA_REQUIRED_PLATFORMS = ["linkedin", "bluesky"]


def load_state(state_path: str = "state.json") -> Dict[str, Any]:
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_json(text: str) -> None:
    """Raise json.JSONDecodeError if the text is not valid JSON."""
    json.loads(text)


def save_state(state: Dict[str, Any], state_path: str = "state.json") -> None:
    """Write state as JSON without destroying a valid file on failure."""
    directory = os.path.dirname(os.path.abspath(state_path))
    os.makedirs(directory, exist_ok=True)
    candidate = json.dumps(state, indent=4, ensure_ascii=False)
    _validate_json(candidate)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(candidate)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, state_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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

    Published layout on GitHub Pages:
        images/output.jpg, images/story.jpg, reels/reel.mp4
    (see workflow "Create _site" step, which copies the root-level
    prepared files into these subfolders before deploy.)
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

    def _clean_path(path: str) -> str:
        path = path.replace("\\", "/")
        # Strip any leading "./" or "/" left over from os.path.join(".", ...)
        while path.startswith("./") or path.startswith("/"):
            path = path[2:] if path.startswith("./") else path[1:]
        return path

    def _to_url(path: str, subdir: str = "") -> str:
        if not path:
            return ""
        if path.startswith("http"):
            return path
        cleaned = _clean_path(path)
        if subdir and not cleaned.startswith(f"{subdir}/"):
            cleaned = f"{subdir}/{cleaned}"
        return base_url + cleaned

    return {
        "image": _to_url(image_path, subdir="images"),
        "reel": _to_url(reel_path, subdir="reels"),
        "story": _to_url(story_path, subdir="images"),
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
