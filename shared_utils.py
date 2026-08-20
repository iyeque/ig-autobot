
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
    if platform in active.get("platforms_posted", []):
        return True
    post_id = active.get("post_id")
    state = load_state(state_path)
    return bool(post_id and post_id in state.get("platform_posted_bundles", {}).get(platform, []))


def is_bundle_consumed_for_platform(
    bundle: Optional[Dict[str, Any]],
    platform: str,
    state_path: str = "state.json",
    state: Optional[Dict[str, Any]] = None,
) -> bool:
    if not isinstance(bundle, dict):
        return False

    platform_in_bundle = platform in (bundle.get("platforms_posted") or [])
    if platform_in_bundle:
        return True

    post_id = bundle.get("post_id")
    if not post_id:
        return False

    state_data = state if isinstance(state, dict) else load_state(state_path)
    platform_history = state_data.get("platform_posted_bundles", {})
    if not isinstance(platform_history, dict):
        return False

    if post_id in platform_history.get(platform, []):
        return True

    return False


def advance_stale_active_bundle(state_path: str = "state.json") -> bool:
    """
    If the current active_bundle has already been posted to every required
    platform according to platform_posted_bundles history, advance the queue
    so publishers don't keep skipping forever. Repeats until a non-stale
    active bundle is found or the queue empties.
    Returns True if advanced or cleared, False otherwise.
    """
    advanced_once = False
    state = load_state(state_path)
    required = required_platforms(state_path)

    while True:
        active = state.get("active_bundle")

        # Handle int/str active_bundle by resolving it from the queue
        if isinstance(active, (int, str)):
            found = None
            for b in state.get("content_queue", []):
                if isinstance(b, dict):
                    if b.get("post_id") == active:
                        found = b
                        break
                elif b == active:
                    found = {"post_id": b}
                    break
            if found:
                state["active_bundle"] = found
                active = found
                save_state(state, state_path)
            else:
                break

        if not isinstance(active, dict):
            break
        post_id = active.get("post_id")
        if not post_id:
            break

        history = state.get("platform_posted_bundles", {})
        posted_for_active = active.get("platforms_posted", [])
        is_stale = all(
            (p in posted_for_active) or (post_id in history.get(p, []))
            for p in required
        )
        if not is_stale:
            break

        queue = state.get("content_queue", [])
        if queue:
            state["active_bundle"] = queue.pop(0)
            state["active_bundle"]["platforms_posted"] = []
            state["active_bundle"]["platforms_prepared"] = []
            print(f"▶ Advanced stale active bundle to {state['active_bundle'].get('post_id')}. Remaining: {len(queue)}")
            advanced_once = True
            # Continue loop in case next queued bundle is also stale
            continue
        else:
            state["active_bundle"] = None
            print("▶ Queue empty; cleared stale active bundle.")
            advanced_once = True
            break

    if advanced_once:
        save_state(state, state_path)
    return advanced_once


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

        if "platform_posted_bundles" not in state:
            state["platform_posted_bundles"] = {}

        post_id = active.get("post_id")
        if post_id:
            state["platform_posted_bundles"].setdefault(platform, [])
            if post_id not in state["platform_posted_bundles"][platform]:
                state["platform_posted_bundles"][platform].append(post_id)
                print(f"Recorded {platform} completion for bundle {post_id}.")

        if "platforms_posted" not in active:
            active["platforms_posted"] = []
        if platform not in active["platforms_posted"]:
            active["platforms_posted"].append(platform)

        posted = active.get("platforms_posted", [])
        required = required_platforms(state_path)
        if all(p in posted for p in required):
            print("All platforms posted, advancing queue.")
            state["active_bundle"] = None
            queue = state.get("content_queue", [])
            if queue:
                state["active_bundle"] = queue.pop(0)
                state["content_queue"] = queue
                print(f"Advanced active bundle to {state['active_bundle'].get('post_id')}. Remaining: {len(queue)}")
            else:
                state["content_queue"] = []

        save_state(state, state_path)

    except Exception as e:
        print(f"Failed to update state: {e}")


def clean_caption_formatting(text: str) -> str:
    if not text:
        return text or ""
    text = text.replace("**", "").replace("*", "").replace("__", "").replace("_", "")
    text = text.replace("—", "-").replace("–", "-").replace("'", "'").replace("'", "'")
    return text.strip()
