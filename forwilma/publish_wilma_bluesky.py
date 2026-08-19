#!/usr/bin/env python3
import os
import sys
import json
from atproto import Client, models
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to import shared_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_utils import update_state_after_post, advance_stale_active_bundle

# Load .env from project root if available
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from {dotenv_path}")

# Setup paths
FORWILMA_DIR = Path(__file__).parent
os.chdir(str(FORWILMA_DIR))

STATE_FILE = FORWILMA_DIR / "state.json"

from datetime import datetime

WEEKDAY_EXPECTED_TYPE = {
    0: "TOFU",
    1: "TOFU",
    2: "BOFU",
    3: "TOFU",
    4: "MOFU",
    5: "Experiment",
    6: "MOFU",
}


def _today_expected_type():
    return WEEKDAY_EXPECTED_TYPE.get(datetime.utcnow().weekday())


def _advance_to_today_pillar(state_path):
    state = _read_state_path(state_path)
    expected = _today_expected_type()
    advanced = False
    for _ in range(20):
        active = state.get("active_bundle")
        if not isinstance(active, dict):
            break
        active_type = (active.get("type") or "").strip().upper()
        if active_type == expected:
            break
        queue = state.get("content_queue", [])
        if not queue:
            state["active_bundle"] = None
            advanced = True
            break
        state["active_bundle"] = queue.pop(0)
        state["active_bundle"]["platforms_posted"] = []
        state["active_bundle"]["platforms_prepared"] = state["active_bundle"].get("platforms_prepared", [])
        print(f"▶ Advanced non-{expected} bundle to {state['active_bundle'].get('post_id')}. Remaining: {len(queue)}")
        advanced = True
    if advanced:
        _write_state(state)
    return state.get("active_bundle") if isinstance(state.get("active_bundle"), dict) else None


def _read_state_path(state_path: Path):
    if not state_path.exists():
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_state(state: dict) -> None:
    tmp_path = STATE_FILE.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, STATE_FILE)


def _resolve_active_bundle(state):
    active = state.get("active_bundle")
    if isinstance(active, (int, str)):
        for b in state.get("content_queue", []):
            if isinstance(b, dict) and b.get("post_id") == active:
                return b
            elif b == active:
                return {"post_id": b}
    return active


def _resolve_wilma_media(active: dict):
    """Return (caption, image_path) preferring state, falling back to legacy files."""
    if not isinstance(active, dict):
        return "", "output.jpg"
    captions = active.get("captions") or {}
    caption = captions.get("bluesky") or ""
    raw_image = active.get("image") or "output.jpg"
    image_path = raw_image.replace("\\", "/") if isinstance(raw_image, str) else str(raw_image)

    if not caption and Path("caption.txt").exists():
        caption = Path("caption.txt").read_text(encoding="utf-8").strip()
    if not Path(image_path).exists() and Path("output.jpg").exists():
        image_path = "output.jpg"

    return caption, image_path


def _post_bluesky(handle, password, caption, image_path, flag_path):
    if len(caption) > 300:
        print(f"⚠ WARNING: Caption too long ({len(caption)}). Truncating.")
        caption = caption[:297] + "..."

    print(f"Logging into Bluesky as {handle}...")
    client = Client()
    try:
        client.login(handle, password)

        print(f"Uploading image {image_path}...")
        with open(image_path, 'rb') as f:
            img_data = f.read()

        upload = client.upload_blob(img_data)
        embed = models.AppBskyEmbedImages.Main(
            images=[models.AppBskyEmbedImages.Image(alt="Digital Guardian - Wilma", image=upload.blob)]
        )

        print("Creating post...")
        client.send_post(text=caption, embed=embed)
        print("✅ Successfully posted to Wilma's Bluesky!")
        update_state_after_post("bluesky", state_path=str(STATE_FILE))

        if flag_path.exists():
            flag_path.unlink()
            print(f"✓ Flag {flag_path} consumed.")

    except Exception as e:
        print(f"❌ Failed to post to Bluesky: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def publish_wilma_to_bluesky():
    flag_path = Path("wilma_bluesky_ready.flag")
    state_path = FORWILMA_DIR / "state.json"
    state = _read_state_path(state_path)
    active = _advance_to_today_pillar(state_path) or _resolve_active_bundle(state) or {}

    if not flag_path.exists():
        print("⏭️ Nothing new to post for Wilma's Bluesky. Skipping.")
        return

    if not isinstance(active, dict) or not active.get("post_id"):
        queue = state.get("content_queue", [])
        if queue:
            state["active_bundle"] = queue.pop(0)
            state["content_queue"] = queue
            if "platforms_posted" not in state["active_bundle"]:
                state["active_bundle"]["platforms_posted"] = []
            if "platforms_prepared" not in state["active_bundle"]:
                state["active_bundle"]["platforms_prepared"] = []
            _write_state(state)
            print(f"▶ Advanced active bundle to {state['active_bundle'].get('post_id')}. Remaining: {len(queue)}")
            active = state["active_bundle"]
        else:
            print("⏭️ Nothing new to post for Wilma's Bluesky. Skipping.")
            return

    # If this platform already posted the active bundle, advance once and retry
    if "bluesky" in (active.get("platforms_posted") or []):
        advance_stale_active_bundle(state_path=str(STATE_FILE))
        state = _read_state_path(state_path)
        active = _advance_to_today_pillar(state_path) or _resolve_active_bundle(state) or {}
        if not isinstance(active, dict) or not active.get("post_id"):
            print("⏭️ No active_bundle after advance. Skipping.")
            return

    caption, image_path = _resolve_wilma_media(active)
    if not caption:
        print("❌ No Bluesky caption available for active bundle.")
        sys.exit(1)
    if not Path(image_path).exists():
        print(f"❌ Image not found for active bundle: {image_path}")
        sys.exit(1)

    handle = os.environ.get("WILMA_BLUESKY_HANDLE") or os.environ.get("BLUESKY_HANDLE")
    password = os.environ.get("WILMA_BLUESKY_PASSWORD") or os.environ.get("BLUESKY_PASSWORD")

    if not handle or not password:
        print("❌ WILMA_BLUESKY_HANDLE (or BLUESKY_HANDLE) / WILMA_BLUESKY_PASSWORD not set")
        sys.exit(1)

    if handle and "." not in handle:
        handle = f"{handle}.bsky.social"

    _post_bluesky(handle, password, caption, image_path, flag_path)


if __name__ == "__main__":
    publish_wilma_to_bluesky()
