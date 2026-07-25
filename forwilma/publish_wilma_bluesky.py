#!/usr/bin/env python3
import os
import sys
import json
from atproto import Client, models
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to import shared_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_utils import update_state_after_post

# Load .env from project root if available
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from {dotenv_path}")

# Setup paths
FORWILMA_DIR = Path(__file__).parent
os.chdir(str(FORWILMA_DIR))

STATE_FILE = FORWILMA_DIR / "state.json"


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


def publish_wilma_to_bluesky():
    # Staleness Protection / queue advance
    flag_path = Path("wilma_bluesky_ready.flag")
    state_path = FORWILMA_DIR / "state.json"
    state = _read_state_path(state_path)
    active = state.get("active_bundle") or {}
    if not flag_path.exists() or not active:
        print("⏭️ Nothing new to post for Wilma's Bluesky. Skipping.")
        return
    if "bluesky" in (active.get("platforms_posted") or []):
        # Advance queue if this active bundle is fully posted
        queue = state.get("content_queue", [])
        if queue:
            state["active_bundle"] = queue.pop(0)
            state["active_bundle"]["platforms_posted"] = []
            state["active_bundle"]["platforms_prepared"] = []
            _write_state(state)
            print(f"▶ Advanced active bundle to {state['active_bundle'].get('post_id')}. Remaining: {len(queue)}")
        else:
            state["active_bundle"] = None
            _write_state(state)
            print("▶ Queue empty; cleared active bundle.")
        return

    # Wilma-specific credentials
    handle = os.environ.get("WILMA_BLUESKY_HANDLE")
    password = os.environ.get("WILMA_BLUESKY_PASSWORD")

    if not handle or not password:
        print("❌ WILMA_BLUESKY_HANDLE or WILMA_BLUESKY_PASSWORD not set")
        sys.exit(1)

    # 1. Read Caption
    caption_path = "caption.txt"
    if not os.path.exists(caption_path):
        print(f"❌ {caption_path} not found")
        sys.exit(1)

    with open(caption_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()

    # Last resort safety check (Bluesky 300 char limit)
    if len(caption) > 300:
        print(f"⚠ WARNING: Caption too long ({len(caption)}). Truncating.")
        caption = caption[:297] + "..."

    # 2. Read Image
    image_path = "output.jpg"
    if not os.path.exists(image_path):
        print(f"❌ {image_path} not found")
        sys.exit(1)

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
        update_state_after_post("bluesky", state_path="state.json")

        # Success: Consume flag
        if flag_path.exists():
            flag_path.unlink()
            print(f"✓ Flag {flag_path} consumed.")

    except Exception as e:
        print(f"❌ Failed to post to Bluesky: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    publish_wilma_to_bluesky()
