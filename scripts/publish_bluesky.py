#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from atproto import Client, models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_utils import (
    update_state_after_post,
    is_platform_posted,
    advance_stale_active_bundle,
    is_bundle_consumed_for_platform,
    load_state,
    save_state,
)

dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

flag_path = Path("bluesky_ready.flag")
state_path = Path("state.json")


def publish_to_bluesky():
    state = load_state(str(state_path))
    active = state.get("active_bundle") or {}

    if not flag_path.exists() or not active:
        print("⏭️ Nothing new to post for Bluesky. Skipping.")
        return

    if is_bundle_consumed_for_platform(active, "bluesky", state=state):
        advance_stale_active_bundle()
        state = load_state(str(state_path))
        active = state.get("active_bundle") or {}
        if not active:
            print("⏭️ No active_bundle after advance. Skipping.")
            return

    handle = os.environ.get("BLUESKY_HANDLE")
    password = os.environ.get("BLUESKY_PASSWORD")

    if not handle or not password:
        print("❌ BLUESKY_HANDLE or BLUESKY_PASSWORD not set")
        sys.exit(1)

    caption = ((active.get("captions") or {}).get("bluesky") or "")
    image_path = (active.get("image") or "output.jpg").replace("\\", "/")
    if not caption and Path("caption.txt").exists():
        caption = Path("caption.txt").read_text(encoding="utf-8").strip()
    if not Path(image_path).exists() and Path("output.jpg").exists():
        image_path = "output.jpg"
    if not caption:
        print("❌ No Bluesky caption available for active bundle.")
        sys.exit(1)
    if not Path(image_path).exists():
        print(f"❌ Image not found for active bundle: {image_path}")
        sys.exit(1)

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
            images=[models.AppBskyEmbedImages.Image(alt=caption[:100], image=upload.blob)]
        )

        print("Creating post...")
        client.send_post(text=caption, embed=embed)
        print("✅ Successfully posted to Bluesky!")
        update_state_after_post("bluesky")

        if flag_path.exists():
            os.remove(flag_path)
            print(f"✓ Flag {flag_path} consumed.")

    except Exception as e:
        import traceback
        print(f"❌ Failed to post to Bluesky: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    publish_to_bluesky()
