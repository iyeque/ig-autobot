#!/usr/bin/env python3
import os
import sys
import json
from atproto import Client, models
from atproto_client.exceptions import UnauthorizedError
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to import shared_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_utils import update_state_after_post, advance_stale_active_bundle, load_state, save_state

# Load .env from project root if available
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from {dotenv_path}")

# Setup paths
FORWILMA_DIR = Path(__file__).parent
os.chdir(str(FORWILMA_DIR))

STATE_FILE = FORWILMA_DIR / "state.json"
SCHEDULE_FILE = FORWILMA_DIR / "schedule.json"

from datetime import datetime

WEEKDAY_EXPECTED_TYPE = {
    0: "TOFU",
    1: "TOFU",
    2: "TOFU",
    3: "TOFU",
    4: "TOFU",
    5: "TOFU",
    6: "TOFU",
}


def _today_expected_type():
    return WEEKDAY_EXPECTED_TYPE.get(datetime.utcnow().weekday())


def _load_schedule_type(day_num: int) -> str:
    """Look up a day's type from schedule.json, falling back to TOFU."""
    if not SCHEDULE_FILE.exists():
        return "TOFU"
    try:
        schedule = json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
        for entry in schedule:
            if isinstance(entry, dict) and entry.get("day") == day_num:
                return (entry.get("type") or "TOFU").upper()
    except Exception:
        pass
    return "TOFU"


def _active_type(active: dict) -> str:
    """Resolve the effective type for an active bundle, using schedule as fallback."""
    if isinstance(active, dict):
        t = (active.get("type") or "").strip().upper()
        if t:
            return t
        post_id = active.get("post_id", "")
        if post_id.startswith("day_"):
            try:
                day_num = int(post_id.split("_")[1])
                return _load_schedule_type(day_num)
            except (ValueError, IndexError):
                pass
    return ""


def _advance_to_today_pillar(state_path):
    state = load_state(state_path)
    expected = _today_expected_type()
    active = state.get("active_bundle")
    if isinstance(active, dict):
        active_type = _active_type(active)
        if active_type != expected:
            state["active_bundle"] = None
            save_state(state, state_path)
            print(f"▶ Cleared stale active bundle (type={active_type}, expected={expected})")
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


def _normalize_bluesky_handle(handle: str) -> str:
    handle = (handle or "").strip().lstrip("@")
    if not handle:
        return ""
    if "." not in handle:
        handle = f"{handle}.bsky.social"
    return handle


def _bluesky_credential_candidates():
    """Return unique (handle, password, label) tuples — Wilma-only, no fallback."""
    candidates = []
    seen = set()
    for label, handle_key, password_key in (
        ("WILMA_BLUESKY", "WILMA_BLUESKY_HANDLE", "WILMA_BLUESKY_PASSWORD"),
    ):
        handle = _normalize_bluesky_handle(os.environ.get(handle_key, ""))
        password = (os.environ.get(password_key) or "").strip()
        if not handle or not password:
            continue
        key = (handle, password)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((handle, password, label))
    return candidates


def _login_bluesky_client():
    candidates = _bluesky_credential_candidates()
    if not candidates:
        print("❌ WILMA_BLUESKY_HANDLE (or BLUESKY_HANDLE) / WILMA_BLUESKY_PASSWORD not set")
        sys.exit(1)

    last_error = None
    for handle, password, label in candidates:
        client = Client()
        try:
            print(f"Logging into Bluesky as {handle} ({label})...")
            client.login(handle, password)
            return client
        except UnauthorizedError as exc:
            last_error = exc
            if len(candidates) > 1:
                print(f"⚠ {label} credentials rejected; trying fallback...")
            continue
        except Exception as exc:
            last_error = exc
            print(f"❌ Bluesky login failed for {label}: {exc}")
            sys.exit(1)

    print("❌ Bluesky authentication failed for all configured credential sets.")
    print("   Use a Bluesky App Password (Settings → App Passwords), not your account password.")
    print("   Update GitHub secrets WILMA_BLUESKY_HANDLE and WILMA_BLUESKY_PASSWORD.")
    if last_error:
        print(f"   Last error: {last_error}")
    sys.exit(1)


def _post_bluesky_text_only(caption, flag_path, state_path=str(STATE_FILE)):
    """Post a caption-only (no image) Bluesky entry for Wilma."""
    if len(caption) > 300:
        print(f"⚠ WARNING: Caption too long ({len(caption)}). Truncating.")
        caption = caption[:297] + "..."

    client = _login_bluesky_client()
    try:
        print("Creating caption-only post...")
        client.send_post(text=caption)
        print("✅ Successfully posted caption-only to Wilma's Bluesky!")
        update_state_after_post("bluesky", state_path=state_path)

        if flag_path.exists():
            flag_path.unlink()
            print(f"✓ Flag {flag_path} consumed.")
    except Exception as e:
        print(f"❌ Failed to post caption-only to Bluesky: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _post_bluesky(caption, image_path, flag_path):
    if len(caption) > 300:
        print(f"⚠ WARNING: Caption too long ({len(caption)}). Truncating.")
        caption = caption[:297] + "..."

    client = _login_bluesky_client()
    try:
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
    active = _resolve_active_bundle(state) or {}
    active = active if isinstance(active, dict) else {}
    if not flag_path.exists() and "bluesky" not in (active.get("platforms_prepared") or []):
        print("⏭️ Nothing new to post for Wilma's Bluesky. Skipping.")
        return
    if not flag_path.exists():
        print("▶ No ready flag on disk, but active bundle was prepared for Bluesky — proceeding.")

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

    # Caption-only mode: if no image is available, post text-only.
    image_exists = Path(image_path).exists() if image_path else False
    if not image_exists:
        print(f"⚠ No image available for active bundle — posting caption-only to Bluesky.")
        _post_bluesky_text_only(caption, flag_path)
        return

    _post_bluesky(caption, image_path, flag_path)


if __name__ == "__main__":
    publish_wilma_to_bluesky()
