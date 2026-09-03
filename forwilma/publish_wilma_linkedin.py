#!/usr/bin/env python3
import os
import sys
import json
import requests
import time
from pathlib import Path
from dotenv import load_dotenv
import hashlib

# Add project root to path to import shared_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_utils import (
    update_state_after_post,
    advance_stale_active_bundle,
    is_bundle_consumed_for_platform,
    load_state,
    save_state,
    required_platforms,
)
from datetime import datetime

WEEKDAY_EXPECTED_TYPE = {
    0: None,   # Monday
    1: None,   # Tuesday
    2: None,   # Wednesday
    3: None,   # Thursday
    4: None,   # Friday
    5: None,   # Saturday
    6: None,   # Sunday
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
    if expected is None:
        return state.get("active_bundle") if isinstance(state.get("active_bundle"), dict) else None
    active = state.get("active_bundle")
    if isinstance(active, dict):
        active_type = _active_type(active)
        if active_type != expected:
            state["active_bundle"] = None
            save_state(state, state_path)
            print(f"▶ Cleared stale active bundle (type={active_type}, expected={expected})")
    return state.get("active_bundle") if isinstance(state.get("active_bundle"), dict) else None


FORWILMA_DIR = Path(__file__).parent
SCHEDULE_FILE = FORWILMA_DIR / "schedule.json"
if (FORWILMA_DIR.parent / '.env').exists():
    load_dotenv(dotenv_path=FORWILMA_DIR.parent / '.env')
os.chdir(str(FORWILMA_DIR))
STATE_FILE = FORWILMA_DIR / "state.json"
state_path = STATE_FILE


def _linkedin_flag_path():
    """Resolve ready flag at publish time (prepare_assets may have just created it)."""
    for name in ("wilma_linkedin_ready.flag", "linkedin_ready.flag"):
        path = FORWILMA_DIR / name
        if path.exists():
            return path
    return None


def _linkedin_prepared_in_state(state: dict) -> bool:
    active = state.get("active_bundle")
    if not isinstance(active, dict):
        return False
    return "linkedin" in (active.get("platforms_prepared") or [])

LINKEDIN_REFRESH_TOKEN = os.environ.get('WILMA_LINKEDIN_REFRESH_TOKEN')
LINKEDIN_CLIENT_ID = os.environ.get('WILMA_LINKEDIN_CLIENT_ID')
LINKEDIN_CLIENT_SECRET = os.environ.get('WILMA_LINKEDIN_CLIENT_SECRET')
LINKEDIN_URN = os.environ.get('WILMA_LINKEDIN_URN')
LINKEDIN_VERSION = '202604'


def get_fresh_linkedin_token():
    """Exchanges a refresh token for a new access token for Wilma's LinkedIn."""
    if not LINKEDIN_REFRESH_TOKEN or not LINKEDIN_CLIENT_ID or not LINKEDIN_CLIENT_SECRET:
        print('❌ Wilma LinkedIn refresh credentials missing. Cannot publish.')
        return None

    print("Refreshing Wilma's LinkedIn Access Token...")
    url = 'https://www.linkedin.com/oauth/v2/accessToken'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': LINKEDIN_REFRESH_TOKEN,
        'client_id': LINKEDIN_CLIENT_ID,
        'client_secret': LINKEDIN_CLIENT_SECRET
    }

    try:
        r = requests.post(url, headers=headers, data=data)
        if r.status_code == 200:
            new_token = r.json().get('access_token')
            print("✅ Successfully refreshed Wilma's LinkedIn Access Token.")
            return new_token
        else:
            print(f'❌ Failed to refresh Wilma token: {r.status_code} {r.text}')
            return None
    except Exception as e:
        print(f'❌ Error during Wilma token refresh: {e}')
        return None


def upload_image_rest(image_path, author_urn, access_token, max_retries=3):
    """Modern LinkedIn image upload flow using /rest/images (v202604+)"""
    headers = {
        'Authorization': f'Bearer {access_token}',
        'LinkedIn-Version': LINKEDIN_VERSION,
        'X-Restli-Protocol-Version': '2.0.0'
    }

    for attempt in range(max_retries):
        try:
            print(f'Initializing LinkedIn image upload (Attempt {attempt+1}/{max_retries})...')
            init_url = 'https://api.linkedin.com/rest/images?action=initializeUpload'
            init_payload = {
                'initializeUploadRequest': {
                    'owner': author_urn
                }
            }
            resp = requests.post(init_url, json=init_payload, headers=headers)
            if resp.status_code != 200:
                print(f'❌ LinkedIn Initialize Upload Failed: {resp.status_code} {resp.text}')
                time.sleep(5 * (attempt + 1))
                continue

            upload_data = resp.json()['value']
            image_urn = upload_data['image']
            upload_url = upload_data['uploadUrl']

            print(f'Uploading image binary {image_path} to LinkedIn...')
            with open(image_path, 'rb') as f:
                img_data = f.read()

            up_resp = requests.put(upload_url, data=img_data, headers={'Authorization': f'Bearer {access_token}'})
            if up_resp.status_code != 201:
                print(f'❌ LinkedIn Physical Upload Failed: {up_resp.status_code}')
                time.sleep(5 * (attempt + 1))
                continue

            print(f'✓ LinkedIn Image created: {image_urn}')
            return image_urn
        except Exception as e:
            print(f'❌ Error during upload: {e}')
            time.sleep(5 * (attempt + 1))

    raise Exception('LinkedIn Image Upload failed after multiple attempts')


def upload_images_rest(image_paths, author_urn, access_token, max_retries=3):
    """Modern LinkedIn multi-image upload flow using /rest/images."""
    headers = {
        'Authorization': f'Bearer {access_token}',
        'LinkedIn-Version': LINKEDIN_VERSION,
        'X-Restli-Protocol-Version': '2.0.0'
    }

    urns = []
    for path in image_paths:
        path = path.replace("\\", "/")
        uploaded = False
        for attempt in range(max_retries):
            try:
                print(f'Initializing LinkedIn image upload for {path} (Attempt {attempt+1}/{max_retries})...')
                init_url = 'https://api.linkedin.com/rest/images?action=initializeUpload'
                init_payload = {'initializeUploadRequest': {'owner': author_urn}}
                resp = requests.post(init_url, json=init_payload, headers=headers)
                if resp.status_code != 200:
                    print(f'❌ LinkedIn Initialize Upload Failed: {resp.status_code} {resp.text}')
                    time.sleep(5 * (attempt + 1))
                    continue

                upload_data = resp.json()['value']
                image_urn = upload_data['image']
                upload_url = upload_data['uploadUrl']

                print(f'Uploading image binary {path} to LinkedIn...')
                with open(path, 'rb') as f:
                    img_data = f.read()

                up_resp = requests.put(upload_url, data=img_data, headers={'Authorization': f'Bearer {access_token}'})
                if up_resp.status_code != 201:
                    print(f'❌ LinkedIn Physical Upload Failed: {up_resp.status_code}')
                    time.sleep(5 * (attempt + 1))
                    continue

                print(f'✓ LinkedIn Image created: {image_urn}')
                urns.append(image_urn)
                uploaded = True
                break
            except Exception as e:
                print(f'❌ Error during upload: {e}')
                time.sleep(5 * (attempt + 1))

        if not uploaded:
            raise Exception(f'LinkedIn Image Upload failed for {path} after multiple attempts')

    return urns


def publish_carousel_linkedin(image_paths, caption, author_urn, access_token):
    """Publish a Wilma LinkedIn carousel from local slide images."""
    if not image_paths:
        raise ValueError('No images provided for carousel')

    urns = upload_images_rest(image_paths, author_urn, access_token)
    post_url = 'https://api.linkedin.com/rest/posts'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'LinkedIn-Version': LINKEDIN_VERSION,
        'X-Restli-Protocol-Version': '2.0.0'
    }

    if len(urns) == 1:
        content = {'media': {'id': urns[0], 'altText': caption[:100]}}
    else:
        content = {'multiImage': {'images': [{'id': urn} for urn in urns]}}

    post_payload = {
        'author': author_urn,
        'commentary': caption,
        'visibility': 'PUBLIC',
        'distribution': {'feedDistribution': 'MAIN_FEED'},
        'content': content,
        'lifecycleState': 'PUBLISHED'
    }

    print(f'Creating LinkedIn carousel post with {len(urns)} images...')
    post_resp = requests.post(post_url, json=post_payload, headers=headers)
    if post_resp.status_code == 201:
        print('✅ LinkedIn carousel post created successfully via REST API!')
        return True
    else:
        raise RuntimeError(f'LinkedIn carousel publish failed: {post_resp.status_code} {post_resp.text}')


def _resolve_active_bundle(state):
    active = state.get("active_bundle")
    if isinstance(active, (int, str)):
        for b in state.get("content_queue", []):
            if isinstance(b, dict) and b.get("post_id") == active:
                return b
            elif b == active:
                return {"post_id": b}
    return active


def publish_to_linkedin_rest():
    state = load_state(str(state_path))
    active = _resolve_active_bundle(state) or {}
    active = _resolve_active_bundle(state) or {}

    if not isinstance(active, dict) or not active.get("post_id"):
        queue = state.get("content_queue", [])
        if queue:
            state["active_bundle"] = queue.pop(0)
            state["content_queue"] = queue
            if "platforms_posted" not in state["active_bundle"]:
                state["active_bundle"]["platforms_posted"] = []
            if "platforms_prepared" not in state["active_bundle"]:
                state["active_bundle"]["platforms_prepared"] = []
            save_state(state, str(state_path))
            print(f"▶ Advanced stale active bundle to {state['active_bundle'].get('post_id')}. Remaining: {len(queue)}")
            active = state["active_bundle"]
        else:
            print("⏭️ Nothing new to post for LinkedIn. Skipping.")
            return

    flag_path = _linkedin_flag_path()
    if flag_path is None and "linkedin" not in (active.get("platforms_prepared") or []):
        print("⏭️ Nothing new to post for LinkedIn. Skipping.")
        return
    if flag_path is None:
        print("▶ No ready flag on disk, but active bundle was prepared for LinkedIn — proceeding.")
        flag_path = FORWILMA_DIR / "wilma_linkedin_ready.flag"

    token = get_fresh_linkedin_token()
    if not token or not LINKEDIN_URN:
        print('❌ Error: Unable to obtain Wilma LinkedIn access token or WILMA_LINKEDIN_URN missing.')
        sys.exit(1)

    # NORMALIZE URN
    author_urn = LINKEDIN_URN.strip()
    if author_urn.endswith('JI') and 'OXbkdK1uiJI' in author_urn:
        author_urn = author_urn[:-1]

    print(f'Publishing to LinkedIn (REST API {LINKEDIN_VERSION}) as author: {author_urn}')

    captions = active.get('captions') or {}
    caption = captions.get('linkedin') or ""

    if not caption and Path('caption.txt').exists():
        caption = Path('caption.txt').read_text(encoding='utf-8').strip()
        print(f"[CI align] fallback caption.txt len={len(caption)}")

    carousel_paths = active.get('carousel') or []
    raw_image = active.get('image') or 'output.jpg'
    image_path = raw_image.replace("\\", "/") if isinstance(raw_image, str) else str(raw_image)

    if not carousel_paths and not Path(image_path).exists() and Path('output.jpg').exists():
        image_path = 'output.jpg'

    if not caption:
        print('❌ No LinkedIn caption available for active bundle.')
        sys.exit(1)

    # Caption-only mode: if no image is available, post text-only.
    image_exists = Path(image_path).exists() if image_path else False
    if not image_exists:
        print(f"⚠ No image available for active bundle — posting text-only to LinkedIn.")
        _post_linkedin_text_only(caption, author_urn, token, flag_path)
        return

    try:
        success = False
        if carousel_paths:
            existing = [p for p in carousel_paths if Path(p).exists()]
            if existing:
                success = publish_carousel_linkedin(existing, caption, author_urn, token)
            else:
                print('⚠ Carousel bundle missing slides; falling back to single image.')
                if Path(image_path).exists():
                    image_urn = upload_image_rest(image_path, author_urn, token)
                    success = _create_linkedin_image_post(author_urn, token, caption, image_urn)
        else:
            image_urn = upload_image_rest(image_path, author_urn, token)
            success = _create_linkedin_image_post(author_urn, token, caption, image_urn)

        if success:
            update_state_after_post('linkedin', state_path=str(STATE_FILE))
            if flag_path.exists():
                flag_path.unlink()
                print(f'✓ Flag {flag_path} consumed.')
    except Exception as e:
        print(f'❌ LinkedIn automation failed: {e}')
        sys.exit(1)


def _post_linkedin_text_only(caption, author_urn, token, flag_path, state_path=str(STATE_FILE)):
    """Post a caption-only (no image) LinkedIn entry for Wilma."""
    post_url = 'https://api.linkedin.com/rest/posts'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'LinkedIn-Version': LINKEDIN_VERSION,
        'X-Restli-Protocol-Version': '2.0.0'
    }
    post_payload = {
        'author': author_urn,
        'commentary': caption,
        'visibility': 'PUBLIC',
        'distribution': {'feedDistribution': 'MAIN_FEED'},
        'lifecycleState': 'PUBLISHED'
    }

    print('Creating LinkedIn text-only post...')
    post_resp = requests.post(post_url, json=post_payload, headers=headers)
    if post_resp.status_code == 201:
        print('✅ LinkedIn text-only post created successfully via REST API!')
        update_state_after_post('linkedin', state_path=state_path)
        if flag_path.exists():
            flag_path.unlink()
            print(f'✓ Flag {flag_path} consumed.')
    else:
        raise RuntimeError(f'LinkedIn text-only publish failed: {post_resp.status_code} {post_resp.text}')


def _create_linkedin_image_post(author_urn, token, caption, image_urn):
    post_url = 'https://api.linkedin.com/rest/posts'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'LinkedIn-Version': LINKEDIN_VERSION,
        'X-Restli-Protocol-Version': '2.0.0'
    }
    post_payload = {
        'author': author_urn,
        'commentary': caption,
        'visibility': 'PUBLIC',
        'distribution': {'feedDistribution': 'MAIN_FEED'},
        'content': {'media': {'id': image_urn, 'altText': 'Digital Guardian - Wilma'}},
        'lifecycleState': 'PUBLISHED'
    }

    post_resp = requests.post(post_url, json=post_payload, headers=headers)
    if post_resp.status_code == 201:
        print('✅ LinkedIn post created successfully via REST API!')
        return True
    raise RuntimeError(f"LinkedIn single-image publish failed: {post_resp.status_code} {post_resp.text}")


if __name__ == '__main__':
    print("[CI align] script_md5=" + hashlib.md5(Path(__file__).read_bytes()).hexdigest())
    publish_to_linkedin_rest()
