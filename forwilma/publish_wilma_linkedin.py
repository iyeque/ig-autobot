#!/usr/bin/env python3
import os
import sys
import json
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to import shared_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_utils import update_state_after_post

# Load .env from project root if available
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

# Configuration from environment (STRICTLY WILMA ONLY)
# Wilma uses refresh-token flow; static token fallback removed.
LINKEDIN_REFRESH_TOKEN = os.environ.get('WILMA_LINKEDIN_REFRESH_TOKEN')
LINKEDIN_CLIENT_ID = os.environ.get('WILMA_LINKEDIN_CLIENT_ID')
LINKEDIN_CLIENT_SECRET = os.environ.get('WILMA_LINKEDIN_CLIENT_SECRET')
LINKEDIN_URN = os.environ.get('WILMA_LINKEDIN_URN')

# Use the latest stable version for LinkedIn REST API
LINKEDIN_VERSION = '202604'

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
            # 1. Initialize Upload
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

            # 2. Upload Binary
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

def publish_to_linkedin_rest():
    # Staleness Protection / queue advance
    state = _read_state_path(STATE_FILE)
    flag_path = Path('wilma_linkedin_ready.flag')
    active = state.get('active_bundle') or {}
    if not flag_path.exists() or not active:
        print("⏭️ Nothing new to post for Wilma's LinkedIn. Skipping.")
        return
    if 'linkedin' in (active.get('platforms_posted') or []):
        queue = state.get('content_queue', [])
        if queue:
            state['active_bundle'] = queue.pop(0)
            state['active_bundle']['platforms_posted'] = []
            state['active_bundle']['platforms_prepared'] = []
            _write_state(state)
            print(f"▶ Advanced active bundle to {state['active_bundle'].get('post_id')}. Remaining: {len(queue)}")
        else:
            state['active_bundle'] = None
            _write_state(state)
            print("▶ Queue empty; cleared active bundle.")
        return

    # Get fresh token (refresh-token flow only; no static fallback)
    token = get_fresh_linkedin_token()

    if not token or not LINKEDIN_URN:
        print('❌ Error: Unable to obtain Wilma LinkedIn access token or WILMA_LINKEDIN_URN missing.')
        sys.exit(1)

    # NORMALIZE URN
    author_urn = LINKEDIN_URN.strip()
    if author_urn.endswith('JI') and 'OXbkdK1uiJI' in author_urn:
         author_urn = author_urn[:-1]

    print(f'Publishing to LinkedIn (REST API {LINKEDIN_VERSION}) as author: {author_urn}')

    active = _read_state_path(STATE_FILE)
    if not active:
        print('❌ No active_bundle in state.')
        sys.exit(1)
    captions = active.get('captions') or {}
    caption = captions.get('linkedin') or ""
    image_path = active.get('image') or 'output.jpg'
    if not caption and Path('caption.txt').exists():
        caption = Path('caption.txt').read_text(encoding='utf-8').strip()
    if not Path(image_path).exists() and Path('output.jpg').exists():
        image_path = 'output.jpg'
    if not caption:
        print('❌ No LinkedIn caption available for active bundle.')
        sys.exit(1)
    if not Path(image_path).exists():
        print(f'❌ Image not found for active bundle: {image_path}')
        sys.exit(1)

    try:
        # 1. Upload media
        image_urn = upload_image_rest(image_path, author_urn, token)

        # 2. Create post
        print('Creating LinkedIn post...')
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
            'distribution': {
                'feedDistribution': 'MAIN_FEED'
            },
            'content': {
                'media': {
                    'id': image_urn,
                    'altText': 'Digital Guardian - Wilma'
                }
            },
            'lifecycleState': 'PUBLISHED'
        }
        
        post_resp = requests.post(post_url, json=post_payload, headers=headers)
        if post_resp.status_code == 201:
            print('✅ LinkedIn post created successfully via REST API!')
            update_state_after_post('linkedin', state_path='state.json')
            # Success: Consume flag
            if flag_path.exists():
                flag_path.unlink()
                print(f'✓ Flag {flag_path} consumed.')
        else:
            print(f'❌ Failed to create post: {post_resp.status_code} {post_resp.text}')
            sys.exit(1)

    except Exception as e:
        print(f'❌ LinkedIn automation failed: {e}')
        sys.exit(1)

if __name__ == '__main__':
    publish_to_linkedin_rest()
