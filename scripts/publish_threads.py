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
from shared_utils import update_state_after_post, is_platform_posted, get_active_bundle, resolve_bundle_media, advance_stale_active_bundle

# Load .env from project root if available
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from {dotenv_path}")

THREADS_APP_ID = os.environ.get("THREADS_APP_ID")
THREADS_APP_SECRET = os.environ.get("THREADS_APP_SECRET")
THREADS_REFRESH_TOKEN = os.environ.get("THREADS_REFRESH_TOKEN")

def refresh_threads_access_token(refresh_token, client_id, client_secret):
    """Exchange a Meta refresh token for a new Threads access token."""
    if not refresh_token or not client_id or not client_secret:
        return None
    url = "https://graph.facebook.com/v19.0/oauth/access_token"
    data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }
    try:
        r = requests.post(url, data=data, timeout=30)
        if r.status_code == 200:
            token = r.json().get("access_token")
            if token:
                print("✅ Refreshed Threads access token.")
                return token
        print(f"❌ Threads token refresh failed: {r.status_code} {r.text[:300]}")
    except Exception as e:
        print(f"❌ Threads token refresh error: {e}")
    return None

def check_url_live(url, max_retries=3, delay=5):
    """Checks if the URL is publicly accessible before proceeding."""
    print(f"Checking if {url} is live...")
    for i in range(max_retries):
        try:
            r = requests.head(url, timeout=10)
            if r.status_code == 200:
                print(f"✓ URL is live (Attempt {i+1})")
                return True
            print(f"Status {r.status_code} for {url}. Waiting {delay}s... (Attempt {i+1}/{max_retries})")
        except Exception as e:
            print(f"Error checking {url}: {e}. Waiting {delay}s... (Attempt {i+1}/{max_retries})")
        time.sleep(delay)
    return False

def wait_for_threads_media(creation_id, access_token, max_checks=25, delay=12):
    """Waits for Threads to finish processing the uploaded media."""
    url = f"https://graph.threads.net/v1.0/{creation_id}"
    params = {"fields": "status,error_message", "access_token": access_token}
    
    for i in range(max_checks):
        r = requests.get(url, params=params)
        if r.status_code == 401:
            err = r.json()
            raise RuntimeError(f"Auth error (401): {err.get('message', 'token expired or invalid')}")
        data = r.json()
        status = data.get("status", "UNKNOWN")
        print(f"  Threads media {creation_id} status: {status}")
        if status == "FINISHED":
            return True
        if status == "ERROR":
            print(f"❌ Threads media processing error: {data.get('error_message')}")
            return False
        time.sleep(delay)
    return False

def create_threads_container(container_url, payload):
    """POSTs container creation and returns the parsed JSON response,
    printing full diagnostic detail on any failure instead of crashing silently."""
    try:
        r = requests.post(container_url, data=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error creating Threads container: {e}")
        sys.exit(1)

    print(f"Container creation HTTP status: {r.status_code}")

    try:
        res = r.json()
    except ValueError:
        print(f"❌ Threads returned non-JSON response (status {r.status_code}):")
        print(r.text[:1000])
        sys.exit(1)

    if r.status_code != 200:
        err_body = json.dumps(res, indent=2) if res else r.text
        raise RuntimeError(f"Threads API error (status {r.status_code}): {err_body}")

    return res

def publish_to_threads():
    flag_path = Path("threads_ready.flag")
    if is_platform_posted("threads"):
        advanced = advance_stale_active_bundle()
        if advanced:
            print("▶ Advanced stale active bundle instead of skipping.")
        else:
            print("⏭️ Threads already posted for active bundle. Skipping.")
        return

    if not flag_path.exists():
        print("⏭️ Nothing new to post for Threads. Skipping.")
        return

    active = get_active_bundle() or {}
    media = resolve_bundle_media(active)

    user_id = os.environ.get("THREADS_USER_ID")

    access_token = os.environ.get("THREADS_ACCESS_TOKEN")
    
    # Attempt token refresh if we have the required credentials.
    # This mirrors the pattern used in publish_linkedin.py.
    token_missing = not access_token or str(access_token).strip() in {"", "EXPIRED_ACCESS_TOKEN"}
    refresh_configured = bool(THREADS_REFRESH_TOKEN and THREADS_APP_ID and THREADS_APP_SECRET)
    
    if token_missing and refresh_configured:
        print("🔄 Attempting to refresh Threads access token...")
        refreshed = refresh_threads_access_token(
            THREADS_REFRESH_TOKEN, THREADS_APP_ID, THREADS_APP_SECRET
        )
        if refreshed:
            access_token = refreshed
            print(f"   Using refreshed token: {access_token[:20]}...")
    
    if not access_token:
        print("❌ THREADS_ACCESS_TOKEN not set (and refresh failed or not configured)")
        sys.exit(1)

    if not user_id:
        print("❌ THREADS_USER_ID not set")
        sys.exit(1)

    caption = ""
    if os.path.exists("caption.txt"):
        with open("caption.txt", "r", encoding="utf-8") as f:
            caption = f.read().strip()
            
    # Redundant safety: Threads has a strict 500 character limit
    if len(caption) > 500:
        print(f"⚠ WARNING: Caption ({len(caption)} chars) exceeds Threads limit. Truncating for safety.")
        caption = caption[:497] + "..."

    # Determine media type and URL (must be public)
    base_url = "https://iyeque.github.io/ig-autobot/"
    
    media_url = ""
    media_type = "TEXT" # Default if no media
    
    import glob
    
    if media.get("reel") and (os.path.exists("reel.mp4") or active.get("reel")):
        media_url = media["reel"]
        media_type = "VIDEO"
    elif media.get("image"):
        media_url = media["image"]
        media_type = "IMAGE"
    elif os.path.exists("output.jpg"):
        media_url = base_url + "images/output.jpg"
        media_type = "IMAGE"

    # Verify the media is actually live on Pages before asking Threads to fetch it.
    if media_type in ("IMAGE", "VIDEO"):
        checked = check_url_live(media_url)
        if not checked:
                # If the canonical prepared URL is not on Pages yet, try the actual
                # bundle path directly (e.g. reels/reel_20260702_055342.mp4).
                fallbacks = []
                if media_type == "VIDEO" and active.get("reel"):
                    fallbacks.append(base_url + active["reel"].replace("\\", "/"))
                if media_type == "IMAGE" and active.get("image"):
                    fallbacks.append(base_url + active["image"].replace("\\", "/"))
                for fb in fallbacks:
                    print(f"Primary URL not accessible. Trying fallback: {fb}")
                    if check_url_live(fb):
                        media_url = fb
                        checked = True
                        break
                if not checked:
                    print(f"❌ Media URL not accessible: {media_url}. Aborting.")
                    sys.exit(1)

    print(f"Creating Threads container (Type: {media_type})...")
    print(f"  media_url: {media_url}")
    container_url = f"https://graph.threads.net/v1.0/{user_id}/threads"
    payload = {
        "media_type": media_type,
        "text": caption,
        "access_token": access_token
    }
    if media_type == "IMAGE":
        payload["image_url"] = media_url
    elif media_type == "VIDEO":
        payload["video_url"] = media_url

    res = create_threads_container(container_url, payload)
    creation_id = res.get("id")

    if not creation_id:
        print(f"❌ Failed to create Threads container: {res}")
        sys.exit(1)

    print(f"Waiting for container {creation_id}...")
    # wait_for_threads_media uses the access_token — if it expires mid-wait,
    # the polling calls will fail. We handle that by retrying once with a refresh.
    try:
        container_ready = wait_for_threads_media(creation_id, access_token)
    except RuntimeError as exc:
        # 401 / expired-token → attempt refresh and retry wait once
        if "401" in str(exc) or "expir" in str(exc).lower():
            print(f"⚠ Container wait failed (token may have expired): {exc}")
            print("   Attempting mid-run token refresh...")
            if THREADS_REFRESH_TOKEN and THREADS_APP_ID and THREADS_APP_SECRET:
                new_token = refresh_threads_access_token(
                    THREADS_REFRESH_TOKEN, THREADS_APP_ID, THREADS_APP_SECRET
                )
                if new_token:
                    access_token = new_token
                    print("   Refreshed — re-checking container status...")
                    container_ready = wait_for_threads_media(creation_id, access_token)
                else:
                    print("   ❌ Refresh failed; aborting.")
                    sys.exit(1)
            else:
                print("   ❌ No refresh token configured; aborting.")
                sys.exit(1)
        else:
            raise

    if container_ready:
        print(f"Publishing Threads container {creation_id}...")
        publish_url = f"https://graph.threads.net/v1.0/{user_id}/threads_publish"
        try:
            r = requests.post(publish_url, data={
                "creation_id": creation_id,
                "access_token": access_token
            })
        except RuntimeError as exc:
            # publish step also covered by mid-run refresh
            if "401" in str(exc) or "expir" in str(exc).lower():
                print(f"⚠ Publish failed (token may have expired): {exc}")
                print("   Attempting mid-run token refresh...")
                if THREADS_REFRESH_TOKEN and THREADS_APP_ID and THREADS_APP_SECRET:
                    new_token = refresh_threads_access_token(
                        THREADS_REFRESH_TOKEN, THREADS_APP_ID, THREADS_APP_SECRET
                    )
                    if new_token:
                        access_token = new_token
                        print("   Refreshed — re-publishing...")
                        r = requests.post(publish_url, data={
                            "creation_id": creation_id,
                            "access_token": access_token
                        })
                    else:
                        print("   ❌ Refresh failed; aborting.")
                        sys.exit(1)
                else:
                    print("   ❌ No refresh token configured; aborting.")
                    sys.exit(1)
            else:
                raise

        res = r.json()
        if "id" in res:
            print(f"✅ Successfully posted to Threads! Post ID: {res['id']}")
            update_state_after_post("threads")
            # Success: Consume the flag
            if os.path.exists(flag_path):
                os.remove(flag_path)
                print(f"✓ Flag {flag_path} consumed.")
        else:
            print(f"❌ Threads publish failed: {res}")
            sys.exit(1)
    else:
        sys.exit(1)

if __name__ == "__main__":
    publish_to_threads()