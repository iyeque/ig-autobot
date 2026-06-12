#!/usr/bin/env python3
import os
import sys
import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to import shared_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_utils import update_state_after_post, is_platform_posted

# Load .env from project root if available
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from {dotenv_path}")

# Configuration
PINTEREST_ACCESS_TOKEN = os.environ.get("PINTEREST_ACCESS_TOKEN")
PINTEREST_REFRESH_TOKEN = os.environ.get("PINTEREST_REFRESH_TOKEN")
PINTEREST_APP_ID = os.environ.get("PINTEREST_APP_ID")
PINTEREST_APP_SECRET = os.environ.get("PINTEREST_APP_SECRET")
PINTEREST_BOARD_ID = os.environ.get("PINTEREST_BOARD_ID")
# Use the GitHub Pages URL as destination to drive traffic
DESTINATION_URL = "https://iyeque.github.io/ig-autobot/"

def check_url_live(url, max_retries=15, delay=20):
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

def get_fresh_access_token():
    """Exchanges a refresh token for a new access token."""
    if not PINTEREST_REFRESH_TOKEN or not PINTEREST_APP_ID or not PINTEREST_APP_SECRET:
        print("ℹ️ Refresh credentials missing, falling back to static PINTEREST_ACCESS_TOKEN")
        return PINTEREST_ACCESS_TOKEN

    print("Refreshing Pinterest Access Token...")
    import base64
    auth_string = f"{PINTEREST_APP_ID}:{PINTEREST_APP_SECRET}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    
    url = "https://api.pinterest.com/v5/oauth/token"
    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": PINTEREST_REFRESH_TOKEN
    }
    
    try:
        r = requests.post(url, headers=headers, data=data)
        if r.status_code == 200:
            new_token = r.json().get("access_token")
            print("✅ Successfully refreshed Pinterest Access Token.")
            return new_token
        else:
            print(f"❌ Failed to refresh token: {r.status_code} {r.text}")
            return PINTEREST_ACCESS_TOKEN
    except Exception as e:
        print(f"❌ Error during token refresh: {e}")
        return PINTEREST_ACCESS_TOKEN

def publish_to_pinterest():
    # Staleness Protection
    flag_path = "pinterest_ready.flag"
    if is_platform_posted("pinterest"):
        print("⏭️ Pinterest already posted for active bundle. Skipping.")
        return

    if not os.path.exists(flag_path):
        print("⏭️ Nothing new to post for Pinterest. Skipping.")
        return

    # Get a fresh token before starting
    token = get_fresh_access_token()

    if not token or not PINTEREST_BOARD_ID:
        print("❌ Error: Pinterest Access Token and PINTEREST_BOARD_ID are required.")
        sys.exit(1)

    # Pins look best vertical, so we prefer story.jpg if available
    image_path = "story.jpg"
    if not os.path.exists(image_path):
        image_path = "output.jpg"
        
    caption_path = "caption.txt"

    if not os.path.exists(image_path) or not os.path.exists(caption_path):
        print(f"❌ Error: {image_path} or caption.txt missing.")
        sys.exit(1)

    with open(caption_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()
    
    # Extract first line as title
    title = caption.split("\n")[0][:95]

    try:
        # Find latest image
        import glob
        pattern = "images/story_*.jpg" if image_path == "story.jpg" else "images/post_*.jpg"
        img_files = sorted(glob.glob(pattern), reverse=True)
        
        if img_files:
            image_url = DESTINATION_URL + img_files[0].replace('\\', '/')
        else:
            image_url = f"{DESTINATION_URL}{image_path}"
        
        print(f"Target Pin Image URL: {image_url}")
        
        if not check_url_live(image_url):
            print("❌ Pinterest Image URL not accessible. Aborting.")
            sys.exit(1)
        
        # USE SANDBOX URL for Trial Access apps
        base_url = "https://api-sandbox.pinterest.com/v5/pins"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "link": DESTINATION_URL,
            "title": title,
            "description": caption[:499],
            "board_id": PINTEREST_BOARD_ID,
            "media_source": {
                "source_type": "image_url",
                "url": image_url
            }
        }

        print(f"Creating Pin on board {PINTEREST_BOARD_ID}...")
        print(f"DEBUG: Using Access Token: {token[:10]}...") 
        max_retries = 3
        for attempt in range(max_retries):
            resp = requests.post(base_url, json=payload, headers=headers)
            if resp.status_code == 201:
                print("✅ Pinterest Pin created successfully!")
                update_state_after_post("pinterest")
                # Success: Consume flag
                if os.path.exists(flag_path):
                    os.remove(flag_path)
                    print(f"✓ Flag {flag_path} consumed.")
                return
            elif resp.status_code == 401:
                print(f"❌ Authentication Failed (401). Check if token is a Sandbox token.")
                print(f"DEBUG: Headers sent: {headers}")
                sys.exit(1)
            else:
                print(f"❌ Attempt {attempt+1} failed: {resp.status_code} {resp.text}")
                if attempt < max_retries - 1:
                    time.sleep(30)
        
        sys.exit(1)

    except Exception as e:
        print(f"❌ ERROR: Pinterest automation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    publish_to_pinterest()
