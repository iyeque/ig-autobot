#!/usr/bin/env python3
import os
import sys
import requests
import json
import time

# Configuration
PINTEREST_ACCESS_TOKEN = os.environ.get("PINTEREST_ACCESS_TOKEN")
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

def publish_to_pinterest():
    if not PINTEREST_ACCESS_TOKEN or not PINTEREST_BOARD_ID:
        print("❌ Error: PINTEREST_ACCESS_TOKEN or PINTEREST_BOARD_ID missing.")
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
        # Pinterest API v5 prefers a public URL for 'image_url'.
        # We find the latest image from the images folder to ensure it's the one we just pushed.
        import glob
        pattern = "images/story_*.jpg" if image_path == "story.jpg" else "images/post_*.jpg"
        img_files = sorted(glob.glob(pattern), reverse=True)
        
        if img_files:
            image_url = DESTINATION_URL + img_files[0].replace('\\', '/')
        else:
            # Fallback to root files if not found in images/
            image_url = f"{DESTINATION_URL}{image_path}"
        
        print(f"Target Pin Image URL: {image_url}")
        
        # Ensure URL is live before calling Pinterest API
        if not check_url_live(image_url):
            print("❌ Pinterest Image URL not accessible. Aborting.")
            sys.exit(1)
        
        url = "https://api.pinterest.com/v5/pins"
        headers = {
            "Authorization": f"Bearer {PINTEREST_ACCESS_TOKEN}",
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
        max_retries = 3
        for attempt in range(max_retries):
            resp = requests.post(url, json=payload, headers=headers)
            if resp.status_code == 201:
                print("✅ Pinterest Pin created successfully!")
                return
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
