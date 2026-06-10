#!/usr/bin/env python3
import os
import sys
import json
import time
import requests
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

def publish_to_facebook(page_id, access_token, image_path, caption):
    """
    Publishes a photo to a Facebook Page feed.
    """
    print(f"Publishing to Facebook Page {page_id}...")
    
    # Use the /photos endpoint which is standard for Page image posts
    url = f"https://graph.facebook.com/v19.0/{page_id}/photos"
    
    is_url = image_path.startswith("http")
    
    # Ensure the caption isn't too long for FB (though usually not an issue)
    payload = {
        "caption": caption,
        "access_token": access_token,
        "published": "true" # Explicitly mark as published
    }
    
    try:
        if is_url:
            print(f"Attempting URL-based upload: {image_path}")
            payload["url"] = image_path
            response = requests.post(url, data=payload)
        else:
            print(f"Attempting local binary upload: {image_path}")
            if not os.path.exists(image_path):
                print(f"❌ Error: Image file not found at {image_path}")
                return False
            
            with open(image_path, "rb") as f:
                files = {"source": f}
                response = requests.post(url, data=payload, files=files)
        
        res_data = response.json()
        
        if response.status_code == 200 and "id" in res_data:
            print(f"✅ Successfully posted to Facebook! Post ID: {res_data['id']}")
            return True
        else:
            # Check for specific "publish_actions" error to provide better advice
            error_msg = res_data.get("error", {}).get("message", "")
            if "publish_actions" in error_msg:
                print("❌ DEPRECATION ERROR DETECTED")
                print("Advice: This usually means you are using a USER token instead of a PAGE token.")
                print("Please ensure FB_PAGE_ACCESS_TOKEN is a 'Page Access Token' from the Graph API Explorer.")
            
            print(f"❌ Facebook API Error: {res_data}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during Facebook publishing: {e}")
        return False

def main():
    # Staleness Protection: Check for ready flag
    # Facebook is usually posted as part of the Instagram workflow
    flag_path = "instagram_ready.flag" 
    if not os.path.exists(flag_path):
        print("⏭️ Nothing new to post for Facebook/Instagram. Skipping.")
        return

    # Load configuration from environment
    page_id = os.environ.get("FB_PAGE_ID")
    access_token = os.environ.get("FB_PAGE_ACCESS_TOKEN")
    
    # Optional fallback to IG_ACCESS_TOKEN if it has Page scopes
    if not access_token:
        access_token = os.environ.get("IG_ACCESS_TOKEN")
        
    if not page_id or not access_token:
        print("❌ Error: FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN (or IG_ACCESS_TOKEN) must be set.")
        sys.exit(1)

    # Determine image path
    # We prefer output.jpg as it is the guaranteed result of the latest generation
    final_image_source = "output.jpg"
    
    if not os.path.exists(final_image_source):
        print(f"⚠ {final_image_source} not found, searching in images/ as fallback...")
        if os.path.exists("images"):
            import glob
            images = glob.glob("images/post_*.png") + glob.glob("images/post_*.jpg")
            if images:
                latest_image = max(images, key=os.path.getmtime)
                final_image_source = latest_image
                print(f"✓ Using latest image found: {final_image_source}")

    # Load caption
    caption = ""
    if os.path.exists("caption.txt"):
        with open("caption.txt", "r", encoding="utf-8") as f:
            caption = f.read()
    else:
        print("❌ Error: caption.txt not found.")
        sys.exit(1)

    success = publish_to_facebook(page_id, access_token, final_image_source, caption)
    if not success:
        sys.exit(1)
    
    update_state_after_post("facebook")
    
    # Success: Consume flag
    if os.path.exists(flag_path):
        os.remove(flag_path)
        print(f"✓ Flag {flag_path} consumed (FB/IG).")

if __name__ == "__main__":
    main()
