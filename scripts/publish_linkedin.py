#!/usr/bin/env python3
import os
import sys
import requests
import json
import time

# Add project root to path to import shared_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_utils import update_state_after_post

# Configuration from environment (GitHub Secrets)
LINKEDIN_ACCESS_TOKEN = os.environ.get("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_URN = os.environ.get("LINKEDIN_URN") 
# Use the latest stable version for LinkedIn REST API
LINKEDIN_VERSION = "202604"

def upload_image_rest(image_path, author_urn, access_token, max_retries=3):
    """Modern LinkedIn image upload flow using /rest/images (v202604+)"""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "LinkedIn-Version": LINKEDIN_VERSION,
        "X-Restli-Protocol-Version": "2.0.0"
    }

    for attempt in range(max_retries):
        try:
            # 1. Initialize Upload
            print(f"Initializing LinkedIn image upload (Attempt {attempt+1}/{max_retries})...")
            init_url = "https://api.linkedin.com/rest/images?action=initializeUpload"
            init_payload = {
                "initializeUploadRequest": {
                    "owner": author_urn
                }
            }
            resp = requests.post(init_url, json=init_payload, headers=headers)
            if resp.status_code != 200:
                print(f"❌ LinkedIn Initialize Upload Failed: {resp.status_code} {resp.text}")
                time.sleep(5 * (attempt + 1))
                continue
            
            upload_data = resp.json()["value"]
            image_urn = upload_data["image"]
            upload_url = upload_data["uploadUrl"]

            # 2. Upload Binary
            print(f"Uploading image binary {image_path} to LinkedIn...")
            with open(image_path, "rb") as f:
                img_data = f.read()
            
            # Binary upload doesn't need the Version header but needs Auth
            up_resp = requests.put(upload_url, data=img_data, headers={"Authorization": f"Bearer {access_token}"})
            if up_resp.status_code != 201:
                print(f"❌ LinkedIn Physical Upload Failed: {up_resp.status_code}")
                time.sleep(5 * (attempt + 1))
                continue

            print(f"✓ LinkedIn Image created: {image_urn}")
            return image_urn
        except Exception as e:
            print(f"❌ Error during upload: {e}")
            time.sleep(5 * (attempt + 1))

    raise Exception("LinkedIn Image Upload failed after multiple attempts")

def publish_to_linkedin_rest():
    # Staleness Protection
    flag_path = "linkedin_ready.flag"
    if not os.path.exists(flag_path):
        print("⏭️ Nothing new to post for LinkedIn. Skipping.")
        return

    if not LINKEDIN_ACCESS_TOKEN or not LINKEDIN_URN:
        print("❌ Error: LINKEDIN_ACCESS_TOKEN or LINKEDIN_URN missing.")
        sys.exit(1)

    print(f"Publishing to LinkedIn (REST API {LINKEDIN_VERSION}) as author: {LINKEDIN_URN}")

    caption_path = "caption.txt"
    image_path = "output.jpg"

    if not os.path.exists(caption_path) or not os.path.exists(image_path):
        print("❌ Error: caption.txt or output.jpg missing.")
        sys.exit(1)

    with open(caption_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()

    try:
        # 1. Upload media
        image_urn = upload_image_rest(image_path, LINKEDIN_URN, LINKEDIN_ACCESS_TOKEN)

        # 2. Create post
        print("Creating LinkedIn post...")
        post_url = "https://api.linkedin.com/rest/posts"
        headers = {
            "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "Content-Type": "application/json",
            "LinkedIn-Version": LINKEDIN_VERSION,
            "X-Restli-Protocol-Version": "2.0.0"
        }
        post_payload = {
            "author": LINKEDIN_URN,
            "commentary": caption,
            "visibility": "PUBLIC",
            "distribution": {
                "feedDistribution": "MAIN_FEED"
            },
            "content": {
                "media": {
                    "id": image_urn,
                    "altText": "Nine Stitches Content"
                }
            },
            "lifecycleState": "PUBLISHED"
        }
        
        post_resp = requests.post(post_url, json=post_payload, headers=headers)
        if post_resp.status_code == 201:
            print("✅ LinkedIn post created successfully via REST API!")
            update_state_after_post("linkedin")
            # Success: Consume flag
            if os.path.exists(flag_path):
                os.remove(flag_path)
                print(f"✓ Flag {flag_path} consumed.")
        else:
            print(f"❌ Failed to create post: {post_resp.status_code} {post_resp.text}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ LinkedIn automation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    publish_to_linkedin_rest()
