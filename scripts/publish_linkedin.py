#!/usr/bin/env python3
import os
import sys
import requests
import json
import time

# Configuration from environment (GitHub Secrets)
LINKEDIN_ACCESS_TOKEN = os.environ.get("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_URN = os.environ.get("LINKEDIN_URN", "urn:li:person:MPB6BAvzm7")

def upload_to_linkedin(image_path, author_urn, access_token, max_retries=3):
    """Modern LinkedIn image upload flow (Register -> Upload -> Verify)"""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-Restli-Protocol-Version": "2.0.0"
    }

    for attempt in range(max_retries):
        try:
            # 1. Register Upload
            print(f"Registering LinkedIn upload (Attempt {attempt+1}/{max_retries})...")
            register_url = "https://api.linkedin.com/v2/assets?action=registerUpload"
            register_payload = {
                "registerUploadRequest": {
                    "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                    "owner": author_urn,
                    "serviceRelationships": [{"relationshipType": "OWNER", "identifier": "urn:li:userGeneratedContent"}]
                }
            }
            reg_resp = requests.post(register_url, json=register_payload, headers=headers)
            if reg_resp.status_code != 200:
                print(f"❌ LinkedIn Register Upload Failed: {reg_resp.text}")
                time.sleep(10 * (attempt + 1))
                continue
            
            upload_data = reg_resp.json()["value"]
            asset_urn = upload_data["asset"]
            upload_url = upload_data["uploadMechanism"]["com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"]["uploadUrl"]

            # 2. Physical Upload
            print(f"Uploading image {image_path} to LinkedIn...")
            with open(image_path, "rb") as f:
                img_data = f.read()
            
            up_resp = requests.put(upload_url, data=img_data, headers={"Authorization": f"Bearer {access_token}"})
            if up_resp.status_code != 201:
                print(f"❌ LinkedIn Physical Upload Failed: {up_resp.status_code}")
                time.sleep(10 * (attempt + 1))
                continue

            print(f"✓ LinkedIn Asset created: {asset_urn}")
            return asset_urn
        except Exception as e:
            print(f"❌ Error during upload: {e}")
            time.sleep(10 * (attempt + 1))

    raise Exception("LinkedIn Upload failed after multiple attempts")

def publish_to_linkedin():
    # Staleness Protection
    flag_path = "linkedin_ready.flag"
    if not os.path.exists(flag_path):
        print("⏭️ Nothing new to post for LinkedIn. Skipping.")
        return

    if not LINKEDIN_ACCESS_TOKEN:
        print("❌ Error: LINKEDIN_ACCESS_TOKEN not found.")
        sys.exit(1)

    print(f"Publishing to LinkedIn as author: {LINKEDIN_URN}")

    # 1. Prepare data
    caption_path = "caption.txt"
    image_path = "output.jpg"

    if not os.path.exists(caption_path) or not os.path.exists(image_path):
        print("❌ Error: caption.txt or output.jpg missing.")
        sys.exit(1)

    with open(caption_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()

    try:
        # 2. Upload media
        asset_urn = upload_to_linkedin(image_path, LINKEDIN_URN, LINKEDIN_ACCESS_TOKEN)

        # 3. Create post
        print("Creating LinkedIn post...")
        li_url = "https://api.linkedin.com/v2/ugcPosts"
        li_headers = {
            "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0"
        }
        li_payload = {
            "author": LINKEDIN_URN,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": caption},
                    "shareMediaCategory": "IMAGE",
                    "media": [{"status": "READY", "media": asset_urn}]
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"}
        }
        
        li_resp = requests.post(li_url, json=li_payload, headers=li_headers)
        if li_resp.status_code == 201:
            print("✅ LinkedIn posted successfully!")
            # Success: Consume flag
            if os.path.exists(flag_path):
                os.remove(flag_path)
                print(f"✓ Flag {flag_path} consumed.")
        else:
            print(f"❌ ERROR: LinkedIn post failed: {li_resp.status_code} {li_resp.text}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ ERROR: LinkedIn automation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    publish_to_linkedin()
