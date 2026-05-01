#!/usr/bin/env python3
import os
import sys
import requests
import json
import time

# Configuration from environment
LINKEDIN_ACCESS_TOKEN = os.environ.get("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_URN = os.environ.get("LINKEDIN_URN") # Can be person, member, or company

def upload_to_linkedin(image_path, author_urn, access_token, max_retries=3):
    """Modern LinkedIn image upload flow (Register -> Upload -> Verify) using v2 API"""
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
                print(f"❌ LinkedIn Register Upload Failed: {reg_resp.status_code} {reg_resp.text}")
                time.sleep(5 * (attempt + 1))
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
                time.sleep(5 * (attempt + 1))
                continue

            print(f"✓ LinkedIn Asset created: {asset_urn}")
            return asset_urn
        except Exception as e:
            print(f"❌ Error during upload: {e}")
            time.sleep(5 * (attempt + 1))

    raise Exception("LinkedIn Upload failed after multiple attempts")

def publish_to_linkedin_v2():
    if not LINKEDIN_ACCESS_TOKEN or not LINKEDIN_URN:
        print("❌ Error: LINKEDIN_ACCESS_TOKEN or LINKEDIN_URN missing.")
        sys.exit(1)

    # USE URN AS IS: LinkedIn v2 API (ugcPosts) uses urn:li:person: or urn:li:organization:
    author_urn = LINKEDIN_URN.strip()
    print(f"Publishing to LinkedIn (v2 API) as author: {author_urn}")

    caption_path = "caption.txt"
    image_path = "output.jpg"

    if not os.path.exists(caption_path) or not os.path.exists(image_path):
        print("❌ Error: caption.txt or output.jpg missing.")
        sys.exit(1)

    with open(caption_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()

    try:
        # 1. Upload media (Using the normalized URN as owner)
        asset_urn = upload_to_linkedin(image_path, author_urn, LINKEDIN_ACCESS_TOKEN)

        # 2. Create post
        print("Creating LinkedIn post...")
        post_url = "https://api.linkedin.com/v2/ugcPosts"
        headers = {
            "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0"
        }
        post_payload = {
            "author": author_urn,
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
        
        # Log the author URN just before sending the request
        print(f"DEBUG: Author URN in post_payload: {post_payload.get('author')}")

        post_resp = requests.post(post_url, json=post_payload, headers=headers)
        if post_resp.status_code == 201:
            print("✅ LinkedIn post created successfully via v2 API!")
        else:
            print(f"❌ Failed to create post: {post_resp.status_code} {post_resp.text}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ LinkedIn automation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    publish_to_linkedin_v2()
