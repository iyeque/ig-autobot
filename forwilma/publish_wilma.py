#!/usr/bin/env python3
import os
import sys
import requests
import json
import time

# Configuration from environment
LINKEDIN_ACCESS_TOKEN = os.environ.get("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_URN = os.environ.get("LINKEDIN_URN") # Should be urn:li:person:SUB_ID

def publish_to_linkedin_rest():
    if not LINKEDIN_ACCESS_TOKEN or not LINKEDIN_URN:
        print("❌ Error: LINKEDIN_ACCESS_TOKEN or LINKEDIN_URN missing.")
        sys.exit(1)

    print(f"Publishing to LinkedIn (REST API) as author: {LINKEDIN_URN}")

    caption_path = "caption.txt"
    image_path = "output.jpg"

    if not os.path.exists(caption_path) or not os.path.exists(image_path):
        print("❌ Error: caption.txt or output.jpg missing.")
        sys.exit(1)

    with open(caption_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()

    # Step 1: Register Image
    print("Registering image asset...")
    register_url = "https://api.linkedin.com/rest/images?action=initializeUpload"
    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
        "LinkedIn-Version": "202604",
        "X-Restli-Protocol-Version": "2.0.0"
    }
    register_payload = {
        "initializeUploadRequest": {
            "owner": LINKEDIN_URN
        }
    }
    
    resp = requests.post(register_url, json=register_payload, headers=headers)
    if resp.status_code != 200:
        print(f"❌ Failed to register image: {resp.text}")
        sys.exit(1)
    
    upload_data = resp.json()["value"]
    image_asset = upload_data["image"]
    upload_url = upload_data["uploadUrl"]

    # Step 2: Upload Image
    print("Uploading image file...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    up_resp = requests.put(upload_url, data=image_bytes, headers={"Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}"})
    if up_resp.status_code != 201:
        print(f"❌ Image upload failed: {up_resp.status_code}")
        sys.exit(1)

    # Step 3: Create Post
    print("Creating post...")
    post_url = "https://api.linkedin.com/rest/posts"
    post_payload = {
        "author": LINKEDIN_URN,
        "commentary": caption,
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": []
        },
        "content": {
            "media": {
                "title": caption.split('\n')[0][:100],
                "id": image_asset
            }
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False
    }
    
    post_resp = requests.post(post_url, json=post_payload, headers=headers)
    if post_resp.status_code == 201:
        print("✅ LinkedIn post created successfully via REST API!")
    else:
        print(f"❌ Failed to create post: {post_resp.status_code} {post_resp.text}")
        sys.exit(1)

if __name__ == "__main__":
    publish_to_linkedin_rest()
