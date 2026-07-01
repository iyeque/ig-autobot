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

# Configuration from environment (GitHub Secrets or .env)
LINKEDIN_ACCESS_TOKEN = os.environ.get("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_URN = os.environ.get("LINKEDIN_URN") or os.environ.get("LINKEDIN_PERSON_URN") 
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

            with open(image_path, "rb") as f:
                img_data = f.read()
            
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

def upload_images_batch(image_paths, author_urn, access_token):
    """Upload multiple images and return list of image URNs."""
    urns = []
    for path in image_paths:
        urns.append(upload_image_rest(path, author_urn, access_token))
    return urns

def publish_carousel_linkedin(image_paths, caption, author_urn, access_token):
    """Publish a LinkedIn carousel (multi-image post) via REST API."""
    if not image_paths:
        raise ValueError("No images provided for carousel")

    print(f"Publishing LinkedIn carousel with {len(image_paths)} images...")
    urns = upload_images_batch(image_paths, author_urn, access_token)

    post_url = "https://api.linkedin.com/rest/posts"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "LinkedIn-Version": LINKEDIN_VERSION,
        "X-Restli-Protocol-Version": "2.0.0"
    }

    # Build multi-image payload
    if len(urns) == 1:
        content = {
            "media": {
                "id": urns[0],
                "altText": "Nine Stitches Carousel"
            }
        }
    else:
        content = {
            "multiImage": {
                "images": [{"id": urn} for urn in urns]
            }
        }

    post_payload = {
        "author": author_urn,
        "commentary": caption,
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED"
        },
        "content": content,
        "lifecycleState": "PUBLISHED"
    }

    post_resp = requests.post(post_url, json=post_payload, headers=headers)
    if post_resp.status_code == 201:
        return post_resp.json()
    
    # Fallback: if multiImage rejected, try single image with first slide
    print(f"⚠️ Carousel rejected ({post_resp.status_code}), falling back to single image...")
    if len(urns) > 1:
        fallback = {
            "author": author_urn,
            "commentary": caption,
            "visibility": "PUBLIC",
            "distribution": {"feedDistribution": "MAIN_FEED"},
            "content": {
                "media": {
                    "id": urns[0],
                    "altText": "Nine Stitches Content"
                }
            },
            "lifecycleState": "PUBLISHED"
        }
        post_resp = requests.post(post_url, json=fallback, headers=headers)
        if post_resp.status_code == 201:
            return post_resp.json()
    
    raise RuntimeError(f"LinkedIn carousel publish failed: {post_resp.status_code} {post_resp.text}")

def publish_to_linkedin_rest():
    # Staleness Protection
    flag_path = "linkedin_ready.flag"
    if is_platform_posted("linkedin"):
        print("⏭️ LinkedIn already posted for active bundle. Skipping.")
        return

    if not os.path.exists(flag_path):
        print("⏭️ Nothing new to post for LinkedIn. Skipping.")
        return

    # Get token for authentication
    token = LINKEDIN_ACCESS_TOKEN

    if not token or not LINKEDIN_URN:
        print("❌ Error: LINKEDIN_ACCESS_TOKEN or LINKEDIN_URN missing.")
        sys.exit(1)

    print(f"Publishing to LinkedIn (REST API {LINKEDIN_VERSION}) as author: {LINKEDIN_URN}")

    caption_path = "caption.txt"
    image_path = "output.jpg"
    state_dir = os.path.dirname(os.path.abspath(flag_path))

    if not os.path.exists(caption_path) or not os.path.exists(image_path):
        print("❌ Error: caption.txt or output.jpg missing.")
        sys.exit(1)

    with open(caption_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()

    try:
        # --- Carousel path ---
        carousel_json = os.path.join(state_dir, "carousel.json")
        if os.path.exists(carousel_json):
            with open(carousel_json, "r", encoding="utf-8") as f:
                carousel_paths = json.load(f)
            if carousel_paths:
                print(f"📱 Detected LinkedIn carousel ({len(carousel_paths)} slides)")
                publish_carousel_linkedin(carousel_paths, caption, LINKEDIN_URN, token)
                update_state_after_post("linkedin")
                if os.path.exists(flag_path):
                    os.remove(flag_path)
                    print(f"✓ Flag {flag_path} consumed.")
                return

        # --- Single image fallback ---
        if not os.path.exists(image_path):
            print("❌ Error: output.jpg missing and no carousel found.")
            sys.exit(1)

        image_urn = upload_image_rest(image_path, LINKEDIN_URN, token)

        print("Creating LinkedIn post...")
        post_url = "https://api.linkedin.com/rest/posts"
        headers = {
            "Authorization": f"Bearer {token}",
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
