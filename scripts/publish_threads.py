#!/usr/bin/env python3
import os
import sys
import requests
import time

def wait_for_threads_media(creation_id, access_token, max_checks=25, delay=12):
    """Waits for Threads to finish processing the uploaded media."""
    url = f"https://graph.threads.net/v1.0/{creation_id}"
    params = {"fields": "status,error_message", "access_token": access_token}
    
    for i in range(max_checks):
        r = requests.get(url, params=params)
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

def publish_to_threads():
    user_id = os.environ.get("THREADS_USER_ID", "me")
    access_token = os.environ.get("THREADS_ACCESS_TOKEN")
    
    if not access_token:
        print("❌ THREADS_ACCESS_TOKEN not set")
        sys.exit(1)

    caption = ""
    if os.path.exists("caption.txt"):
        with open("caption.txt", "r", encoding="utf-8") as f:
            caption = f.read().strip()
            
    # Mechanical truncation removed. Bot.py now ensures AI generates within limits.

    # Determine media type and URL (must be public)
    base_url = "https://iyeque.github.io/ig-autobot/"
    
    media_url = ""
    media_type = "TEXT" # Default if no media
    
    import glob
    
    if os.path.exists("reel.mp4"):
        # Threads supports video
        reel_files = sorted(glob.glob("reels/reel_*.mp4"), reverse=True)
        if reel_files:
            media_url = base_url + reel_files[0].replace('\\', '/')
        else:
            media_url = base_url + "reel.mp4"
        media_type = "VIDEO"
    elif os.path.exists("output.jpg"):
        img_files = sorted(glob.glob("images/post_*.jpg"), reverse=True)
        if not img_files:
            img_files = sorted(glob.glob("images/post_*.png"), reverse=True)
            
        if img_files:
            media_url = base_url + img_files[0].replace('\\', '/')
        else:
            media_url = base_url + "output.jpg"
        media_type = "IMAGE"

    print(f"Creating Threads container (Type: {media_type})...")
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

    r = requests.post(container_url, data=payload)
    res = r.json()
    creation_id = res.get("id")
    
    if not creation_id:
        print(f"❌ Failed to create Threads container: {res}")
        sys.exit(1)

    print(f"Waiting for container {creation_id}...")
    if wait_for_threads_media(creation_id, access_token):
        print(f"Publishing Threads container {creation_id}...")
        publish_url = f"https://graph.threads.net/v1.0/{user_id}/threads_publish"
        r = requests.post(publish_url, data={
            "creation_id": creation_id,
            "access_token": access_token
        })
        res = r.json()
        if "id" in res:
            print(f"✅ Successfully posted to Threads! Post ID: {res['id']}")
        else:
            print(f"❌ Threads publish failed: {res}")
            sys.exit(1)
    else:
        sys.exit(1)

if __name__ == "__main__":
    publish_to_threads()
