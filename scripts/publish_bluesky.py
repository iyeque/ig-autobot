#!/usr/bin/env python3
import os
import sys
from atproto import Client, models

def publish_to_bluesky():
    handle = os.environ.get("BLUESKY_HANDLE")
    password = os.environ.get("BLUESKY_PASSWORD") # App Password
    
    if not handle or not password:
        print("❌ BLUESKY_HANDLE or BLUESKY_PASSWORD not set")
        sys.exit(1)

    caption_path = "caption.txt"
    image_path = "output.jpg"

    if not os.path.exists(caption_path) or not os.path.exists(image_path):
        print(f"❌ {caption_path} or {image_path} missing")
        sys.exit(1)

    with open(caption_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()
    
    # Final hard limit safety check
    if len(caption) > 300:
        caption = caption[:290] + "..."

    client = Client()
    try:
        print(f"Logging into Bluesky as {handle}...")
        client.login(handle, password)
        
        print(f"Uploading image {image_path}...")
        with open(image_path, "rb") as f:
            img_data = f.read()
            
        upload = client.upload_blob(img_data)
        
        print("Creating post...")
        embed = models.AppBskyEmbedImages.Main(
            images=[models.AppBskyEmbedImages.Image(alt=caption[:100], image=upload.blob)]
        )
        
        client.send_post(text=caption, embed=embed)
        print("✅ Successfully posted to Bluesky!")
        
    except Exception as e:
        print(f"❌ Failed to post to Bluesky: {e}")
        sys.exit(1)

if __name__ == "__main__":
    publish_to_bluesky()
