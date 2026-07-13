#!/usr/bin/env python3
import os
import sys
import json
from atproto import Client, models
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

# Setup paths
FORWILMA_DIR = Path(__file__).parent
os.chdir(str(FORWILMA_DIR))

def publish_wilma_to_bluesky():
    # Staleness Protection
    flag_path = "wilma_bluesky_ready.flag"
    if not os.path.exists(flag_path):
        print("⏭️ Nothing new to post for Wilma's Bluesky. Skipping.")
        return

    # Wilma-specific credentials
    handle = os.environ.get("WILMA_BLUESKY_HANDLE")
    password = os.environ.get("WILMA_BLUESKY_PASSWORD") 
    
    if not handle or not password:
        print("❌ WILMA_BLUESKY_HANDLE or WILMA_BLUESKY_PASSWORD not set")
        sys.exit(1)

    # 1. Read Caption
    caption_path = "caption.txt"
    if not os.path.exists(caption_path):
        print(f"❌ {caption_path} not found")
        sys.exit(1)
        
    with open(caption_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()
    
    # Last resort safety check (Bluesky 300 char limit)
    if len(caption) > 300:
        print(f"⚠ WARNING: Caption too long ({len(caption)}). Truncating.")
        caption = caption[:297] + "..."

    # 2. Read Image
    image_path = "output.jpg"
    if not os.path.exists(image_path):
        print(f"❌ {image_path} not found")
        sys.exit(1)

    print(f"Logging into Bluesky as {handle}...")
    client = Client()
    try:
        client.login(handle, password)
        
        print(f"Uploading image {image_path}...")
        with open(image_path, 'rb') as f:
            img_data = f.read()
            
        upload = client.upload_blob(img_data)
        embed = models.AppBskyEmbedImages.Main(
            images=[models.AppBskyEmbedImages.Image(alt="Digital Guardian - Wilma", image=upload.blob)]
        )
        
        print("Creating post...")
        client.send_post(text=caption, embed=embed)
        print("✅ Successfully posted to Wilma's Bluesky!")
        update_state_after_post("bluesky", state_path="forwilma/state.json")
        
        # Success: Consume flag
        if os.path.exists(flag_path):
            os.remove(flag_path)
            print(f"✓ Flag {flag_path} consumed.")
        
    except Exception as e:
        print(f"❌ Failed to post to Bluesky: {e}")
        sys.exit(1)

if __name__ == "__main__":
    publish_wilma_to_bluesky()
