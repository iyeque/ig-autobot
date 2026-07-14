#!/usr/bin/env python3
import os
import sys
import json
import time
import requests
import base64
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to import shared_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_utils import (
    update_state_after_post,
    is_platform_posted,
    get_active_bundle,
    resolve_bundle_media,
)

# Load .env from project root if available
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from {dotenv_path}")

def check_url_live(url, max_retries=15, delay=20):
    """Checks if the URL is publicly accessible before proceeding."""
    print(f"Checking if {url} is live...")
    for i in range(max_retries):
        try:
            r = requests.head(url, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                print(f"✓ URL is live (Attempt {i+1})")
                return True
            if r.status_code == 404:
                # Some CDNs/Pages serve 404 on HEAD but 200 on GET
                g = requests.get(url, stream=True, timeout=10, allow_redirects=True)
                if g.status_code == 200:
                    g.close()
                    print(f"✓ URL is live via GET fallback (Attempt {i+1})")
                    return True
            print(f"Status {r.status_code} for {url}. Waiting {delay}s... (Attempt {i+1}/{max_retries})")
        except Exception as e:
            print(f"Error checking {url}: {e}. Waiting {delay}s... (Attempt {i+1}/{max_retries})")
        time.sleep(delay)
    return False

def wait_for_media(user_id, creation_id, access_token, max_checks=10, delay=10):
    """Waits for Instagram to finish processing the uploaded media."""
    url = f"https://graph.facebook.com/v18.0/{creation_id}"
    params = {"fields": "status_code", "access_token": access_token}
    
    for i in range(max_checks):
        r = requests.get(url, params=params)
        data = r.json()
        status = data.get("status_code", "UNKNOWN")
        print(f"  Media {creation_id} status: {status}")
        if status == "FINISHED":
            return True
        if status == "ERROR":
            print(f"❌ Instagram media processing error: {data}")
            return False
        time.sleep(delay)
    return False

def publish_single(user_id, image_path, caption, access_token):
    """Publishes a single image post by uploading the binary file directly."""
    local_path = image_path.replace("https://iyeque.github.io/ig-autobot/", "")
    print(f"Uploading image file: {local_path}")
    url = f"https://graph.facebook.com/v18.0/{user_id}/media"
    
    try:
        with open(local_path, "rb") as f:
            # DO NOT include image_url here
            payload = {
                "caption": caption,
                "access_token": access_token,
                "media_type": "IMAGE"
            }
            files = {"file": f}
            
            max_retries = 3
            for attempt in range(max_retries):
                r = requests.post(url, data=payload, files=files)
                res = r.json()
                creation_id = res.get("id")
                if creation_id:
                    if wait_for_media(user_id, creation_id, access_token):
                        return publish_container(user_id, creation_id, access_token)
                    return False
                error = res.get("error", {})
                print(f"❌ Attempt {attempt + 1} failed: {res}")
                if (error.get("is_transient") or error.get("code") in [1, 2, 20]) and attempt < max_retries - 1:
                    time.sleep(30)
                    continue
                break
    except Exception as e:
        print(f"❌ Failed to upload image: {e}")
        return False
    return False

def publish_story(user_id, image_url, access_token):
    """Publishes an image to Instagram Stories using URL-based upload."""
    print(f"Creating story container for {image_url}")
    url = f"https://graph.facebook.com/v18.0/{user_id}/media"
    payload = {
        "image_url": image_url,
        "media_type": "STORIES",
        "access_token": access_token
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        r = requests.post(url, data=payload)
        res = r.json()
        creation_id = res.get("id")
        if creation_id:
            if wait_for_media(user_id, creation_id, access_token):
                return publish_container(user_id, creation_id, access_token)
            return False
            
        error = res.get("error", {})
        print(f"❌ Story attempt {attempt + 1} failed: {res}")
        if (error.get("is_transient") or error.get("code") in [1, 2, 20]) and attempt < max_retries - 1:
            time.sleep(20)
            continue
        break
    return False

def publish_reel_with_name(user_id, video_url, caption, audio_name, access_token):
    """Publishes a Reel (video) post using URL-based upload."""
    print(f"Creating reel container for {video_url} with audio: {audio_name}")
    url = f"https://graph.facebook.com/v18.0/{user_id}/media"
    payload = {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
        "audio_name": audio_name,
        "access_token": access_token
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        r = requests.post(url, data=payload)
        res = r.json()
        creation_id = res.get("id")
        if creation_id:
            if wait_for_media(user_id, creation_id, access_token, max_checks=25, delay=15):
                return publish_container(user_id, creation_id, access_token)
            return False
            
        error = res.get("error", {})
        print(f"❌ Reel attempt {attempt + 1} failed: {res}")
        if (error.get("is_transient") or error.get("code") in [1, 2, 20]) and attempt < max_retries - 1:
            time.sleep(45)
            continue
        break
    return False

def publish_carousel(user_id, image_urls, caption, access_token):
    """Publishes a carousel post with retries for child items."""
    child_ids = []
    for url in image_urls:
        print(f"Creating child item for {url}")
        
        # Resolve local path for file upload fallback
        local_path = None
        if isinstance(url, str):
            if url.startswith("https://iyeque.github.io/ig-autobot/"):
                local_path = url.replace("https://iyeque.github.io/ig-autobot/", "", 1)
                local_path = local_path.replace("/", os.sep)
            elif url.startswith("./") or url.startswith(".\\"):
                local_path = url[2:]
        
        max_retries = 3
        cid = None
        print(f"DEBUG url={url} local_path={local_path} exists={os.path.exists(local_path) if local_path else False}")
        for attempt in range(max_retries):
            # Prefer local file upload when available (avoids GitHub Pages deploy dependency)
            if local_path and os.path.exists(local_path):
                try:
                    with open(local_path, "rb") as f:
                        res = requests.post(
                            f"https://graph.facebook.com/v18.0/{user_id}/media",
                            data={
                                "is_carousel_item": "true",
                                "access_token": access_token,
                            },
                            files={"file": f},
                    ).json()
                    print(f"DEBUG local upload result: {res}")
                    cid = res.get("id")
                    if cid:
                        break
                except Exception as e:
                    print(f"⚠ Local upload attempt {attempt + 1} failed: {e}")
            
            if not cid:
                res = requests.post(f"https://graph.facebook.com/v18.0/{user_id}/media", data={
                    "image_url": url,
                    "is_carousel_item": "true",
                    "access_token": access_token
                }).json()
                cid = res.get("id")
            
            if cid:
                break
            
            error = res.get("error", {})
            print(f"❌ Child item attempt {attempt + 1} failed: {res}")
            if (error.get("is_transient") or error.get("code") in [1, 2, 20]) and attempt < max_retries - 1:
                time.sleep(20)
                continue
            break
            
        if not cid:
            print(f"❌ Failed to create child after retries: {url}")
            return False
        child_ids.append(cid)

    print("Waiting for all child items to be ready...")
    for cid in child_ids:
        if not wait_for_media(user_id, cid, access_token):
            return False

    print("Creating carousel container...")
    payload = {
        "media_type": "CAROUSEL",
        "children": ",".join(child_ids),
        "caption": caption,
        "access_token": access_token
    }
    
    # Retry for carousel container
    max_retries = 3
    for attempt in range(max_retries):
        res = requests.post(f"https://graph.facebook.com/v18.0/{user_id}/media", data=payload).json()
        container_id = res.get("id")
        if container_id:
            print(f"Waiting for carousel container {container_id} to be ready...")
            if wait_for_media(user_id, container_id, access_token):
                return publish_container(user_id, container_id, access_token)
            return False
        
        error = res.get("error", {})
        print(f"❌ Carousel container attempt {attempt + 1} failed: {res}")
        if (error.get("is_transient") or error.get("code") in [1, 2, 20]) and attempt < max_retries - 1:
            time.sleep(30)
            continue
        break
        
    return False

def publish_container(user_id, creation_id, access_token):
    """Final step to publish any media container."""
    print(f"Publishing container {creation_id}...")
    url = f"https://graph.facebook.com/v18.0/{user_id}/media_publish"
    r = requests.post(url, data={
        "creation_id": creation_id,
        "access_token": access_token
    })
    res = r.json()
    if "id" in res:
        print(f"✓ Successfully published! Post ID: {res['id']}")
        update_state_after_post("instagram")
        return True
    print(f"❌ Publish failed: {res}")
    return False

def main():
    user_id = os.environ.get("IG_USER_ID")
    access_token = os.environ.get("IG_ACCESS_TOKEN")
    base_url = "https://iyeque.github.io/ig-autobot/" # Adjust if needed
    flag_path = "instagram_ready.flag"

    if not user_id or not access_token:
        print("❌ Missing IG_USER_ID or IG_ACCESS_TOKEN")
        sys.exit(1)

    if is_platform_posted("instagram"):
        print("⏭️ Instagram already posted for active bundle. Skipping.")
        return

    if not os.path.exists(flag_path):
        print("⏭️ Nothing new to post for Instagram. Skipping.")
        return

    active = get_active_bundle() or {}
    media = resolve_bundle_media(active, base_url=base_url)

    # Read caption
    caption = ""
    if os.path.exists("caption.txt"):
        with open("caption.txt", "r", encoding="utf-8") as f:
            caption = f.read()
    
    # Check for carousel / reel / single image
    image_urls = []
    reel_urls = []
    is_carousel = False
    is_reel = False
    audio_name = "Ambient Reflection"

    if os.path.exists("post_reel.flag"):
        if os.path.exists("post_reel.flag"):
            try:
                with open("post_reel.flag", "r", encoding="utf-8") as f:
                    flag_content = f.read().strip()
                    if flag_content and flag_content.lower() != "true":
                        audio_name = flag_content
            except Exception:
                pass

        if media.get("reel"):
            reel_urls = [media["reel"]]
            is_reel = True
        else:
            print("❌ Reel expected but no reel media found for active bundle.")
            sys.exit(1)

    if os.path.exists("carousel.json"):
        with open("carousel.json", "r", encoding="utf-8") as f:
            paths = json.load(f)
            image_urls = [base_url + p for p in paths]
            is_carousel = True
    elif not is_reel:
        if media.get("image_local") and os.path.exists(str(media["image_local"])):
            image_urls = [str(media["image_local"]).replace("\\", "/")]
        elif media.get("image"):
            image_urls = [media["image"]]
        else:
            print("❌ No image found for active bundle.")
            sys.exit(1)

    # Check if media URL is live
    if is_reel:
        checked = check_url_live(reel_urls[0])
        if not checked:
            fallbacks = []
            if active.get("reel"):
                fallbacks.append(base_url + active["reel"].replace("\\", "/"))
            for fb in fallbacks:
                print(f"Reel URL not accessible. Trying fallback: {fb}")
                if check_url_live(fb):
                    reel_urls[0] = fb
                    checked = True
                    break
            if not checked:
                print("❌ Reel URL not accessible. Aborting.")
                sys.exit(1)
    elif not is_carousel:
        # For single images, verify remote URL; carousel uploads handled per-item below
        checked = check_url_live(image_urls[0])
        if not checked:
            fallbacks = []
            if active.get("image"):
                fallbacks.append(base_url + active["image"].replace("\\", "/"))
            for fb in fallbacks:
                print(f"Image URL not accessible. Trying fallback: {fb}")
                if check_url_live(fb):
                    image_urls[0] = fb
                    checked = True
                    break
            if not checked:
                print("❌ Image URL not accessible. Aborting.")
                sys.exit(1)

    success = False
    if is_reel:
        # Pass the dynamic audio name to publish_reel
        print(f"Publishing Reel with audio name: {audio_name}")
        url = f"https://graph.facebook.com/v18.0/{user_id}/media"
        payload = {
            "media_type": "REELS",
            "video_url": reel_urls[0],
            "caption": caption,
            "audio_name": audio_name,
            "access_token": access_token
        }
        
        # We reuse the retry logic inside publish_reel but calling it directly here for the specific payload
        success = publish_reel_with_name(user_id, reel_urls[0], caption, audio_name, access_token)
    elif is_carousel:
        success = publish_carousel(user_id, image_urls, caption, access_token)
    else:
        success = publish_single(user_id, image_urls[0], caption, access_token)

    # Handle Story if flag exists
    story_success = False
    if os.path.exists("post_story.flag"):
        story_type = "post_amplifier"
        try:
            with open("post_story.flag", "r", encoding="utf-8") as f:
                story_type = (f.read().strip() or "post_amplifier")
        except Exception:
            pass

        print(f"Story flag detected (type={story_type}). Posting to Stories...")
        import glob
        if media.get("story"):
            story_url = media["story"]
        else:
            import glob
            story_files = sorted(glob.glob("images/story_*.jpg"), reverse=True)
            if story_files:
                story_url = base_url + story_files[0].replace('\\', '/')
            elif image_urls:
                story_url = image_urls[0] if image_urls[0].startswith("http") else base_url + image_urls[0]
            else:
                print("❌ No story image available for story fallback. Skipping story publish.")
                story_success = False
                story_url = ""

        if story_url and not check_url_live(story_url):
            print("❌ Story URL not accessible. Skipping story publish.")
            story_success = False
        elif story_url:
            story_success = publish_story(user_id, story_url, access_token)

        if story_success:
            print("✓ Story published.")
            os.remove("post_story.flag")

    if not success:
        sys.exit(1)

    if os.path.exists(flag_path):
        os.remove(flag_path)
        print(f"✓ Flag {flag_path} consumed.")

if __name__ == "__main__":
    main()
