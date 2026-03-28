#!/usr/bin/env python3
import os
import sys
import json
import time
import requests
import base64

def check_url_live(url, max_retries=15, delay=20):
    """Checks if the URL is publicly accessible before proceeding."""
    print(f"Checking if {url} is live...")
    for i in range(max_retries):
        try:
            r = requests.head(url, timeout=10)
            if r.status_code == 200:
                print(f"✓ URL is live (Attempt {i+1})")
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

def publish_single(user_id, image_url, caption, access_token):
    """Publishes a single image post."""
    print(f"Creating media container for {image_url}")
    url = f"https://graph.facebook.com/v18.0/{user_id}/media"
    payload = {
        "image_url": image_url,
        "caption": caption,
        "access_token": access_token
    }
    r = requests.post(url, data=payload)
    res = r.json()
    creation_id = res.get("id")
    if not creation_id:
        print(f"❌ Failed to create media container: {res}")
        return False

    if wait_for_media(user_id, creation_id, access_token):
        return publish_container(user_id, creation_id, access_token)
    return False

def publish_story(user_id, image_url, access_token):
    """Publishes an image to Instagram Stories."""
    print(f"Creating story container for {image_url}")
    url = f"https://graph.facebook.com/v18.0/{user_id}/media"
    payload = {
        "image_url": image_url,
        "media_type": "STORIES",
        "access_token": access_token
    }
    r = requests.post(url, data=payload)
    res = r.json()
    creation_id = res.get("id")
    if not creation_id:
        print(f"❌ Failed to create story container: {res}")
        return False

    if wait_for_media(user_id, creation_id, access_token):
        return publish_container(user_id, creation_id, access_token)
    return False

def publish_reel(user_id, video_url, caption, access_token):
    """Publishes a Reel (video) post."""
    print(f"Creating reel container for {video_url}")
    url = f"https://graph.facebook.com/v18.0/{user_id}/media"
    payload = {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
        "access_token": access_token
    }
    r = requests.post(url, data=payload)
    res = r.json()
    creation_id = res.get("id")
    if not creation_id:
        print(f"❌ Failed to create reel container: {res}")
        return False

    if wait_for_media(user_id, creation_id, access_token, max_checks=20, delay=12):
        return publish_container(user_id, creation_id, access_token)
    return False

def publish_carousel(user_id, image_urls, caption, access_token):
    """Publishes a carousel post."""
    child_ids = []
    for url in image_urls:
        print(f"Creating child item for {url}")
        res = requests.post(f"https://graph.facebook.com/v18.0/{user_id}/media", data={
            "image_url": url,
            "is_carousel_item": "true",
            "access_token": access_token
        }).json()
        cid = res.get("id")
        if not cid:
            print(f"❌ Failed to create child: {res}")
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
    res = requests.post(f"https://graph.facebook.com/v18.0/{user_id}/media", data=payload).json()
    container_id = res.get("id")
    if not container_id:
        print(f"❌ Failed to create carousel container: {res}")
        return False

    print(f"Waiting for carousel container {container_id} to be ready...")
    if wait_for_media(user_id, container_id, access_token):
        return publish_container(user_id, container_id, access_token)
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
        return True
    print(f"❌ Publish failed: {res}")
    return False

def main():
    user_id = os.environ.get("IG_USER_ID")
    access_token = os.environ.get("IG_ACCESS_TOKEN")
    base_url = "https://iyeque.github.io/ig-autobot/" # Adjust if needed
    
    if not user_id or not access_token:
        print("❌ Missing IG_USER_ID or IG_ACCESS_TOKEN")
        sys.exit(1)

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

    if os.path.exists("post_reel.flag"):
        base_reel_url = "https://iyeque.github.io/ig-autobot/reels/"
        import glob
        reel_files = sorted(glob.glob("reels/reel_*.mp4"), reverse=True)
        if reel_files:
            reel_urls = [base_reel_url + os.path.basename(reel_files[0])]
            is_reel = True
        elif os.path.exists("reel.mp4"):
            # Fallback when workflow has not moved the file yet
            reel_urls = ["https://iyeque.github.io/ig-autobot/reel.mp4"]
            is_reel = True
        else:
            print("❌ Reel flag found but no reel file discovered.")
            sys.exit(1)

    if os.path.exists("carousel.json"):
        with open("carousel.json", "r", encoding="utf-8") as f:
            paths = json.load(f)
            image_urls = [base_url + p for p in paths]
            is_carousel = True
    elif not is_reel:
        # Single image
        # Find latest post_*.jpg in images/
        import glob
        img_files = sorted(glob.glob("images/post_*.jpg"), reverse=True)
        if img_files:
            image_urls = [base_url + img_files[0].replace('\\', '/')]
        else:
            print("❌ No images found to post.")
            sys.exit(1)

    # Check if media URL is live
    if is_reel:
        if not check_url_live(reel_urls[0]):
            print("❌ Reel URL not accessible. Aborting.")
            sys.exit(1)
    else:
        if not check_url_live(image_urls[0]):
            print("❌ Image URL not accessible. Aborting.")
            sys.exit(1)

    success = False
    if is_reel:
        success = publish_reel(user_id, reel_urls[0], caption, access_token)
    elif is_carousel:
        success = publish_carousel(user_id, image_urls, caption, access_token)
    else:
        success = publish_single(user_id, image_urls[0], caption, access_token)

    # Handle Story if flag exists
    if os.path.exists("post_story.flag"):
        story_type = "post_amplifier"
        try:
            with open("post_story.flag", "r", encoding="utf-8") as f:
                story_type = (f.read().strip() or "post_amplifier")
        except Exception:
            pass

        print(f"Story flag detected (type={story_type}). Posting to Stories...")
        import glob
        story_files = sorted(glob.glob("images/story_*.jpg"), reverse=True)
        if story_files:
            story_url = base_url + story_files[0].replace('\\', '/')
        else:
            # Fallback to a known image source without assuming image_urls is populated
            post_files = sorted(glob.glob("images/post_*.jpg"), reverse=True)
            if post_files:
                story_url = base_url + post_files[0].replace('\\', '/')
            elif image_urls:
                story_url = image_urls[0]
            else:
                print("❌ No story or post image available for story fallback. Skipping story publish.")
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

if __name__ == "__main__":
    main()
