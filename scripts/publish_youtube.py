#!/usr/bin/env python3
import os
import sys
import time
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def get_youtube_service():
    client_id = os.environ.get("YOUTUBE_CLIENT_ID")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN")

    if not client_id or not client_secret or not refresh_token:
        print("❌ Missing YouTube OAuth credentials (ID, Secret, or Refresh Token)")
        sys.exit(1)

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/youtube.upload"]
    )

    if not creds.valid:
        print("Refreshing YouTube access token...")
        creds.refresh(Request())

    return build("youtube", "v3", credentials=creds)

def publish_to_youtube():
    video_path = "reel.mp4"
    if not os.path.exists(video_path):
        print(f"❌ {video_path} missing. YouTube Shorts require a video.")
        sys.exit(1)

    caption_path = "caption.txt"
    caption = "The Nine Stitches #TheNineStitches #Shorts"
    if os.path.exists(caption_path):
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

    # YouTube title limit is 100 chars. Use first line of caption.
    title = caption.split('\n')[0]
    if len(title) > 95:
        title = title[:92] + "..."
    if "#Shorts" not in title and len(title) < 90:
        title += " #Shorts"

    youtube = get_youtube_service()

    body = {
        "snippet": {
            "title": title,
            "description": caption,
            "tags": ["TheNineStitches", "Philosophy", "Shorts"],
            "categoryId": "22" 
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    
    print(f"Uploading {video_path} to YouTube Shorts...")
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")

    print(f"✅ Successfully uploaded to YouTube! Video ID: {response.get('id')}")

if __name__ == "__main__":
    publish_to_youtube()
