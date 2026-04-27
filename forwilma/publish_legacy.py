#!/usr/bin/env python3
import os
import sys
import requests
import json

LINKEDIN_ACCESS_TOKEN = os.environ.get("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_URN = os.environ.get("LINKEDIN_URN")

def publish_legacy_share():
    if not LINKEDIN_ACCESS_TOKEN or not LINKEDIN_URN:
        print("❌ Error: LINKEDIN_ACCESS_TOKEN or LINKEDIN_URN missing.")
        sys.exit(1)

    print(f"Attempting Legacy Share as: {LINKEDIN_URN}")

    # For legacy shares, we don't use the LinkedIn-Version header
    url = "https://api.linkedin.com/v2/shares"
    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }

    payload = {
        "owner": LINKEDIN_URN,
        "text": {
            "text": "Day 8: Setting Screen Time in 5 Minutes\n\nSetting boundaries in a digital world doesn't have to be a battle. Check out my latest tips for parents!\n\n#Parenting #DigitalWellbeing #ScreenTime"
        },
        "distribution": {
            "linkedInDistributionMode": "MAIN_FEED"
        }
    }

    resp = requests.post(url, json=payload, headers=headers)
    print(f"Status: {resp.status_code}")
    print(resp.text)

if __name__ == "__main__":
    publish_legacy_share()
