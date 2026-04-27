#!/usr/bin/env python3
import os
import sys
import requests
import json

LINKEDIN_ACCESS_TOKEN = os.environ.get("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_URN = os.environ.get("LINKEDIN_URN")

def test_text_post():
    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
        "LinkedIn-Version": "202510",
        "X-Restli-Protocol-Version": "2.0.0",
        "X-RestLi-Method": "create"
    }
    url = "https://api.linkedin.com/rest/posts"
    payload = {
        "author": LINKEDIN_URN,
        "commentary": "Testing automated LinkedIn post for Day 8. (Text only)",
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": []
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False
    }
    
    resp = requests.post(url, json=payload, headers=headers)
    print(f"Status: {resp.status_code}")
    print(resp.text)

if __name__ == "__main__":
    test_text_post()
