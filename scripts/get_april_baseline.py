import os
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

# Confirmed credentials
ACCESS_TOKEN = "EAAZBDBV6rCu4BRVuVHPWZCvuwhT2r62IV2KSdwxaS5ArWkqDEuuminImdW3z5uw5dRGCsEEFNIJzdu9hE421rMNth8yDZBfNf3z9QvI1jMO0M9U3vD1rQ9UwzSztZA3TkQSzvJUILlRAc3L9KFMaE6pUrCnvprWMvqozl8zFfsUsIwmmMPewJmqcZBO7G9V4IHytSgPY1k7K8ct7c"
USER_ID = "17841457358687812"

def get_historical_insights():
    print(f"📊 Fetching Accurate April 2026 Baseline for {USER_ID}...")
    url = f"https://graph.facebook.com/v20.0/{USER_ID}/insights"
    
    totals = {"reach": 0, "views": 0, "follower_count": 0, "total_interactions": 0}
    
    # April 2026 Accurate Range (UTC)
    APR_START = 1775001600
    APR_END = 1777593600
    
    # 1. Reach
    print("Step 1: Fetching Reach...")
    params1 = {"metric": "reach", "period": "day", "metric_type": "total_value", "since": APR_START, "until": APR_END, "access_token": ACCESS_TOKEN}
    try:
        r1 = requests.get(url, params=params1).json()
        if "data" in r1: totals["reach"] = r1["data"][0]["total_value"]["value"]
    except: pass

    # 2. Views
    print("Step 2: Fetching Views...")
    params2 = {"metric": "views", "period": "day", "metric_type": "total_value", "since": APR_START, "until": APR_END, "access_token": ACCESS_TOKEN}
    try:
        r2 = requests.get(url, params=params2).json()
        if "data" in r2: totals["views"] = r2["data"][0]["total_value"]["value"]
    except: pass

    # 3. Follower Growth (30-day window)
    print("Step 3: Fetching Follower Growth (30-day window)...")
    THIRTY_DAYS_AGO = int((datetime.now() - timedelta(days=30)).timestamp())
    YESTERDAY = int((datetime.now() - timedelta(days=1)).timestamp())
    params3 = {"metric": "follower_count", "period": "day", "since": THIRTY_DAYS_AGO, "until": YESTERDAY, "access_token": ACCESS_TOKEN}
    try:
        r3 = requests.get(url, params=params3).json()
        if "data" in r3: totals["follower_count"] = sum(v["value"] for v in r3["data"][0]["values"])
    except: pass

    # 4. Total Interactions
    print("Step 4: Fetching Total Interactions...")
    params4 = {"metric": "total_interactions", "period": "day", "metric_type": "total_value", "since": APR_START, "until": APR_END, "access_token": ACCESS_TOKEN}
    try:
        r4 = requests.get(url, params=params4).json()
        if "data" in r4: totals["total_interactions"] = r4["data"][0]["total_value"]["value"]
    except: pass

    # Save
    DATA_FILE = Path("growth_data.json")
    history = []
    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f: history = json.load(f)
    
    history = [h for h in history if h.get("date") != "2026-04-ACCURATE-BASELINE"]
    history.insert(0, {
        "date": "2026-04-ACCURATE-BASELINE",
        "reach": totals["reach"],
        "views": totals["views"],
        "follower_growth_rolling_30d": totals["follower_count"],
        "interactions": totals["total_interactions"],
        "source": "Official Instagram Graph API (2026 Unified Metrics)"
    })
    
    with open(DATA_FILE, "w") as f: json.dump(history, f, indent=2)
    print(f"✅ Baseline saved to {DATA_FILE}")

if __name__ == "__main__":
    get_historical_insights()
