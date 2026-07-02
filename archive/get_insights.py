import os
import requests
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment
ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN") or os.environ.get("IG_ACCESS_TOKEN")
USER_ID = os.environ.get("INSTAGRAM_USER_ID") or os.environ.get("IG_USER_ID")

# Output directory for data
DATA_FILE = Path(__file__).parent.parent / "growth_data.json"

def get_account_insights():
    """Fetches daily insights (Unified 2026 standard) and appends to growth_data.json."""
    if not ACCESS_TOKEN or not USER_ID:
        print("❌ Error: Missing credentials!")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    new_entry = {"date": today}
    url = f"https://graph.facebook.com/v20.0/{USER_ID}/insights"
    
    # 1. Fetch Unified Metrics (requires total_value for most in 2026)
    # We include reach, views (unified), and total_interactions
    print(f"Fetching 2026 Daily Metrics for {today}...")
    params = {
        "metric": "reach,views,total_interactions",
        "period": "day",
        "metric_type": "total_value",
        "access_token": ACCESS_TOKEN
    }
    try:
        r1 = requests.get(url, params=params)
        d1 = r1.json()
        if "data" in d1:
            for item in d1["data"]:
                val = item.get("total_value", {}).get("value", 0)
                new_entry[item['name']] = val
                print(f"  {item['name'].upper()}: {val}")
    except Exception as e:
        print(f"Error fetching unified metrics: {e}")

    # 2. Fetch Time-Series Metrics (follower_count)
    print("Fetching Follower Growth...")
    params2 = {
        "metric": "follower_count",
        "period": "day",
        "access_token": ACCESS_TOKEN
    }
    try:
        r2 = requests.get(url, params=params2)
        d2 = r2.json()
        if "data" in d2:
            # For daily, we just take the first value in the series (the current day)
            val = d2["data"][0]["values"][0]["value"]
            new_entry["follower_count"] = val
            print(f"  FOLLOWER_COUNT: {val}")
    except Exception as e:
        print(f"Error fetching follower count: {e}")

    # Save to file
    if len(new_entry) > 1:
        print(f"\n✅ Successfully retrieved account insights for {today}")
        history = []
        if DATA_FILE.exists():
            with open(DATA_FILE, "r") as f:
                try: history = json.load(f)
                except: history = []
        
        # Keep baseline at the top, replace existing 'today' entry if it exists
        history = [h for h in history if h.get("date") != today]
        history.append(new_entry)
        
        # Sort to keep baseline first, then dates
        history.sort(key=lambda x: x['date'], reverse=True)
        
        with open(DATA_FILE, "w") as f:
            json.dump(history, f, indent=2)
        print(f"💾 Metrics saved to {DATA_FILE}")
    else:
        print("❌ No data retrieved.")

if __name__ == "__main__":
    get_account_insights()
