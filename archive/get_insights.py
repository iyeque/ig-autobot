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
    print("\n✅ Successfully retrieved account insights for %s" % today)
    history = []
    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            try:
                history = json.load(f)
            except Exception:
                history = []

    # 1) Never drop the baseline — it is the historical reference anchor.
    baseline = [h for h in history if h.get("date") == "2026-04-ACCURATE-BASELINE"]
    # 2) Deduplicate on date so a re-run never appends a duplicate today's entry.
    rolling = [h for h in history if h.get("date") != "2026-04-ACCURATE-BASELINE"]
    seen = set()
    rolling = [h for h in rolling if not (h.get("date") in seen or seen.add(h.get("date")))]
    if len(baseline) > 1:
        print("⚠ More than one baseline entry found; keeping latest only.")
        baseline = baseline[:1]
    # 3) Baseline first, then newest entries, then today's (if new).
    history = baseline + rolling
    if new_entry["date"] not in [h.get("date") for h in history]:
        history.insert(0, new_entry)

    with open(DATA_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"💾 Metrics saved to {DATA_FILE}")

if __name__ == "__main__":
    get_account_insights()
