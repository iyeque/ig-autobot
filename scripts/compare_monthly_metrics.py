import os
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configuration from environment
ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN")
USER_ID = os.environ.get("INSTAGRAM_USER_ID")
DATA_FILE = Path("growth_data.json")

def fetch_monthly_totals(start_date, end_date):
    url = f"https://graph.facebook.com/v20.0/{USER_ID}/insights"
    metrics = "reach,views,total_interactions"
    params = {
        "metric": metrics,
        "period": "day",
        "metric_type": "total_value",
        "since": int(start_date.timestamp()),
        "until": int(end_date.timestamp()),
        "access_token": ACCESS_TOKEN
    }
    
    totals = {"reach": 0, "views": 0, "total_interactions": 0}
    try:
        r = requests.get(url, params=params).json()
        if "data" in r:
            for item in r["data"]:
                totals[item["name"]] = item.get("total_value", {}).get("value", 0)
        return totals
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None

def compare():
    if not ACCESS_TOKEN or not USER_ID:
        print("❌ Error: Missing credentials (INSTAGRAM_ACCESS_TOKEN / USER_ID)")
        return

    # 1. Load Last Month (April) Baseline
    if not DATA_FILE.exists():
        print("❌ growth_data.json missing. Cannot compare.")
        return
    
    with open(DATA_FILE, "r") as f:
        history = json.load(f)
    
    april = next((h for h in history if h.get("date") == "2026-04-ACCURATE-BASELINE"), None)
    if not april:
        print("❌ April baseline not found in growth_data.json")
        return

    # 2. Fetch This Month (May) so far
    now = datetime.now()
    may_start = datetime(2026, 5, 1)
    may_now = fetch_monthly_totals(may_start, now)
    
    if not may_now:
        return

    print("\n📊 --- MONTHLY PERFORMANCE COMPARISON ---")
    print(f"Comparing April 2026 (Full) vs May 2026 (to {now.strftime('%b %d')})")
    print("-" * 40)
    
    for metric in ["reach", "views", "interactions"]:
        apr_val = april.get(metric, 0)
        # Handle 'total_interactions' vs 'interactions' key naming
        may_val = may_now.get(metric if metric != "interactions" else "total_interactions", 0)
        
        diff = may_val - apr_val
        percent = (diff / apr_val * 100) if apr_val > 0 else 0
        
        status = "📈" if diff >= 0 else "📉"
        print(f"{metric.upper():12}: {apr_val:>6} (Apr) -> {may_val:>6} (May) | {status} {percent:>+6.1f}%")

    print("-" * 40)
    print("💡 NOTE: May is still in progress. The percentages will improve as the month continues.")

if __name__ == "__main__":
    compare()
