import os
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configuration from environment
ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN") or os.environ.get("IG_ACCESS_TOKEN")
USER_ID = os.environ.get("INSTAGRAM_USER_ID") or os.environ.get("IG_USER_ID")
DATA_FILE = Path("growth_data.json")

def fetch_monthly_totals(start_date, end_date):
    url = f"https://graph.facebook.com/v20.0/{USER_ID}/insights"
    metrics = "reach,views,total_interactions"
    
    # We fetch daily values for the entire range
    params = {
        "metric": metrics,
        "period": "day",
        "since": int(start_date.timestamp()),
        "until": int(end_date.timestamp()),
        "access_token": ACCESS_TOKEN
    }
    
    totals = {"reach": 0, "views": 0, "total_interactions": 0}
    try:
        r = requests.get(url, params=params).json()
        if "data" in r:
            for item in r["data"]:
                metric_name = item["name"]
                values = item.get("values", [])
                sum_val = sum(v.get("value", 0) for v in values)
                totals[metric_name] = sum_val
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

    # 2. Fetch May 2026 Metrics (May 1st to June 1st)
    may_start = datetime(2026, 5, 1)
    may_end = datetime(2026, 6, 1)
    
    print(f"Fetching live metrics for May 2026...")
    may_now = fetch_monthly_totals(may_start, may_end)
    
    if not may_now:
        print("❌ Failed to retrieve May metrics.")
        return

    print("\n📊 --- MONTHLY PERFORMANCE COMPARISON ---")
    print(f"Comparing April 2026 (Full) vs May 2026 (Full)")
    print("-" * 50)
    
    metrics_map = {
        "reach": "REACH",
        "views": "VIEWS",
        "total_interactions": "INTERACTIONS"
    }
    
    for api_key, display_name in metrics_map.items():
        apr_val = april.get("reach" if api_key == "reach" else ("views" if api_key == "views" else "interactions"), 0)
        may_val = may_now.get(api_key, 0)
        
        diff = may_val - apr_val
        percent = (diff / apr_val * 100) if apr_val > 0 else 0
        
        status = "📈" if diff >= 0 else "📉"
        print(f"{display_name:12}: {apr_val:>6} (Apr) -> {may_val:>6} (May) | {status} {percent:>+7.1f}%")

    print("-" * 50)
    
    # 💡 Strategic Assessment
    print("\n💡 STRATEGIC INSIGHTS FOR MAY:")
    
    # Check interaction rate
    reach_may = may_now.get("reach", 0)
    inter_may = may_now.get("total_interactions", 0)
    rate_may = (inter_may / reach_may * 100) if reach_may > 0 else 0
    
    reach_apr = april.get("reach", 0)
    inter_apr = april.get("interactions", 0)
    rate_apr = (inter_apr / reach_apr * 100) if reach_apr > 0 else 0
    
    print(f"* Engagement Rate: {rate_apr:.2f}% (Apr) -> {rate_may:.2f}% (May)")
    
    if reach_may > reach_apr and inter_may < inter_apr:
        print("* WARNING: 'Volume Throttling' detected. Reach is up but engagement quality is down.")
        print("  Our 4x-6x daily posting strategy is driving impressions but potentially fatiguing followers.")
    elif rate_may < rate_apr * 0.8:
        print("* INSIGHT: Engagement rate has dropped significantly. The 2026 algorithm update prioritizing 'Shares' requires more controversial or deeply relatable hooks.")
    else:
        print("* POSITIVE: The strategy is holding steady. Your 'Professional Failure Expert' persona is resonating.")

    print("\n🚀 RECOMMENDATIONS FOR JUNE:")
    print("1. Reduce frequency to 3x daily (Peak GST) to avoid 'Low-Value Volume' flags.")
    print("2. Focus on 'Pattern Interrupts' at 3s in Reels to boost completion rates.")
    print("3. Use the new AI-Editor to ensure every hook is high-impact and concisely fits SEO needs.")

if __name__ == "__main__":
    compare()
