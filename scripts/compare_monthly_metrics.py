import os
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN") or os.environ.get("IG_ACCESS_TOKEN")
USER_ID = os.environ.get("INSTAGRAM_USER_ID") or os.environ.get("IG_USER_ID")
DATA_FILE = Path("growth_data.json")
IG_API_VERSION = "v20.0"
MAX_WINDOW_DAYS = 30  # IG Graph limit per request


def _chunked_dates(start: datetime, end: datetime):
    """Yield (chunk_start, chunk_end) tuples so no window exceeds MAX_WINDOW_DAYS."""
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=MAX_WINDOW_DAYS), end)
        yield cur, nxt
        cur = nxt


def fetch_monthly_totals(start_date, end_date):
    url = f"https://graph.facebook.com/{IG_API_VERSION}/{USER_ID}/insights"
    metrics = "reach,views,total_interactions"

    totals = {"reach": 0, "views": 0, "total_interactions": 0}
    try:
        for chunk_start, chunk_end in _chunked_dates(start_date, end_date):
            params = {
                "metric": metrics,
                "period": "day",
                "metric_type": "total_value",
                "since": int(chunk_start.timestamp()),
                "until": int(chunk_end.timestamp()),
                "access_token": ACCESS_TOKEN,
            }
            r = requests.get(url, params=params, timeout=60).json()
            if "data" in r:
                for item in r["data"]:
                    tv = item.get("total_value")
                    if isinstance(tv, dict):
                        totals[item["name"]] = totals.get(item["name"], 0) + tv.get("value", 0)
                    else:
                        vals = item.get("values", [])
                        totals[item["name"]] = totals.get(item["name"], 0) + sum(v.get("value", 0) for v in vals)
        return totals
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None


def load_history():
    if not DATA_FILE.exists():
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def find_baseline(history, label):
    return next((h for h in history if h.get("date") == label), None)


def compare():
    if not ACCESS_TOKEN or not USER_ID:
        print("❌ Error: Missing credentials")
        return

    history = load_history()
    april = find_baseline(history, "2026-04-ACCURATE-BASELINE")
    if not april:
        print("❌ April baseline not found in growth_data.json")
        return

    print("Fetching live metrics for May 2026...")
    may_now = fetch_monthly_totals(datetime(2026, 5, 1), datetime(2026, 6, 1))
    if not may_now:
        print("❌ Failed to retrieve May metrics.")
        return

    print("Fetching live metrics for June 2026...")
    jun_now = fetch_monthly_totals(datetime(2026, 6, 1), datetime(2026, 7, 1))
    if not jun_now:
        print("❌ Failed to retrieve June metrics.")
        return

    # Normalize keys so baseline and live use same names
    def clean(d):
        return {
            "reach": d.get("reach", 0) or d.get("total_value", {}).get("reach", 0),
            "views": d.get("views", 0) or d.get("total_value", {}).get("views", 0),
            "total_interactions": d.get("total_interactions", 0) or d.get("interactions", 0) or d.get("total_value", {}).get("total_interactions", 0),
        }

    april_n = clean(april)
    may_n = clean(may_now)
    jun_n = clean(jun_now)

    print("\n📊 --- MONTHLY PERFORMANCE COMPARISON ---")
    print(f"{'Metric':<16} {'April':>10} {'May':>10} {'June':>10} {'May vs Apr':>14} {'Jun vs May':>14}")
    print("-" * 76)

    metrics_map = {
        "reach": "REACH",
        "views": "VIEWS",
        "total_interactions": "INTERACTIONS",
    }

    for key, label in metrics_map.items():
        apr = april_n[key]
        may = may_n[key]
        jun = jun_n[key]

        def pct(new, old):
            return f"{(new - old) / old * 100:+.1f}%" if old else "n/a"

        print(f"{label:<16} {apr:>10,} {may:>10,} {jun:>10,} {pct(may, apr):>14} {pct(jun, may):>14}")

    print("-" * 76)

    # Engagement rates
    eng = lambda m, r: (m / r * 100) if r else 0
    apr_er = eng(april_n["total_interactions"], april_n["reach"])
    may_er = eng(may_n["total_interactions"], may_n["reach"])
    jun_er = eng(jun_n["total_interactions"], jun_n["reach"])

    print(f"\n* Engagement Rate: {apr_er:.2f}% (Apr) -> {may_er:.2f}% (May) -> {jun_er:.2f}% (Jun)")

    # Diagnostic flags
    if jun_n["reach"] < april_n["reach"] * 0.5:
        print("\n⚠ ALERT: June reach is less than 50% of April baseline.")
        print("  This suggests algorithm suppression or severe audience fatigue.")
    if jun_er < apr_er * 0.5:
        print("⚠ ALERT: Engagement rate halved since April.")
        print("  Content is no longer resonating with the audience being reached.")

    print("\n🚀 STRATEGIC ASSESSMENT:")
    if jun_n["reach"] < may_n["reach"]:
        print("* Reach continues to collapse May → June. Likely cause: posting cadence outpacing follower growth.")
    else:
        print("* Reach stabilized or improved. Maintain current cadence and optimize hooks.")

    if jun_er < may_er:
        print("* Engagement efficiency declining. Pivot pillar mix toward highest historical engagement topics.")
    else:
        print("* Engagement efficiency stable or improving. Double down on current pillar weights.")

    print("\n📋 NEXT STEPS:")
    print("1. Validate each platform attribution with UTM campaign params before any spend.")
    print("2. Create 3 test variants of highest-engagement pillar and run A/B via scheduled posts.")
    print("3. If June is broken, pause automated posting for 48h to avoid further algorithm penalty.")


if __name__ == "__main__":
    compare()
