#!/usr/bin/env python3
"""Fetch engagement metrics for published posts across platforms.

Collects metrics at 24h, 7d, and 30d intervals after posting.
Stores results in metrics.json (main) and forwilma/metrics.json (Wilma).

Usage:
    python scripts/fetch_metrics.py [--brand main|wilma|all]
    python scripts/fetch_metrics.py --brand main --interval 24h
"""
import argparse
import json
import os
import sys
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("WARNING: python-dotenv not installed; env vars must be set manually", file=sys.stderr)

REPO = Path(__file__).resolve().parents[1]
INTERVALS = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30)}


def load_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: Could not read {path}: {e}", file=sys.stderr)
            return {}
    return {}


def save_json(path, data):
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"ERROR: Could not write {path}: {e}", file=sys.stderr)


# ── Instagram ──────────────────────────────────────────────────────
def fetch_instagram_metrics(token, media_id):
    try:
        r = requests.get(
            f"https://graph.facebook.com/v18.0/{media_id}/insights",
            params={"metric": "reach,likes,comments,saved,shares,total_interactions,views", "access_token": token},
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"    IG request failed: {e}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(f"    HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return None
    try:
        data = r.json()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    JSON parse error: {e}", file=sys.stderr)
        return None
    out = {}
    for item in data.get("data", []):
        vals = item.get("values", [{}])
        out[item["name"]] = vals[0].get("value", 0) if vals else 0
    return out


# ── Threads ─────────────────────────────────────────────────────────
def fetch_threads_metrics(token, media_id):
    try:
        r = requests.get(
            f"https://graph.threads.net/v1.0/{media_id}/insights",
            params={"metric": "views,likes,replies,reposts,quotes,shares", "access_token": token},
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"    Threads request failed: {e}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(f"    HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return None
    try:
        data = r.json()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    JSON parse error: {e}", file=sys.stderr)
        return None
    out = {}
    for item in data.get("data", []):
        vals = item.get("values", [{}])
        out[item["name"]] = vals[0].get("value", 0) if vals else 0
    return out


# ── Bluesky ─────────────────────────────────────────────────────────
def create_bluesky_session(handle, password):
    """Create a single Bluesky session for reuse across multiple posts."""
    try:
        r = requests.post("https://bsky.social/xrpc/com.atproto.server.createSession",
                          json={"identifier": handle, "password": password}, timeout=30)
    except requests.RequestException as e:
        print(f"    Bluesky session failed: {e}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(f"    Bluesky session HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return None
    try:
        data = r.json()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    Bluesky session JSON parse error: {e}", file=sys.stderr)
        return None
    return data.get("accessJwt")


def fetch_bluesky_metrics(jwt, post_uri):
    try:
        r = requests.get("https://bsky.social/xrpc/app.bsky.feed.getPostThread",
                         params={"uri": post_uri}, headers={"Authorization": f"Bearer {jwt}"}, timeout=30)
    except requests.RequestException as e:
        print(f"    Bluesky request failed: {e}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(f"    Bluesky HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return None
    try:
        data = r.json()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    Bluesky JSON parse error: {e}", file=sys.stderr)
        return None
    post = data.get("thread", {}).get("post", {})
    return {
        "likes": post.get("likeCount", 0),
        "reposts": post.get("repostCount", 0),
        "replies": post.get("replyCount", 0),
    }


# ── Facebook ────────────────────────────────────────────────────────
def fetch_facebook_metrics(token, post_id):
    try:
        r = requests.get(
            f"https://graph.facebook.com/v18.0/{post_id}/insights",
            params={"metric": "post_impressions,post_engaged_users,post_clicks,post_reactions_by_type_total", "access_token": token},
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"    FB request failed: {e}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(f"    HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return None
    try:
        data = r.json()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    JSON parse error: {e}", file=sys.stderr)
        return None
    out = {}
    for item in data.get("data", []):
        vals = item.get("values", [{}])
        out[item["name"]] = vals[0].get("value", 0) if vals else 0
    return out


# ── Main collection logic ──────────────────────────────────────────
def collect_main(interval):
    state = load_json(REPO / "state.json")
    metrics = load_json(REPO / "metrics.json")
    pbc = state.get("posted_bundle_content", {})
    posted = state.get("platform_posted_bundles", {})

    ig_token = os.getenv("IG_ACCESS_TOKEN")
    fb_token = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN") or os.getenv("FB_PAGE_ACCESS_TOKEN")
    threads_token = os.getenv("THREADS_ACCESS_TOKEN")
    bsky_handle = os.getenv("BLUESKY_HANDLE")
    bsky_pass = os.getenv("BLUESKY_PASSWORD")
    bsky_jwt = None

    now = datetime.now(timezone.utc)

    for bundle_id, content in pbc.items():
        if bundle_id not in metrics:
            metrics[bundle_id] = {"topic": content.get("topic"), "pillar": content.get("pillar"),
                                  "image": content.get("image"), "posted_at": None, "snapshots": {}}

        # Determine post time from image filename or use 24h ago as fallback
        post_time_str = content.get("posted_at") or metrics[bundle_id].get("posted_at")
        if post_time_str:
            try:
                post_time = datetime.fromisoformat(post_time_str.replace("Z", "+00:00"))
            except Exception:
                post_time = now - timedelta(hours=25)
                metrics[bundle_id]["posted_at"] = post_time.isoformat()
        else:
            post_time = now - timedelta(hours=25)
            metrics[bundle_id]["posted_at"] = post_time.isoformat()

        age = now - post_time
        target_interval = INTERVALS.get(interval)
        if target_interval and age < target_interval:
            continue  # Too early for this interval

        snap_key = interval
        if snap_key in metrics[bundle_id].get("snapshots", {}):
            continue  # Already collected

        snap = {}
        # Instagram
        try:
            bid_int = int(bundle_id)
        except (ValueError, TypeError):
            continue
        if ig_token and "instagram" in posted and bid_int in posted.get("instagram", []):
            # Need media ID — store in content or look up
            media_id = content.get("ig_media_id")
            if media_id:
                m = fetch_instagram_metrics(ig_token, media_id)
                if m:
                    snap["instagram"] = m
        # Threads
        if threads_token and "threads" in posted and bid_int in posted.get("threads", []):
            media_id = content.get("threads_media_id")
            if media_id:
                m = fetch_threads_metrics(threads_token, media_id)
                if m:
                    snap["threads"] = m
        # Bluesky
        if bsky_handle and bsky_pass and "bluesky" in posted and bid_int in posted.get("bluesky", []):
            uri = content.get("bluesky_uri")
            if uri and (bsky_jwt is None or bsky_jwt == "RETRY"):
                bsky_jwt = create_bluesky_session(bsky_handle, bsky_pass)
                if bsky_jwt is None:
                    bsky_jwt = "RETRY"  # allow retry on next post
            if uri and bsky_jwt and bsky_jwt not in ("RETRY", "FAILED"):
                m = fetch_bluesky_metrics(bsky_jwt, uri)
                if m:
                    snap["bluesky"] = m
        # Facebook
        if fb_token and "facebook" in posted and bid_int in posted.get("facebook", []):
            post_id = content.get("facebook_post_id")
            if post_id:
                m = fetch_facebook_metrics(fb_token, post_id)
                if m:
                    snap["facebook"] = m

        if snap:
            metrics[bundle_id].setdefault("snapshots", {})[snap_key] = {
                "collected_at": now.isoformat(),
                "data": snap,
            }
            print(f"  Bundle {bundle_id}: collected {snap_key} snapshot ({list(snap.keys())})")

    save_json(REPO / "metrics.json", metrics)
    print(f"Saved metrics.json ({len(metrics)} bundles)")


def collect_wilma(interval):
    state = load_json(REPO / "forwilma" / "state.json")
    metrics = load_json(REPO / "forwilma" / "metrics.json")
    history = state.get("history", [])
    now = datetime.now(timezone.utc)

    bsky_handle = os.getenv("WILMA_BLUESKY_HANDLE")
    bsky_pass = os.getenv("WILMA_BLUESKY_PASSWORD")
    bsky_jwt = None

    for entry in history:
        day = entry.get("day")
        key = f"day_{day}"
        if key not in metrics:
            metrics[key] = {"day": day, "posted_at": entry.get("timestamp"), "snapshots": {}}

        post_time_str = entry.get("timestamp")
        if post_time_str:
            try:
                post_time = datetime.fromisoformat(post_time_str.replace("Z", "+00:00"))
            except Exception:
                post_time = now - timedelta(hours=25)
        else:
            post_time = now - timedelta(hours=25)

        age = now - post_time
        target_interval = INTERVALS.get(interval)
        if target_interval and age < target_interval:
            continue
        if interval in metrics[key].get("snapshots", {}):
            continue

        snap = {}
        uri = entry.get("bluesky_uri")
        if bsky_handle and bsky_pass and uri:
            if bsky_jwt is None or bsky_jwt == "RETRY":
                bsky_jwt = create_bluesky_session(bsky_handle, bsky_pass)
                if bsky_jwt is None:
                    bsky_jwt = "RETRY"
            if bsky_jwt and bsky_jwt not in ("RETRY", "FAILED"):
                m = fetch_bluesky_metrics(bsky_jwt, uri)
                if m:
                    snap["bluesky"] = m

        if snap:
            metrics[key].setdefault("snapshots", {})[interval] = {
                "collected_at": now.isoformat(),
                "data": snap,
            }
            print(f"  Wilma day {day}: collected {interval} snapshot")

    save_json(REPO / "forwilma" / "metrics.json", metrics)
    print(f"Saved forwilma/metrics.json ({len(metrics)} days)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default="all", choices=["main", "wilma", "all"])
    parser.add_argument("--interval", default="24h", choices=["24h", "7d", "30d"])
    args = parser.parse_args()

    print(f"Collecting {args.interval} metrics for {args.brand}...")
    if args.brand in ("main", "all"):
        collect_main(args.interval)
    if args.brand in ("wilma", "all"):
        collect_wilma(args.interval)


if __name__ == "__main__":
    main()
