#!/usr/bin/env python3
"""Delete bundle 322's live posts — they shipped leaked placeholder garbage
("(Paste here)", "(Execute the final output...)", persona headers) to all six
platforms on Oct 9 2026 at 08:19 UTC (run 37904213653).

The state.json never recorded post IDs for 322 (posted_bundle_content['322']
has no IDs, and 322 isn't in platform_posted_bundles), so Instagram and
Threads IDs come from the publish run logs and the other four platforms are
found by listing the account's recent posts and matching today's date.

Usage:
    python scripts/delete_bundle_322.py --dry-run   # report what would be deleted
    python scripts/delete_bundle_322.py             # actually delete
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

# Load .env for local runs (CI injects env directly)
_env = os.path.join(REPO, ".env")
if os.path.exists(_env):
    for line in open(_env, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

TODAY = date.today().isoformat()

# Known from the publish run logs
IG_MEDIA_ID = "18109497476615325"
THREADS_MEDIA_ID = "18144164257598151"


def _ig_token() -> str:
    return os.environ.get("IG_ACCESS_TOKEN", "").strip()


def delete_instagram(media_id: str, dry: bool) -> bool:
    token = _ig_token()
    if not token:
        print("  instagram: no IG_ACCESS_TOKEN — skipping")
        return False
    if dry:
        print(f"  instagram: would DELETE media {media_id}")
        return True
    r = requests.delete(
        f"https://graph.instagram.com/v21.0/{media_id}",
        params={"access_token": token},
        timeout=30,
    )
    ok = r.status_code == 200 and r.json().get("success", True)
    print(f"  instagram: DELETE {media_id} -> {r.status_code} {r.text[:120]}")
    return ok


def delete_threads(media_id: str, dry: bool) -> bool:
    token = _ig_token()
    if not token:
        print("  threads: no IG_ACCESS_TOKEN (threads uses same token) — skipping")
        return False
    if dry:
        print(f"  threads: would DELETE media {media_id}")
        return True
    r = requests.delete(
        f"https://graph.threads.net/v1.0/{media_id}",
        params={"access_token": token},
        timeout=30,
    )
    ok = r.status_code == 200
    print(f"  threads: DELETE {media_id} -> {r.status_code} {r.text[:120]}")
    return ok


def delete_linkedin(dry: bool) -> bool:
    """List the author's posts, delete the one created today (bundle 322)."""
    try:
        from publish_linkedin import refresh_linkedin_access_token
    except Exception as e:
        print(f"  linkedin: import failed ({e}) — skipping")
        return False
    token = os.environ.get("LINKEDIN_ACCESS_TOKEN", "").strip()
    # The .env access token expires; the publisher refreshes it via the
    # refresh token. Do the same or every call 401s.
    fresh = refresh_linkedin_access_token(
        os.environ.get("LINKEDIN_REFRESH_TOKEN", "").strip(),
        os.environ.get("LINKEDIN_CLIENT_ID", "").strip(),
        os.environ.get("LINKEDIN_CLIENT_SECRET", "").strip(),
    )
    if fresh:
        token = fresh
    if not token:
        print("  linkedin: no usable token — skipping")
        return False
    author = os.environ.get("LINKEDIN_URN", "").strip() or os.environ.get("LINKEDIN_AUTHOR_URN", "").strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "LinkedIn-Version": "202604",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    # Try the posts API first (v2/posts), then ugcPosts as fallback
    candidates: list[str] = []
    if author:
        for url in (
            f"https://api.linkedin.com/v2/posts?q=author&author={author}&count=10",
            f"https://api.linkedin.com/v2/ugcPosts?q=authors&authors=List({author})&count=10",
        ):
            try:
                r = requests.get(url, headers=headers, timeout=30)
                if r.status_code == 200:
                    for el in r.json().get("elements", []):
                        urn = el.get("id") or ""
                        created = str(el.get("createdAt") or el.get("firstPublishedAt") or "")
                        # created is epoch ms
                        from datetime import datetime, timezone
                        try:
                            d = datetime.fromtimestamp(int(created) / 1000, tz=timezone.utc).date().isoformat()
                        except Exception:
                            d = ""
                        print(f"  linkedin: found post {urn[:50]} created {d}")
                        if d == TODAY and urn:
                            candidates.append(urn)
                    break
            except Exception as e:
                print(f"  linkedin: list failed ({e})")
                continue
    if not candidates:
        print("  linkedin: no post from today found via list — skipping")
        return False
    deleted = False
    for urn in candidates:
        if dry:
            print(f"  linkedin: would DELETE {urn}")
            deleted = True
            continue
        r = requests.delete(
            f"https://api.linkedin.com/rest/posts/{urn}",
            headers={**headers, "Content-Type": "application/json"},
            timeout=30,
        )
        print(f"  linkedin: DELETE {urn} -> {r.status_code} {r.text[:120]}")
        deleted = deleted or r.status_code in (200, 204)
    return deleted


def delete_youtube(dry: bool) -> bool:
    """List recent channel uploads, delete the one published today."""
    try:
        from publish_youtube import get_youtube_service
        yt = get_youtube_service()
    except Exception as e:
        print(f"  youtube: service init failed ({e}) — skipping")
        return False
    try:
        # Uploads playlist = UU + channel id
        ch = yt.channels().list(part="contentDetails", mine=True).execute()
        uploads = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        items = yt.playlistItems().list(
            part="snippet,contentDetails", playlistId=uploads, maxResults=5
        ).execute().get("items", [])
        from datetime import datetime, timezone
        for it in items:
            vid = it["contentDetails"]["videoId"]
            pub = it["snippet"].get("publishedAt", "")
            try:
                d = datetime.fromisoformat(pub.replace("Z", "+00:00")).astimezone(timezone.utc).date().isoformat()
            except Exception:
                d = ""
            title = it["snippet"].get("title", "")[:60]
            print(f"  youtube: found {vid} published {d} title={title!r}")
            if d == TODAY:
                if dry:
                    print(f"  youtube: would DELETE {vid}")
                    return True
                yt.videos().delete(id=vid).execute()
                print(f"  youtube: DELETE {vid} -> done")
                return True
    except Exception as e:
        print(f"  youtube: {e}")
        return False
    print("  youtube: no video from today found — skipping")
    return False


def delete_bluesky(dry: bool) -> bool:
    """List recent posts, delete the one from today."""
    try:
        from atproto import Client
        from atproto_client.models.com.atproto.repo.list_records import Params as ListParams
        handle = os.environ.get("BLUESKY_HANDLE", "").strip()
        password = os.environ.get("BLUESKY_PASSWORD", "").strip()
        if not handle or not password:
            print("  bluesky: no credentials — skipping")
            return False
        client = Client()
        client.login(handle, password)
        did = client.me.did
        from datetime import datetime, timezone
        from atproto_client.models.com.atproto.repo.delete_record import Data as DeleteData
        recs = client.com.atproto.repo.list_records(
            ListParams(repo=did, collection="app.bsky.feed.post", limit=10)
        ).records
        for rec in recs:
            rkey = rec.uri.split("/")[-1]
            created = getattr(rec.value, "created_at", "") or ""
            try:
                d = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(timezone.utc).date().isoformat()
            except Exception:
                d = ""
            text = (getattr(rec.value, "text", "") or "")[:50]
            print(f"  bluesky: found {rkey} created {d} text={text!r}")
            if d == TODAY:
                if dry:
                    print(f"  bluesky: would DELETE rkey {rkey}")
                    return True
                client.com.atproto.repo.delete_record(
                    DeleteData(repo=did, collection="app.bsky.feed.post", rkey=rkey)
                )
                print(f"  bluesky: DELETE {rkey} -> done")
                return True
    except Exception as e:
        print(f"  bluesky: {e}")
        return False
    print("  bluesky: no post from today found — skipping")
    return False


def delete_pinterest(dry: bool) -> bool:
    token = os.environ.get("PINTEREST_ACCESS_TOKEN", "").strip()
    if not token:
        print("  pinterest: no PINTEREST_ACCESS_TOKEN — skipping")
        return False
    try:
        r = requests.get(
            "https://api.pinterest.com/v5/pins",
            headers={"Authorization": f"Bearer {token}"},
            params={"page_size": 10},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"  pinterest: list failed {r.status_code} {r.text[:120]}")
            return False
        for pin in r.json().get("items", []):
            pid = pin.get("id", "")
            created = str(pin.get("created_at", ""))[:10]
            title = (pin.get("title") or pin.get("description") or "")[:50]
            print(f"  pinterest: found {pid} created {created} title={title!r}")
            if created == TODAY:
                if dry:
                    print(f"  pinterest: would DELETE {pid}")
                    return True
                d = requests.delete(
                    f"https://api.pinterest.com/v5/pins/{pid}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                )
                print(f"  pinterest: DELETE {pid} -> {d.status_code} {d.text[:120]}")
                return d.status_code in (200, 204)
    except Exception as e:
        print(f"  pinterest: {e}")
        return False
    print("  pinterest: no pin from today found — skipping")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--platforms",
        default="instagram,threads,linkedin,youtube,bluesky,pinterest",
        help="comma-separated platforms to attempt",
    )
    args = ap.parse_args()
    wanted = [p.strip() for p in args.platforms.split(",") if p.strip()]
    mode = "DRY RUN" if args.dry_run else "LIVE DELETE"
    print(f"=== Deleting bundle 322 posts ({mode}, date={TODAY}) ===\n")

    handlers = {
        "instagram": lambda d: delete_instagram(IG_MEDIA_ID, d),
        "threads": lambda d: delete_threads(THREADS_MEDIA_ID, d),
        "linkedin": delete_linkedin,
        "youtube": delete_youtube,
        "bluesky": delete_bluesky,
        "pinterest": delete_pinterest,
    }

    results = {}
    for plat in wanted:
        fn = handlers.get(plat)
        if fn is None:
            print(f"  {plat}: unknown platform — skipping")
            continue
        results[plat] = fn(args.dry_run)

    print("\n=== Summary ===")
    for plat, ok in results.items():
        print(f"  {plat:12} {'deleted' if ok else 'not deleted'}")
    n = sum(1 for v in results.values() if v)
    print(f"\n{n}/{len(results)} platforms handled")


if __name__ == "__main__":
    main()
