#!/usr/bin/env python3
"""Refresh the Threads long-lived access token and persist it to GitHub Secrets.

Threads tokens expire after 60 days and CANNOT be revived once expired.
This script is run weekly by threads_token_refresh.yml to keep the token fresh.

Requires:
  - THREADS_ACCESS_TOKEN: current (unexpired, ≥24h old) long-lived token
  - GH_PAT: GitHub PAT with repo secrets:write scope (to update the secret)
  - GITHUB_REPOSITORY: owner/repo (set automatically in Actions)

Usage:
  python scripts/refresh_threads_token.py          # refresh + update GitHub Secret
  python scripts/refresh_threads_token.py --local  # refresh + update .env only
"""
import os
import sys
import subprocess
import requests
from pathlib import Path
from dotenv import load_dotenv

dotenv_path = Path(__file__).parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)


def refresh_token(current_token: str) -> str | None:
    """Call Meta's th_refresh_token endpoint. Returns new token or None."""
    if not current_token or current_token.strip() in {"", "EXPIRED_ACCESS_TOKEN"}:
        print("❌ No valid THREADS_ACCESS_TOKEN to refresh.")
        return None
    url = "https://graph.threads.net/refresh_access_token"
    params = {"grant_type": "th_refresh_token", "access_token": current_token}
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            token = data.get("access_token")
            if token:
                days = data.get("expires_in", 0) // 86400
                print(f"✅ Refreshed Threads token (valid {days} days).")
                return token
        print(f"❌ Refresh failed: {r.status_code} {r.text[:300]}")
    except Exception as e:
        print(f"❌ Refresh error: {e}")
    return None


def update_github_secret(new_token: str) -> bool:
    """Write the new token to GitHub Secrets via gh CLI using GH_PAT."""
    repo = os.environ.get("GITHUB_REPOSITORY", "iyeque/ig-autobot")
    pat = os.environ.get("GH_PAT")
    if not pat:
        print("❌ GH_PAT not set — cannot update GitHub Secret.")
        return False
    env = os.environ.copy()
    env["GH_TOKEN"] = pat
    try:
        result = subprocess.run(
            ["gh", "secret", "set", "THREADS_ACCESS_TOKEN", "--body", new_token, "--repo", repo],
            env=env, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"✅ Updated GitHub Secret THREADS_ACCESS_TOKEN in {repo}.")
            return True
        print(f"❌ gh secret set failed: {result.stderr[:300]}")
    except Exception as e:
        print(f"❌ gh secret set error: {e}")
    return False


def update_local_env(new_token: str) -> bool:
    """Update THREADS_ACCESS_TOKEN in the local .env file."""
    if not dotenv_path.exists():
        return False
    lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("THREADS_ACCESS_TOKEN="):
            lines[i] = f"THREADS_ACCESS_TOKEN={new_token}"
            updated = True
            break
    if not updated:
        lines.append(f"THREADS_ACCESS_TOKEN={new_token}")
    dotenv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("✅ Updated local .env THREADS_ACCESS_TOKEN.")
    return True


def main():
    local_only = "--local" in sys.argv
    current = os.environ.get("THREADS_ACCESS_TOKEN", "")
    new_token = refresh_token(current)
    if not new_token:
        print("\n⚠ Token refresh failed. If the token is expired (>60 days),")
        print("  you must manually re-auth via the Threads OAuth flow:")
        print("  1. Get a new short-lived token from the browser OAuth flow")
        print("  2. Exchange it: GET graph.threads.net/oauth/access_token")
        print("     ?grant_type=th_exchange_token&client_secret=<APP_SECRET>")
        print("     &access_token=<SHORT_LIVED_TOKEN>")
        print("  3. Update THREADS_ACCESS_TOKEN in .env and GitHub Secrets")
        sys.exit(1)

    if local_only:
        update_local_env(new_token)
    else:
        ok = update_github_secret(new_token)
        update_local_env(new_token)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
