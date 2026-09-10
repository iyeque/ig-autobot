#!/usr/bin/env python3
"""Threads access token refresh / validity check.

Handles three scenarios:
1. refresh_token is set → attempt OAuth refresh, write new access_token to .env
2. refresh_token is empty BUT access_token still valid → keep using it, report OK
3. refresh_token is empty AND access_token expired → report that re-auth is required
"""
import os
import sys
import time
import requests
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from {dotenv_path}")

APP_ID = os.environ.get("THREADS_APP_ID", "").strip()
APP_SECRET = os.environ.get("THREADS_APP_SECRET", "").strip()
ACCESS_TOKEN = os.environ.get("THREADS_ACCESS_TOKEN", "").strip()
REFRESH_TOKEN = os.environ.get("THREADS_REFRESH_TOKEN", "").strip()

print(f"APP_ID:         {'set' if APP_ID else 'MISSING'}")
print(f"APP_SECRET:     {'set' if APP_SECRET else 'MISSING'}")
print(f"ACCESS_TOKEN:   {'set (' + str(len(ACCESS_TOKEN)) + ' chars)' if ACCESS_TOKEN else 'MISSING/empty'}")
print(f"REFRESH_TOKEN:  {'set (' + str(len(REFRESH_TOKEN)) + ' chars)' if REFRESH_TOKEN else 'empty'}")
print()

# ── Helper: check if a Threads access token is still valid ──────────────────
def is_token_valid(token: str) -> bool:
    """Ping the Threads API with the token; return True if it's still accepted."""
    if not token:
        return False
    uid = os.environ.get("THREADS_USER_ID", "").strip()
    if not uid:
        # No user id — just check that the token is parseable by Meta
        url = "https://graph.facebook.com/v19.0/debug_token"
        params = {"input_token": token, "access_token": f"{APP_ID}|{APP_SECRET}"}
    else:
        url = f"https://graph.threads.net/v19.0/{uid}"
        params = {"fields": "id", "access_token": token}
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        error = data.get("error")
        if error:
            code = error.get("code")
            msg = error.get("message", "")
            if code == 190 and ("expired" in msg.lower() or "expir" in msg.lower()):
                print(f"   Token check: expired ({msg[:80]}...)")
                return False
            print(f"   Token check: error code {code} — {msg[:80]}")
            return False
        return True
    except Exception as e:
        print(f"   Token check error: {e}")
        return False

# ── Helper: write a new value into .env ─────────────────────────────────────
def set_env_var(name: str, value: str):
    lines = dotenv_path.read_text().splitlines()
    new_lines = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{name}="):
            new_lines.append(f"{name}={value}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"{name}={value}")
    dotenv_path.write_text("\n".join(new_lines) + "\n")
    print(f"✅ Updated {dotenv_path} — {name} set")

# ── Scenario 1: refresh_token available → try OAuth refresh ────────────────
if REFRESH_TOKEN:
    print("📌 Scenario 1: refresh_token present — attempting OAuth refresh...")
    url = "https://graph.facebook.com/v19.0/oauth/access_token"
    payload = {
        "grant_type": "refresh_token",
        "client_id": APP_ID,
        "client_secret": APP_SECRET,
        "refresh_token": REFRESH_TOKEN,
    }
    r = requests.post(url, data=payload, timeout=30)
    print(f"   Refresh request: status {r.status_code}")
    data = r.json()
    print(f"   Response: {data}")

    if "access_token" in data:
        new_token = data["access_token"]
        print(f"\n✅ New access token obtained ({len(new_token)} chars)!")
        print(f"   Expires in: {data.get('expires_in', '?')}s")
        set_env_var("THREADS_ACCESS_TOKEN", new_token)

        # Rotate refresh token if Meta returned a new one
        if "refresh_token" in data and data["refresh_token"] != REFRESH_TOKEN:
            new_refresh = data["refresh_token"]
            print(f"   New refresh token available — saving it too")
            set_env_var("THREADS_REFRESH_TOKEN", new_refresh)
    else:
        print(f"\n❌ Refresh failed. Response: {data}")
        # In case Meta returned a new refresh_token without an access_token
        if "refresh_token" in data:
            new_refresh = data["refresh_token"]
            print(f"   New refresh token available: saving it")
            set_env_var("THREADS_REFRESH_TOKEN", new_refresh)
    sys.exit(0)

# ── Scenario 2: no refresh_token but access_token still valid ──────────────
print("📌 Scenario 2: no refresh_token — checking if current access_token still works...")
if ACCESS_TOKEN:
    if is_token_valid(ACCESS_TOKEN):
        print("\n✅ Current access_token is still valid — no refresh needed.")
        print("   publish_threads.py will use it as-is on next run.")
        sys.exit(0)
    else:
        print("\n❌ Current access_token is expired and no refresh_token is available.")
else:
    print("\n❌ No access_token set at all.")

# ── Scenario 3: nothing works → need re-auth ───────────────────────────────
print()
print("╔══════════════════════════════════════════════════════════════════╗")
print("║  RE-AUTHENTICATION REQUIRED                                      ║")
print("║                                                                  ║")
print("║  The Threads Meta app (ID: " + APP_ID + ") cannot obtain a new        ║")
print("║  access token because:                                          ║")
print("║                                                                  ║")
print("║  1. THREADS_REFRESH_TOKEN is empty (no refresh grant available) ║")
print("║  2. The current access_token is expired                         ║")
print("║  3. The app itself returns 'Cannot get application info' on     ║")
print("║     every API call — the app may be disabled or need re-        ║")
print("║     authorization in Meta Business Suite.                       ║")
print("║                                                                  ║")
print("║  To fix:                                                         ║")
print("║  1. Go to https://developers.facebook.com/apps/" + APP_ID + "/dashboard/║")
print("║  2. Re-authorize the app / reconnect Threads                    ║")
print("║  3. Generate a new long-lived access_token + refresh_token      ║")
print("║  4. Update .env with the new values                             ║")
print("╚══════════════════════════════════════════════════════════════════╝")
sys.exit(1)
