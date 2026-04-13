import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("LINKEDIN_ACCESS_TOKEN")

def get_linkedin_info():
    if not token or "your_linkedin_access_token" in token:
        print("❌ Error: LINKEDIN_ACCESS_TOKEN not found in .env")
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # 1. Try to get personal info via OpenID UserInfo (Modern way)
    print("--- Fetching Personal Profile (OpenID) ---")
    r_user = requests.get("https://api.linkedin.com/v2/userinfo", headers=headers)
    if r_user.status_code == 200:
        user = r_user.json()
        # The URN is 'sub' in OpenID response
        sub = user.get("sub")
        print(f"✅ Found Person: {user.get('given_name')} {user.get('family_name')}")
        print(f"📌 Personal URN: urn:li:person:{sub}")
    else:
        print(f"❌ Could not fetch userinfo: {r_user.status_code} {r_user.text}")

    # 2. Try to get organizations (requires w_organization_social/rw_organization_admin)
    print("\n--- Fetching Administered Organizations ---")
    # Adding Restli header for this specific call
    headers["X-Restli-Protocol-Version"] = "2.0.0"
    r_orgs = requests.get("https://api.linkedin.com/v2/organizationalEntityAcls?q=roleAssignee&role=ADMINISTRATOR", headers=headers)
    if r_orgs.status_code == 200:
        orgs = r_orgs.json()
        elements = orgs.get("elements", [])
        if not elements:
            print("ℹ️ No organizations found for this token.")
        for el in elements:
            org_urn = el.get("organizationalTarget")
            print(f"✅ Found Organization URN: {org_urn}")
    else:
        print(f"ℹ️ Organization check failed (likely missing permissions): {r_orgs.status_code}")

if __name__ == "__main__":
    get_linkedin_info()
