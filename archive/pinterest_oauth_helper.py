import os
import base64
import requests
import json
from pathlib import Path

# Try to load .env file
try:
    from dotenv import load_dotenv
    dotenv_path = Path(__file__).parent.parent / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded credentials from {dotenv_path}")
except ImportError:
    pass

# Ensure these are set in your .env or environment
APP_ID = os.environ.get("PINTEREST_APP_ID")
APP_SECRET = os.environ.get("PINTEREST_APP_SECRET")
# Default to localhost for recording but allow override from .env
REDIRECT_URI = os.environ.get("PINTEREST_REDIRECT_URI", "http://localhost:8085/")

def generate_auth_url():
    # Scopes requested by Eloise
    scopes = "boards:read,boards:write,pins:read,pins:write"
    url = (
        f"https://www.pinterest.com/oauth/?"
        f"client_id={APP_ID}&"
        f"redirect_uri={REDIRECT_URI}&"
        f"response_type=code&"
        f"scope={scopes}"
    )
    print("\n--- STEP 1: AUTHORIZATION URL ---")
    print("Copy and paste this URL into your browser (or it may open automatically):")
    print(f"\n{url}\n")
    return url

def exchange_code(code):
    print("\n--- STEP 2: EXCHANGING CODE FOR TOKEN ---")
    
    # Eloise's requirement: Base64-encoded string of App_ID:App_Secret
    auth_str = f"{APP_ID}:{APP_SECRET}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    
    url = "https://api.pinterest.com/v5/oauth/token"
    
    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "grant_type": "authorization_code",
        "code": code.strip(),
        "redirect_uri": REDIRECT_URI
    }
    
    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        print("✅ SUCCESS! Token received.")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ FAILED: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    import json
    if not APP_ID or not APP_SECRET:
        print("❌ Error: PINTEREST_APP_ID and PINTEREST_APP_SECRET must be set.")
    else:
        generate_auth_url()
        code = input("Enter the 'code' from the URL after you are redirected: ")
        if code:
            exchange_code(code)
