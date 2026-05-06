import os
import requests
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Configuration from environment
ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN")
USER_ID = os.environ.get("INSTAGRAM_USER_ID")


def validate_token():
    """Tests the validity of the access token by making a basic API call."""
    if not ACCESS_TOKEN or not USER_ID:
        print("❌ Error: Missing credentials!")
        return

    # A very basic API call that should work with any valid user token
    # Querying /me is a common way to test token validity.
    url = "https://graph.facebook.com/v20.0/me"
    params = {
        "access_token": ACCESS_TOKEN
    }
    
    print("Attempting to validate access token...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        
        if "id" in data:
            print("✅ Token is valid! Successfully retrieved User ID.")
            print(f"  User ID (from token): {data.get('id')}") # This is the FB User ID associated with the token
            return True
        else:
            print(f"❌ Token is valid but did not return expected data: {data}")
            return False

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        try:
            error_details = e.response.json()
            print(f"   API Error Details: {error_details}")
        except json.JSONDecodeError:
            print(f"   Response body: {e.response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Network or Request Error: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    validate_token()
