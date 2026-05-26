import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load env to get token
load_dotenv()
token = os.environ.get("PINTEREST_ACCESS_TOKEN")

if not token:
    print("❌ Error: PINTEREST_ACCESS_TOKEN not set.")
    exit()

headers = {"Authorization": f"Bearer {token}"}
url = "https://api-sandbox.pinterest.com/v5/boards"

print("Fetching your Sandbox boards...")
r = requests.get(url, headers=headers)
if r.status_code == 200:
    boards = r.json().get("items", [])
    for b in boards:
        print(f"Board Name: {b['name']} | ID: {b['id']}")
else:
    print(f"Error: {r.status_code} {r.text}")
