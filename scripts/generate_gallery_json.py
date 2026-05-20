import json
import os
import re

# Get all images and reels
image_dir = "images"
reel_dir = "reels"

def get_files(directory, extensions):
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(extensions)]

images = get_files(image_dir, ('.jpg', '.png', '.jpeg'))
reels = get_files(reel_dir, ('.mp4',))

# Combine and normalize paths for web
all_media = [m.replace('\\', '/') for m in images + reels]

# Prioritize keeping recent items, filter out temp files
curated = [m for m in all_media if ('post_' in m or 'reel_' in m or 'story_' in m or 'output' in m) and 'tmp_test' not in m]

def extract_timestamp(path):
    # Extracts YYYYMMDD_HHMMSS from the filename
    match = re.search(r'(\d{8}_\d{6})', path)
    return match.group(1) if match else "00000000_000000"

# Sort by timestamp (newest first)
# This ensures images and reels are mixed chronologically
curated.sort(key=extract_timestamp, reverse=True)

# Limit to top 24
gallery_data = curated[:24]

with open("gallery.json", "w") as f:
    json.dump(gallery_data, f, indent=2)

print(f"✓ Generated gallery.json with {len(gallery_data)} media items (Sorted chronologically).")
