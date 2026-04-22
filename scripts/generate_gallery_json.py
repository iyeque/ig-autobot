import json
import os

# Get all images and reels
image_dir = "images"
reel_dir = "reels"

def get_files(directory, extensions):
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(extensions)]

images = get_files(image_dir, ('.jpg', '.png', '.jpeg'))
reels = get_files(reel_dir, ('.mp4',))

# Combine and curate
base_path = "/ig-autobot/"
all_media = [base_path + m for m in images + reels]
# Prioritize keeping recent items, filter out temp files
curated = [m for m in all_media if ('post_' in m or 'reel_' in m or 'output' in m) and 'tmp_test' not in m]

# Sort by name (newest first)
curated.sort(reverse=True)

# Limit to top 24
gallery_data = curated[:24]

with open("gallery.json", "w") as f:
    json.dump(gallery_data, f)

print(f"✓ Generated gallery.json with {len(gallery_data)} media items.")
