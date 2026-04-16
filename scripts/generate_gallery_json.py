import json
import os

# Get all images in the images folder
image_dir = "images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Filter for relevant images (exclude story artifacts if possible for a cleaner gallery)
all_files = os.listdir(image_dir)
images = [f"images/{f}" for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Priority: Prefer images starting with 'post_', 'output', or YYYYMMDD timestamps
# Avoid 'story_' prefix and small utility images
curated = [img for img in images if ('post_' in img or 'output' in img or (os.path.basename(img)[0].isdigit() and 'story_' not in img)) and 'tmp_test' not in img]
if curated:
    images = curated

# Sort by name (newest first based on timestamp naming convention)
images.sort(reverse=True)

# Limit to top 24 for performance
images = images[:24]

with open("gallery.json", "w") as f:
    json.dump(images, f)

print(f"✓ Generated gallery.json with {len(images)} images.")
