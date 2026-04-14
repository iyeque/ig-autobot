import json
import os

# Get all images in the images folder
image_dir = "images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

images = [f"images/{f}" for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Sort by name (newest first)
images.sort(reverse=True)

with open("gallery.json", "w") as f:
    json.dump(images, f)
