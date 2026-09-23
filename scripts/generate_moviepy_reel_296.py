#!/usr/bin/env python3
"""Generate MoviePy reel for bundle 296 (CI can't render HyperFrames)."""
import sys
sys.path.insert(0, '.')

from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load the image
img_path = 'images/post_20260922_072453.jpg'
img = Image.open(img_path).convert('RGB')
img = img.resize((1080, 1920), Image.LANCZOS)
img_arr = np.array(img)

# Reel duration
duration_s = 9

# Create base video clip
base = ImageClip(img_arr).set_duration(duration_s)

# Beat text overlays (from caption)
beats = [
    "THE FEEDBACK LOOP",
    "WE LIVE IN",
    "",
    "We are caught in loops",
    "we didn't design.",
    "",
    "Save this if you're ready",
    "to redesign your",
    "digital environment.",
]

# Generate text clips
text_clips = []
try:
    font = ImageFont.truetype("arial.ttf", 80)
except:
    font = ImageFont.load_default()

for i, text in enumerate(beats):
    if not text:
        continue
    # Create text image
    txt_img = Image.new('RGBA', (1080, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_img)
    # Get text bbox
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (1080 - tw) // 2
    y = (200 - th) // 2
    # Draw with outline
    draw.text((x-2, y-2), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x+2, y-2), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x-2, y+2), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    txt_arr = np.array(txt_img)
    # Show each beat for ~1.1s starting at staggered times
    start = 0.8 + i * 1.1
    if start + 1.0 > duration_s:
        break
    txt_clip = (ImageClip(txt_arr)
                .set_start(start)
                .set_duration(1.0)
                .set_position(('center', 'center')))
    text_clips.append(txt_clip)

# Compose
video = CompositeVideoClip([base] + text_clips, size=(1080, 1920))

# Write
output = 'reels/reel_20260922_072453.mp4'
import os
os.makedirs('reels', exist_ok=True)
video.write_videofile(output, fps=24, codec='libx264', audio=False, logger=None)
print(f"✅ Reel saved: {output}")
