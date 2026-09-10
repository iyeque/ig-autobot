#!/usr/bin/env python3
"""
LinkedIn image branding step.
Uses bot.py's actual add_static_text_overlay + apply_logo_watermark.
Reads active bundle image, applies branding, writes output.jpg.
"""
import os
import sys
import json
import shutil

# Prevent .env double-load noise
os.environ.pop('LOADED_ENV', None)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
state_path = os.path.join(project_root, 'state.json')

with open(state_path) as f:
    state = json.load(f)

active = state.get('active_bundle', {})
if not active:
    sys.exit("No active_bundle in state.json")

image_rel = active.get('image')
if not image_rel:
    sys.exit("No image in active_bundle")

src = None
for c in [image_rel, os.path.join(project_root, image_rel)]:
    if os.path.exists(c):
        src = c
        break
if not src:
    sys.exit(f"Source not found: {image_rel}")

print(f"Source: {src} ({os.path.getsize(src)} bytes)")

# Overlay text from post title
post_id = active.get('post_id')
text = ""
if post_id:
    pp = os.path.join(project_root, 'posts.json')
    if os.path.exists(pp):
        with open(pp) as f:
            for p in json.load(f).get('posts', []):
                if p.get('id') == post_id and p.get('title'):
                    text = p['title']
                    break
if not text:
    text = (active.get('captions') or {}).get('linkedin', '').split('\n')[0].strip()
print(f"Overlay text: {text or '(none)'}")

# --- Import bot.py functions (won't trigger main()) ---
sys.path.insert(0, project_root)
import importlib.util
spec = importlib.util.spec_from_file_location('bot', os.path.join(project_root, 'bot.py'))
bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot)

add_static_text_overlay = bot.add_static_text_overlay
apply_logo_watermark = bot.apply_logo_watermark

# --- Copy source to output.jpg, then brand the copy ---
out_path = os.path.join(project_root, 'output.jpg')
shutil.copy(src, out_path)
print(f"Copied source to output.jpg ({os.path.getsize(out_path)} bytes)")

# Apply branding to output.jpg (bot.py functions modify in-place)
add_static_text_overlay(out_path, text)
print(f"✓ Text overlay applied: '{text}' (bot.py)")

apply_logo_watermark(out_path)
print("✓ Logo watermark applied (bot.py)")

size = os.path.getsize(out_path)
print(f"✓ output.jpg final: {size} bytes")
print("Done.")
