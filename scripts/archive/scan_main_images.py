#!/usr/bin/env python3
"""Scan all post_* JPGs and PNGs for usable fallback bases.
Outputs text row counts for every file so we can assess each one."""
from PIL import Image
import numpy as np, glob, os

print("=== post_*.jpg (branded JPGs from main bot) ===")
jpg_data = []
for p in sorted(glob.glob('images/post_*.jpg')):
    try:
        img = Image.open(p)
        arr = np.array(img.convert('L'))
        row_std = arr.std(axis=1)
        text_count = int((row_std > 25).sum())
        jpg_data.append((p, img.size, text_count, os.path.getsize(p)))
    except Exception: pass

jpg_data.sort(key=lambda x: x[2])
print(f'Total: {len(jpg_data)} images')
print("Top 20 cleanest (lowest text row count):")
for p, sz, tr, szb in jpg_data[:20]:
    print(f'  {os.path.basename(p):45s} {sz}  {tr:3d} text rows  {szb//1024}KB')

print()
print("=== post_*.png (pre-branding raw outputs) ===")
png_data = []
for p in sorted(glob.glob('images/post_*.png')):
    try:
        img = Image.open(p)
        arr = np.array(img.convert('L'))
        row_std = arr.std(axis=1)
        text_count = int((row_std > 25).sum())
        png_data.append((p, img.size, text_count, os.path.getsize(p)))
    except Exception: pass

png_data.sort(key=lambda x: x[2])
print(f'Total: {len(png_data)} images')
print("Top 20 cleanest (lowest text row count):")
for p, sz, tr, szb in png_data[:20]:
    print(f'  {os.path.basename(p):45s} {sz}  {tr:3d} text rows  {szb//1024}KB')
