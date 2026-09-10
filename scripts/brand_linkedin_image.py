#!/usr/bin/env python3
"""LinkedIn image branding: logo watermark + centered text overlay -> output.jpg"""
import os, sys, json, shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
state_path = os.path.join(project_root, 'state.json')

with open(state_path) as f:
    state = json.load(f)

active = state.get('active_bundle', {})
if not active:
    sys.exit("No active_bundle")
image_rel = active.get('image')
if not image_rel:
    sys.exit("No image in active_bundle")

src = None
for c in [image_rel, os.path.join(project_root, image_rel)]:
    if os.path.exists(c):
        src = c; break
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
                    text = p['title']; break
if not text:
    text = (active.get('captions') or {}).get('linkedin', '').split('\n')[0].strip()
print(f"Overlay: {text or '(none)'}")

from PIL import Image, ImageDraw, ImageFont

def _font(sz):
    for p in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "arialbd.ttf",
        "Arial Bold.ttf",
    ]:
        try: return ImageFont.truetype(p, size=sz)
        except: pass
    return ImageFont.load_default()

def add_logo(img, lp):
    if not os.path.exists(lp):
        print(f"⚠ Logo missing: {lp}"); return
    logo = Image.open(lp).convert("RGBA")
    lw = 160; lh = int(logo.height * lw / logo.width)
    logo = logo.resize((lw, lh), Image.Resampling.LANCZOS)
    pad = 30; x = img.width - lw - pad; y = img.height - lh - pad
    img.paste(logo, (x, y), logo)
    print("✓ Logo applied")

def add_overlay(img, text):
    if not text: return
    draw = ImageDraw.Draw(img); w, h = img.size
    fs = 72 if len(text) < 25 else 58
    font = _font(fs)
    pad_x, pad_y = 60, 50
    max_tw = w - 160; max_th = h - 160
    words = text.strip().upper().split()
    lines, cur = [], ""
    for word in words:
        test = (cur + " " + word).strip()
        bb = draw.textbbox((0,0), test, font=font)
        if (bb[2]-bb[0]) > max_tw and cur: lines.append(cur); cur = word
        else: cur = test
    if cur: lines.append(cur)
    wrapped = "\n".join(lines)
    bb = draw.multiline_textbbox((0,0), wrapped, font=font, spacing=18, align="center")
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    while (tw > max_tw or th > max_th) and fs > 28:
        fs -= 4; font = _font(fs)
        lines, cur = [], ""
        for word in words:
            test = (cur + " " + word).strip()
            bb = draw.textbbox((0,0), test, font=font)
            if (bb[2]-bb[0]) > max_tw and cur: lines.append(cur); cur = word
            else: cur = test
        if cur: lines.append(cur)
        wrapped = "\n".join(lines)
        bb = draw.multiline_textbbox((0,0), wrapped, font=font, spacing=18, align="center")
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
    bw = min(tw + pad_x*2, w-40); bh = min(th + pad_y*2, h-40)
    bx = max(20, min(int((w-bw)//2), w-bw-20))
    by = max(20, min(int((h-bh)//2 - h*0.03), h-bh-20))
    layer = Image.new("RGBA", (w,h), (0,0,0,0))
    ld = ImageDraw.Draw(layer)
    ld.rectangle((bx, by, bx+bw, by+bh), fill=(0,0,0,155))
    img2 = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    draw2 = ImageDraw.Draw(img2)
    draw2.multiline_text(((w-tw)//2, by+pad_y), wrapped, font=font, fill=(255,255,255), spacing=18, align="center")
    img2.save(os.path.join(project_root, 'output.jpg'), format="JPEG", quality=95, optimize=True)
    print(f"✓ Overlay: {text}")
    # Replace img reference
    globals()['_last_img'] = img2

logo_path = os.path.join(project_root, 'wp logo.png')

img = Image.open(src).convert("RGB")
add_logo(img, logo_path)

_last_img = img
if text:
    # We need to re-open since add_logo modified in place but add_overlay creates new
    img2 = Image.open(src).convert("RGB")
    # Apply logo first on this copy too
    logo = Image.open(logo_path).convert("RGBA")
    lw = 160; lh = int(logo.height * lw / logo.width)
    logo = logo.resize((lw, lh), Image.Resampling.LANCZOS)
    pad = 30; x = img2.width - lw - pad; y = img2.height - lh - pad
    img2.paste(logo, (x, y), logo)
    # Now overlay
    draw = ImageDraw.Draw(img2); w, h = img2.size
    fs = 72 if len(text) < 25 else 58
    font = _font(fs)
    pad_x, pad_y = 60, 50
    max_tw = w - 160; max_th = h - 160
    words = text.strip().upper().split()
    lines, cur = [], ""
    for word in words:
        test = (cur + " " + word).strip()
        bb = draw.textbbox((0,0), test, font=font)
        if (bb[2]-bb[0]) > max_tw and cur: lines.append(cur); cur = word
        else: cur = test
    if cur: lines.append(cur)
    wrapped = "\n".join(lines)
    bb = draw.multiline_textbbox((0,0), wrapped, font=font, spacing=18, align="center")
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    while (tw > max_tw or th > max_th) and fs > 28:
        fs -= 4; font = _font(fs)
        lines, cur = [], ""
        for word in words:
            test = (cur + " " + word).strip()
            bb = draw.textbbox((0,0), test, font=font)
            if (bb[2]-bb[0]) > max_tw and cur: lines.append(cur); cur = word
            else: cur = test
        if cur: lines.append(cur)
        wrapped = "\n".join(lines)
        bb = draw.multiline_textbbox((0,0), wrapped, font=font, spacing=18, align="center")
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
    bw = min(tw + pad_x*2, w-40); bh = min(th + pad_y*2, h-40)
    bx = max(20, min(int((w-bw)//2), w-bw-20))
    by = max(20, min(int((h-bh)//2 - h*0.03), h-bh-20))
    layer = Image.new("RGBA", (w,h), (0,0,0,0))
    ld = ImageDraw.Draw(layer)
    ld.rectangle((bx, by, bx+bw, by+bh), fill=(0,0,0,155))
    img3 = Image.alpha_composite(img2.convert("RGBA"), layer).convert("RGB")
    draw3 = ImageDraw.Draw(img3)
    draw3.multiline_text(((w-tw)//2, by+pad_y), wrapped, font=font, fill=(255,255,255), spacing=18, align="center")
    img3.save(os.path.join(project_root, 'output.jpg'), format="JPEG", quality=95, optimize=True)
    print(f"✓ Overlay+Logo: {text}")

print(f"Done: {os.path.getsize(os.path.join(project_root, 'output.jpg'))} bytes")
