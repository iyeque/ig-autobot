#!/usr/bin/env python3
"""
Fast Reel Generator using ffmpeg-native filters (C-speed, no Python per frame).
"""
import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont

BRAND = "@theninestitches"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_REG = "C:/Windows/Fonts/arial.ttf"
FONT_SERIF = "C:/Windows/Fonts/georgia.ttf"
FONT_ITALIC = "C:/Windows/Fonts/georgiai.ttf"

FPS = 24
DURATION = 9
WIDTH, HEIGHT = 1080, 1920


def load_state(post_id=None):
    state_path = "state.json"
    if not os.path.exists(state_path):
        print("❌ state.json not found")
        sys.exit(1)
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    active = state.get("active_bundle", {})
    if not active and post_id is None:
        print("❌ No active bundle and no --post_id specified")
        sys.exit(1)
    if post_id:
        for b in state.get("content_queue", []):
            if str(b.get("post_id")) == str(post_id):
                return state, b
        active = {
            "post_id": post_id,
            "image": "images/post_20260922_072453.jpg",
            "master_reflection": "We live in systems designed to capture our attention and return us to the same loops.",
            "topic": f"Bundle #{post_id}",
            "captions": {"instagram": ""},
        }
    else:
        active = state["active_bundle"]
    return state, active


def render_text_png(text, output_path, fontsize=90, font_path=FONT_BOLD,
                    color="white", stroke_color="black", stroke_width=3,
                    size=(1000, 300)):
    """Render text to PNG file."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, fontsize)
    except (OSError, IOError):
        font = ImageFont.load_default()

    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= size[0] - 40:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    line_height = fontsize + 10
    total_h = len(lines) * line_height
    y_start = (size[1] - total_h) // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size[0] - tw) // 2
        y = y_start + i * line_height
        if stroke_width > 0:
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), line, font=font, fill=stroke_color)
        draw.text((x, y), line, font=font, fill=color)

    img.save(output_path)


def extract_hooks(bundle):
    text = bundle.get("master_reflection", "")
    topic = bundle.get("topic", "")
    hooks = []
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    for s in sentences[:8]:
        words = s.split()
        if len(words) > 5:
            mid = len(words) // 2
            hooks.append(" ".join(words[:mid]).upper())
            hooks.append(" ".join(words[mid:]).upper())
        else:
            hooks.append(s.upper())
    if topic:
        hooks.insert(0, topic.upper()[:30])
    while len(hooks) < int(DURATION / 1.2):
        hooks.extend(hooks[:2])
    return hooks[:int(DURATION / 1.2)]


def extract_quotes(bundle):
    text = bundle.get("master_reflection", "")
    quotes = []
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 8]
    for s in sentences:
        words = s.split()
        if len(words) > 12:
            mid = len(words) // 2
            quotes.append(" ".join(words[:mid]))
            quotes.append(" ".join(words[mid:]))
        else:
            quotes.append(s)
    while len(quotes) < 3:
        quotes.extend(quotes[:1])
    return quotes[:3]


def generate_hook_blast(image_path, output_path, hooks):
    """Generate Hook-Text Blast using ffmpeg filters."""
    print(f"  Generating Hook-Text Blast: {len(hooks)} hooks")
    
    tmpdir = tempfile.mkdtemp()
    try:
        hook_pngs = []
        for i, hook in enumerate(hooks):
            png_path = os.path.join(tmpdir, f"hook_{i}.png")
            render_text_png(hook, png_path, fontsize=90, font_path=FONT_BOLD)
            hook_pngs.append(png_path)
        
        brand_png = os.path.join(tmpdir, "brand.png")
        render_text_png(BRAND, brand_png, fontsize=28, font_path=FONT_REG, color="white", size=(300, 60))
        
        inputs = ["-loop", "1", "-t", str(DURATION), "-i", image_path]
        for png in hook_pngs:
            inputs.extend(["-loop", "1", "-t", str(DURATION), "-i", png])
        inputs.extend(["-loop", "1", "-t", str(DURATION), "-i", brand_png])
        
        filter_parts = []
        zoom_expr = "1+0.000694*n"
        filter_parts.append(
            f"[0:v]zoompan=z='{zoom_expr}':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}[bg]"
        )
        
        current = "[bg]"
        for i, hook in enumerate(hooks):
            start = i * 1.2
            end = start + 1.2
            filter_parts.append(
                f"[{i+1}]fade=t=in:st=0:d=0.25:alpha=1,fade=t=out:st=0.9:d=0.3:alpha=1[hook{i}]"
            )
            filter_parts.append(
                f"{current}[hook{i}]overlay=enable='between(t\\,{start}\\,{end})':x=(W-w)/2:y=(H-h)/2[v{i}]"
            )
            current = f"[v{i}]"
        
        hook_count = len(hooks)
        brand_idx = hook_count + 1
        filter_parts.append(
            f"[{brand_idx}]format=rgba,colorchannelmixer=aa=0.7[brand]"
        )
        filter_parts.append(
            f"{current}[brand]overlay=x=30:y=H-h-30[out]"
        )
        
        filter_str = ";".join(filter_parts)
        print(f"    Filter: {filter_str[:200]}...")
        
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_str,
            "-map", "[out]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "fast",
            "-crf", "23",
            "-an",
            output_path
        ]
        
        print(f"    Running ffmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ❌ ffmpeg error: {result.stderr[-500:]}")
            return False
        
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def generate_cinematic_quote(image_path, output_path, quotes):
    """Generate Cinematic Quote using ffmpeg filters."""
    print(f"  Generating Cinematic Quote: {len(quotes)} slides")
    
    tmpdir = tempfile.mkdtemp()
    try:
        quote_pngs = []
        for i, quote in enumerate(quotes):
            png_path = os.path.join(tmpdir, f"quote_{i}.png")
            render_text_png(f'"{quote}"', png_path, fontsize=60, font_path=FONT_SERIF,
                          color="white", stroke_color="#1a1a2e", stroke_width=2, size=(950, 400))
            quote_pngs.append(png_path)
        
        dash_png = os.path.join(tmpdir, "dash.png")
        render_text_png("—", dash_png, fontsize=50, font_path=FONT_ITALIC, color="#c8c8c8", size=(200, 80))
        
        brand_png = os.path.join(tmpdir, "brand.png")
        render_text_png(BRAND, brand_png, fontsize=28, font_path=FONT_SERIF, color="white", size=(300, 60))
        
        inputs = ["-loop", "1", "-t", str(DURATION), "-i", image_path]
        for png in quote_pngs:
            inputs.extend(["-loop", "1", "-t", str(DURATION), "-i", png])
        inputs.extend(["-loop", "1", "-t", str(DURATION), "-i", dash_png])
        inputs.extend(["-loop", "1", "-t", str(DURATION), "-i", brand_png])
        
        filter_parts = []
        zoom_expr = "1+0.00037*n"
        filter_parts.append(
            f"[0:v]zoompan=z='{zoom_expr}':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}[bg]"
        )
        
        filter_parts.append(
            f"[bg]drawbox=x=0:y=0:w={WIDTH}:h={HEIGHT}:color=black@0.3:t=fill[bg_dark]"
        )
        
        current = "[bg_dark]"
        for i, quote in enumerate(quotes):
            start = i * 3.0
            end = start + 3.0
            filter_parts.append(
                f"[{i+1}]fade=t=in:st=0:d=0.6:alpha=1,fade=t=out:st=2.4:d=0.6:alpha=1[quote{i}]"
            )
            filter_parts.append(
                f"{current}[quote{i}]overlay=enable='between(t\\,{start}\\,{end})':x=(W-w)/2:y=(H-h)/2[v{i}]"
            )
            current = f"[v{i}]"
        
        quote_count = len(quotes)
        dash_idx = quote_count + 1
        brand_idx = quote_count + 2
        
        for i in range(quote_count):
            start = i * 3.0
            filter_parts.append(
                f"[{dash_idx}]fade=t=in:st=0:d=0.6:alpha=1,fade=t=out:st=2.4:d=0.6:alpha=1[dash{i}]"
            )
            filter_parts.append(
                f"{current}[dash{i}]overlay=enable='between(t\\,{start}\\,{start+3})':x=(W-w)/2:y=H-250[v_d{i}]"
            )
            current = f"[v_d{i}]"
        
        filter_parts.append(
            f"[{brand_idx}]format=rgba,colorchannelmixer=aa=0.7[brand]"
        )
        filter_parts.append(
            f"{current}[brand]overlay=enable='gte(t\\,6)':x=(W-w)/2:y=H-80[out]"
        )
        
        filter_str = ";".join(filter_parts)
        print(f"    Filter: {filter_str[:200]}...")
        
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_str,
            "-map", "[out]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "fast",
            "-crf", "23",
            "-an",
            output_path
        ]
        
        print(f"    Running ffmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ❌ ffmpeg error: {result.stderr[-500:]}")
            return False
        
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Generate reels using ffmpeg-native filters")
    parser.add_argument("--template", choices=["hook_blast", "cinematic_quote"], default="hook_blast")
    parser.add_argument("--post_id", help="Bundle post_id")
    parser.add_argument("--output", help="Output path")
    args = parser.parse_args()
    
    state, bundle = load_state(args.post_id)
    post_id = bundle.get("post_id", "unknown")
    print(f"🎬 Generating {args.template} reel for bundle {post_id}")
    
    image_path = bundle.get("image", "images/post_20260922_072453.jpg")
    if not os.path.exists(image_path):
        image_path = "images/post_20260922_072453.jpg"
    
    if args.output:
        output_path = args.output
    else:
        os.makedirs("reels", exist_ok=True)
        output_path = f"reels/reel_{post_id}_{args.template}.mp4"
    
    if args.template == "hook_blast":
        hooks = extract_hooks(bundle)
        success = generate_hook_blast(image_path, output_path, hooks)
    elif args.template == "cinematic_quote":
        quotes = extract_quotes(bundle)
        success = generate_cinematic_quote(image_path, output_path, quotes)
    
    if success:
        file_size = os.path.getsize(output_path)
        print(f"✅ Done! Size: {file_size / 1024:.0f} KB")
    else:
        print("❌ Failed")


if __name__ == "__main__":
    main()
