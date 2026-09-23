#!/usr/bin/env python3
"""
MoviePy Reel Template 2: Cinematic Quote (Slow Elegant Reveal)

One striking quote per slide, 3s each.
Fade-in text with stroke outline. Background gently pans/zooms.
3 slides, clean transitions. Elegant serif font.

Text rendered with PIL (no ImageMagick dependency).

Usage:
    python scripts/generate_reel_cinematic_quote.py --post_id 296
"""
import os
import sys
import json
import argparse
import numpy as np
from moviepy.editor import ImageClip, CompositeVideoClip, transfx
from PIL import Image, ImageDraw, ImageFont

# ─── CONFIGURATION ───────────────────────────────────────────────
BRAND = "@theninestitches"
FONT_BOLD = "C:/Windows/Fonts/georgiab.ttf"
FONT_REG = "C:/Windows/Fonts/georgia.ttf"
FONT_ITALIC = "C:/Windows/Fonts/georgiai.ttf"

# Animation
SLIDE_DURATION = 3.0
FADE_DURATION = 0.6
FPS = 24
DURATION = 9
SIZE = (1080, 1920)


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


def extract_quotes(bundle):
    """Extract 3 cinematic quotes from master_reflection."""
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


def render_text_pil(text, size=(1000, 400), fontsize=65, font_path=FONT_REG,
                    color=(255, 255, 255, 255), stroke_color=(26, 26, 46, 255),
                    stroke_width=2, align="center"):
    """Render text with PIL and return as RGBA numpy array."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, fontsize)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Word wrap
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= size[0] - 60:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    line_height = fontsize + 12
    total_h = len(lines) * line_height
    y_start = (size[1] - total_h) // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if align == "center":
            x = (size[0] - tw) // 2
        else:
            x = 30
        y = y_start + i * line_height
        # Stroke
        if stroke_width > 0:
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), line, font=font, fill=stroke_color)
        # Main text
        draw.text((x, y), line, font=font, fill=color)

    return np.array(img)


def create_pan_clip(image_path, duration, start_pos=(0.02, 0.02), end_pos=(0.08, 0.08), zoom=1.1):
    """Apply gentle pan (Ken Burns) to background image."""
    if not os.path.exists(image_path):
        bg = ImageClip(np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8))
        return bg.set_duration(duration)

    img = Image.open(image_path).convert("RGB")
    img = img.resize((SIZE[0], SIZE[1]), Image.LANCZOS)
    img_arr = np.array(img)
    base = ImageClip(img_arr).set_duration(duration)

    def pan_effect(get_frame, t):
        frame = get_frame(t)
        progress = t / duration
        x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
        current_zoom = 1.0 + (zoom - 1.0) * progress
        h, w = frame.shape[:2]
        new_w, new_h = int(w / current_zoom), int(h / current_zoom)
        if new_w < 1 or new_h < 1:
            return frame
        x1 = int(x * (w - new_w))
        y1 = int(y * (h - new_h))
        cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
        pil = Image.fromarray(cropped.astype("uint8"))
        pil = pil.resize((w, h), Image.LANCZOS)
        return np.array(pil)

    base = base.fl(pan_effect)
    return base


def create_quote_slide(quote_text, image_path, slide_index, total_slides=3):
    """Create a single slide with quote text, background pan, and dark overlay."""
    # Vary pan direction per slide
    directions = [
        ((0.02, 0.02), (0.08, 0.08)),
        ((0.05, 0.0), (0.05, 0.1)),
        ((0.08, 0.08), (0.02, 0.02)),
    ]
    start_pos, end_pos = directions[slide_index % len(directions)]
    bg = create_pan_clip(image_path, SLIDE_DURATION, start_pos, end_pos, zoom=1.08)

    # Dark overlay
    overlay_arr = np.full((SIZE[1], SIZE[0], 3), 30, dtype=np.uint8)
    overlay = ImageClip(overlay_arr, ismask=False).set_duration(SLIDE_DURATION)
    overlay = overlay.set_opacity(0.3)

    # Quote text with serif font
    txt_arr = render_text_pil(f'"{quote_text}"')
    quote_clip = ImageClip(txt_arr).set_duration(SLIDE_DURATION).set_position("center")

    # Accent dash
    dash_arr = render_text_pil("—", size=(200, 100), fontsize=60, font_path=FONT_ITALIC, color=(200, 200, 200, 180))
    accent_clip = ImageClip(dash_arr).set_duration(SLIDE_DURATION).set_position(("center", "bottom"))
    accent_clip = accent_clip.margin(bottom=250, opacity=0)

    # Slide composition
    slide = CompositeVideoClip([bg, overlay, quote_clip, accent_clip], size=SIZE)
    slide = slide.fx(transfx.fadein, FADE_DURATION)
    slide = slide.fx(transfx.fadeout, FADE_DURATION)

    return slide


def generate_reel(post_id=None, output_path=None, quotes=None):
    """Generate Cinematic Quote reel for the specified bundle."""
    state, bundle = load_state(post_id)
    post_id = bundle.get("post_id", "unknown")
    print(f"🎬 Generating Cinematic Quote reel for bundle {post_id}")

    image_path = bundle.get("image", "images/post_20260922_072453.jpg")
    if not os.path.exists(image_path):
        print(f"⚠ Image not found: {image_path}, using fallback")
        image_path = "images/post_20260922_072453.jpg"

    if quotes is None:
        quotes = extract_quotes(bundle)
    print(f"  Quotes ({len(quotes)}):")
    for q in quotes:
        print(f"    - {q[:50]}...")

    # Create slides
    slides = []
    for i, quote in enumerate(quotes):
        slide = create_quote_slide(quote, image_path, i, len(quotes))
        slide = slide.set_start(i * SLIDE_DURATION)
        slides.append(slide)

    # Brand watermark (last slide only)
    brand_arr = render_text_pil(BRAND, size=(400, 80), fontsize=35, font_path=FONT_REG,
                                color=(255, 255, 255, 150))
    brand_clip = ImageClip(brand_arr).set_duration(SLIDE_DURATION).set_position(("center", "bottom"))
    brand_clip = brand_clip.margin(bottom=80, opacity=0)
    brand_clip = brand_clip.set_start(2 * SLIDE_DURATION)

    # Compose all slides
    all_clips = slides + [brand_clip]
    video = CompositeVideoClip(all_clips, size=SIZE)

    # Write
    if output_path is None:
        os.makedirs("reels", exist_ok=True)
        output_path = f"reels/reel_{post_id}_cinematic_quote.mp4"

    print(f"  Writing to {output_path}...")
    video.write_videofile(output_path, fps=FPS, codec="libx264", audio=False,
                          preset="fast", threads=2, logger=None)
    file_size = os.path.getsize(output_path)
    print(f"✅ Done! Size: {file_size / 1024:.0f} KB")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Cinematic Quote reel")
    parser.add_argument("--post_id", help="Bundle post_id")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--quotes", nargs="+", help="Custom quotes (space-separated)")
    args = parser.parse_args()
    quotes = args.quotes if args.quotes else None
    generate_reel(post_id=args.post_id, output_path=args.output, quotes=quotes)
