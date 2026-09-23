#!/usr/bin/env python3
"""
MoviePy Reel Template 1: Hook-Text Blast (Kinetic Typography)

Big bold hook text changes every ~1.2s with elastic pop-in animation.
Slow Ken Burns zoom on background. Brand watermark, fade out at end.

Text rendered with PIL (no ImageMagick dependency).

Usage:
    python scripts/generate_reel_hook_blast.py --post_id 296
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
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_REG = "C:/Windows/Fonts/arial.ttf"

# Animation
WORD_DURATION = 1.2
POP_IN_DURATION = 0.25
ZOOM_FACTOR = 1.15

# Output
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
    while len(hooks) < int(DURATION / WORD_DURATION):
        hooks.extend(hooks[:2])
    return hooks[:int(DURATION / WORD_DURATION)]


def render_text_pil(text, size=(1000, 300), fontsize=90, font_path=FONT_BOLD,
                    color=(255, 255, 255, 255), stroke_color=(0, 0, 0, 255), 
                    stroke_width=3, align="center"):
    """Render text with PIL and return as RGBA numpy array."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, fontsize)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Handle color strings
    if isinstance(color, str):
        if color.startswith("rgba("):
            import re
            nums = re.findall(r'\d+', color)
            color = tuple(int(x) for x in nums)
        elif color == "white":
            color = (255, 255, 255, 255)
        elif color == "black":
            color = (0, 0, 0, 255)
        else:
            color = (255, 255, 255, 255)
    if isinstance(stroke_color, str):
        if stroke_color == "black":
            stroke_color = (0, 0, 0, 255)
        elif stroke_color == "white":
            stroke_color = (255, 255, 255, 255)
        else:
            stroke_color = (0, 0, 0, 255)

    # Word wrap
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

    # Total text height
    line_height = fontsize + 10
    total_h = len(lines) * line_height
    y_start = (size[1] - total_h) // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if align == "center":
            x = (size[0] - tw) // 2
        else:
            x = 20
        y = y_start + i * line_height
        # Stroke/outline
        if stroke_width > 0:
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), line, font=font, fill=stroke_color)
        # Main text
        draw.text((x, y), line, font=font, fill=color)

    return np.array(img)


def create_ken_burns_clip(image_path, duration, zoom_factor=1.15):
    """Apply Ken Burns zoom to background image."""
    if not os.path.exists(image_path):
        bg = ImageClip(np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8))
        return bg.set_duration(duration)

    img = Image.open(image_path).convert("RGB")
    img = img.resize((SIZE[0], SIZE[1]), Image.LANCZOS)
    img_arr = np.array(img)
    base = ImageClip(img_arr).set_duration(duration)

    def zoom_effect(get_frame, t):
        frame = get_frame(t)
        progress = t / duration
        current_zoom = 1.0 + (zoom_factor - 1.0) * progress
        h, w = frame.shape[:2]
        new_w, new_h = int(w / current_zoom), int(h / current_zoom)
        if new_w < 1 or new_h < 1:
            return frame
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
        pil = Image.fromarray(cropped.astype("uint8"))
        pil = pil.resize((w, h), Image.LANCZOS)
        return np.array(pil)

    base = base.fl(zoom_effect)
    return base


def create_text_frame_animated(hook, duration, pop_in=0.25):
    """Create text frame with pop-in animation."""
    txt_arr = render_text_pil(hook)
    # Ensure RGBA
    if txt_arr.shape[2] == 3:
        alpha = np.full((txt_arr.shape[0], txt_arr.shape[1], 1), 255, dtype=np.uint8)
        txt_arr = np.concatenate([txt_arr, alpha], axis=2)
    
    txt_clip = ImageClip(txt_arr).set_duration(duration)

    # Pop-in: fade + slight scale via MoviePy
    def pop_in_effect(get_frame, t):
        frame = get_frame(t)
        progress = min(t / pop_in, 1.0)
        if progress < 1.0:
            # Scale up from 30% to 100%
            scale = 0.3 + 0.7 * (1 - (1 - progress) ** 2)
            h, w = frame.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            # Ensure same number of channels
            if frame.shape[2] == 4:
                canvas = np.zeros((h, w, 4), dtype=np.uint8)
            else:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
            pil = Image.fromarray(frame)
            pil = pil.resize((new_w, new_h), Image.LANCZOS)
            pil_arr = np.array(pil)
            # If canvas is RGBA but pil is RGB, add alpha
            if canvas.shape[2] == 4 and pil_arr.shape[2] == 3:
                alpha = np.full((pil_arr.shape[0], pil_arr.shape[1], 1), 255, dtype=np.uint8)
                pil_arr = np.concatenate([pil_arr, alpha], axis=2)
            x = (w - new_w) // 2
            y = (h - new_h) // 2
            canvas[y:y+new_h, x:x+new_w] = pil_arr[:new_h, :new_w]
            return canvas
        return frame

    txt_clip = txt_clip.fl(pop_in_effect)
    txt_clip = txt_clip.fx(transfx.fadein, pop_in)
    txt_clip = txt_clip.fx(transfx.fadeout, 0.3)
    txt_clip = txt_clip.set_position("center")

    return txt_clip


def generate_reel(post_id=None, output_path=None, hooks=None):
    """Generate Hook-Text Blast reel for the specified bundle."""
    state, bundle = load_state(post_id)
    post_id = bundle.get("post_id", "unknown")
    print(f"🎬 Generating Hook-Text Blast reel for bundle {post_id}")

    image_path = bundle.get("image", "images/post_20260922_072453.jpg")
    if not os.path.exists(image_path):
        print(f"⚠ Image not found: {image_path}, using fallback")
        image_path = "images/post_20260922_072453.jpg"

    if hooks is None:
        hooks = extract_hooks(bundle)
    print(f"  Hooks ({len(hooks)}): {hooks[:5]}...")

    bg = create_ken_burns_clip(image_path, DURATION, ZOOM_FACTOR)

    text_clips = []
    for i, hook in enumerate(hooks):
        start_time = i * WORD_DURATION
        if start_time + WORD_DURATION > DURATION:
            break
        txt_clip = create_text_frame_animated(hook, WORD_DURATION, POP_IN_DURATION)
        txt_clip = txt_clip.set_start(start_time)
        text_clips.append(txt_clip)

    # Brand watermark
    brand_arr = render_text_pil(BRAND, size=(400, 80), fontsize=35, font_path=FONT_REG,
                                color="rgba(255,255,255,180)")
    brand_clip = ImageClip(brand_arr).set_duration(DURATION).set_position(("left", "bottom"))
    brand_clip = brand_clip.margin(left=40, bottom=40, opacity=0)

    all_clips = [bg] + text_clips + [brand_clip]
    video = CompositeVideoClip(all_clips, size=SIZE)
    video = video.fx(transfx.fadeout, 0.5)

    if output_path is None:
        os.makedirs("reels", exist_ok=True)
        output_path = f"reels/reel_{post_id}_hook_blast.mp4"

    print(f"  Writing to {output_path}...")
    video.write_videofile(output_path, fps=FPS, codec="libx264", audio=False,
                          preset="fast", threads=2, logger=None)
    file_size = os.path.getsize(output_path)
    print(f"✅ Done! Size: {file_size / 1024:.0f} KB")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hook-Text Blast reel")
    parser.add_argument("--post_id", help="Bundle post_id")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--hooks", nargs="+", help="Custom hook text (space-separated)")
    args = parser.parse_args()
    hooks = args.hooks if args.hooks else None
    generate_reel(post_id=args.post_id, output_path=args.output, hooks=hooks)
