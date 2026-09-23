#!/usr/bin/env python3
"""
Fast MoviePy-style Reel Generator using ffmpeg-python + PIL

MoviePy is too slow for 1080p because it calls Python per frame.
This uses PIL for frame rendering (fast) + ffmpeg for video encoding (C speed).

Usage:
    python scripts/generate_reel_fast.py --template hook_blast --post_id 296
    python scripts/generate_reel_fast.py --template cinematic_quote --post_id 296
"""
import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── CONFIGURATION ───────────────────────────────────────────────
BRAND = "@theninestitches"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_REG = "C:/Windows/Fonts/arial.ttf"
FONT_SERIF = "C:/Windows/Fonts/georgia.ttf"
FONT_SERIF_BOLD = "C:/Windows/Fonts/georgiab.ttf"
FONT_ITALIC = "C:/Windows/Fonts/georgiai.ttf"
FONT_IMPACT = "C:/Windows/Fonts/impact.ttf"

FPS = 24
DURATION = 9
SIZE = (1080, 1920)
WIDTH, HEIGHT = SIZE


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


def render_text(text, size=(1000, 300), fontsize=90, font_path=FONT_BOLD,
                color=(255, 255, 255, 255), stroke_color=(0, 0, 0, 255),
                stroke_width=3, align="center"):
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
        if align == "center":
            x = (size[0] - tw) // 2
        else:
            x = 20
        y = y_start + i * line_height
        # Stroke
        if stroke_width > 0:
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), line, font=font, fill=stroke_color)
        draw.text((x, y), line, font=font, fill=color)

    return img


def load_and_resize_image(image_path):
    """Load image and resize to reel dimensions."""
    if not os.path.exists(image_path):
        return Image.new("RGB", SIZE, (20, 20, 30))
    img = Image.open(image_path).convert("RGB")
    return img.resize(SIZE, Image.LANCZOS)


def ffmpeg_encode(frame_dir, output_path, fps=FPS):
    """Encode frames to video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "fast",
        "-crf", "23",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ffmpeg error: {result.stderr}")
        return False
    return True


def extract_hooks(bundle):
    """Extract hook phrases from bundle's master_reflection."""
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


def generate_hook_blast(image_path, output_path, hooks, hooks_per_sec=1.2):
    """Generate Hook-Text Blast reel (kinetic typography)."""
    print(f"  Generating Hook-Text Blast: {len(hooks)} hooks")
    
    img = load_and_resize_image(image_path)
    total_frames = DURATION * FPS
    word_frames = int(1.2 * FPS)
    
    tmpdir = tempfile.mkdtemp()
    try:
        for frame_idx in range(total_frames):
            t = frame_idx / FPS
            img_arr = np.array(img, dtype=np.float64)
            
            # Ken Burns zoom
            zoom = 1.0 + 0.15 * (t / DURATION)
            zh = int(HEIGHT / zoom)
            zw = int(WIDTH / zoom)
            y1 = (HEIGHT - zh) // 2
            x1 = (WIDTH - zw) // 2
            
            pil_frame = Image.fromarray(img_arr.astype(np.uint8))
            cropped = pil_frame.crop((x1, y1, x1 + zw, y1 + zh))
            cropped = cropped.resize(SIZE, Image.LANCZOS)
            frame = np.array(cropped)
            
            # Current hook
            hook_idx = int(t / 1.2)
            hook_start = hook_idx * 1.2
            hook_progress = (t - hook_start) / 0.25  # pop-in over 0.25s
            
            if hook_idx < len(hooks) and hook_progress < 2.0:
                # Pop-in animation
                if hook_progress < 1.0:
                    scale = 0.3 + 0.7 * (1 - (1 - hook_progress) ** 2)
                    alpha = min(hook_progress * 2, 1.0)
                else:
                    scale = 1.0
                    alpha = max(0, 1 - (hook_progress - 1.0) * 2)
                
                txt_img = render_text(hooks[hook_idx], size=(1000, 300), fontsize=90)
                txt_arr = np.array(txt_img)
                
                # Scale text
                th, tw = txt_arr.shape[:2]
                new_tw, new_th = int(tw * scale), int(th * scale)
                if new_tw > 0 and new_th > 0:
                    txt_pil = Image.fromarray(txt_arr)
                    txt_pil = txt_pil.resize((new_tw, new_th), Image.LANCZOS)
                    txt_arr = np.array(txt_pil)
                    
                    # Center on frame
                    y_off = (HEIGHT - new_th) // 2
                    x_off = (WIDTH - new_tw) // 2
                    
                    # Alpha blend
                    txt_rgb = txt_arr[:, :, :3]
                    txt_a = txt_arr[:, :, 3:] / 255.0 * alpha
                    
                    y_end = min(y_off + new_th, HEIGHT)
                    x_end = min(x_off + new_tw, WIDTH)
                    sy = y_end - y_off
                    sx = x_end - x_off
                    
                    if sy > 0 and sx > 0:
                        frame[y_off:y_end, x_off:x_end] = (
                            frame[y_off:y_end, x_off:x_end] * (1 - txt_a[:sy, :sx]) +
                            txt_rgb[:sy, :sx] * txt_a[:sy, :sx]
                        ).astype(np.uint8)
            
            # Brand watermark
            brand = render_text(BRAND, size=(300, 60), fontsize=28, font_path=FONT_REG, color=(255, 255, 255, 140))
            brand_arr = np.array(brand)
            by = HEIGHT - 80
            bx = 30
            bah, baw = brand_arr.shape[:2]
            ba = brand_arr[:, :, 3:] / 255.0 * 0.7
            brgb = brand_arr[:, :, :3]
            frame[by:by+bah, bx:bx+baw] = (
                frame[by:by+bah, bx:bx+baw] * (1 - ba) + brgb * ba
            ).astype(np.uint8)
            
            # Save frame
            Image.fromarray(frame).save(os.path.join(tmpdir, f"frame_{frame_idx:05d}.png"))
            
            if frame_idx % 24 == 0:
                print(f"    Frame {frame_idx}/{total_frames}", end="\r")
        
        print(f"\n    Encoding with ffmpeg...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        success = ffmpeg_encode(tmpdir, output_path)
        return success
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def generate_cinematic_quote(image_path, output_path, quotes):
    """Generate Cinematic Quote reel (slow elegant reveal)."""
    print(f"  Generating Cinematic Quote: {len(quotes)} slides")
    
    img = load_and_resize_image(image_path)
    slide_duration = 3.0
    total_frames = DURATION * FPS
    
    # Pan directions per slide
    pans = [
        ((0.02, 0.02), (0.08, 0.08)),
        ((0.05, 0.0), (0.05, 0.1)),
        ((0.08, 0.08), (0.02, 0.02)),
    ]
    
    tmpdir = tempfile.mkdtemp()
    try:
        for frame_idx in range(total_frames):
            t = frame_idx / FPS
            slide_idx = int(t / slide_duration)
            slide_t = t - slide_idx * slide_duration
            slide_progress = slide_t / slide_duration
            
            if slide_idx >= len(quotes):
                slide_idx = len(quotes) - 1
                slide_progress = 0
            
            img_arr = np.array(img, dtype=np.float64)
            
            # Ken Burns pan + zoom
            start_pos, end_pos = pans[slide_idx % len(pans)]
            x = start_pos[0] + (end_pos[0] - start_pos[0]) * slide_progress
            y = start_pos[1] + (end_pos[1] - start_pos[1]) * slide_progress
            zoom = 1.0 + 0.08 * slide_progress
            zh = int(HEIGHT / zoom)
            zw = int(WIDTH / zoom)
            x1 = int(x * (WIDTH - zw))
            y1 = int(y * (HEIGHT - zh))
            
            pil_frame = Image.fromarray(img_arr.astype(np.uint8))
            cropped = pil_frame.crop((x1, y1, x1 + zw, y1 + zh))
            cropped = cropped.resize(SIZE, Image.LANCZOS)
            frame = np.array(cropped).copy()
            
            # Dark overlay
            overlay = np.full_like(frame, 30)
            alpha = 0.3
            frame = (frame * (1 - alpha) + overlay * alpha).astype(np.uint8)
            
            # Quote text with fade
            fade_in = min(slide_t / 0.6, 1.0)
            fade_out = max(0, 1 - (slide_t - (slide_duration - 0.6)) / 0.6) if slide_t > slide_duration - 0.6 else 1.0
            text_alpha = min(fade_in, fade_out)
            
            if text_alpha > 0:
                quote = quotes[slide_idx]
                txt_img = render_text(f'"{quote}"', size=(950, 400), fontsize=60, font_path=FONT_SERIF,
                                      color=(255, 255, 255, 255), stroke_color=(26, 26, 46, 255), stroke_width=2)
                txt_arr = np.array(txt_img)
                th, tw = txt_arr.shape[:2]
                y_off = (HEIGHT - th) // 2
                x_off = (WIDTH - tw) // 2
                
                y_end = min(y_off + th, HEIGHT)
                x_end = min(x_off + tw, WIDTH)
                sy = y_end - y_off
                sx = x_end - x_off
                
                if sy > 0 and sx > 0:
                    txt_rgb = txt_arr[:sy, :sx, :3]
                    txt_a = txt_arr[:sy, :sx, 3:] / 255.0 * text_alpha
                    frame[y_off:y_end, x_off:x_end] = (
                        frame[y_off:y_end, x_off:x_end] * (1 - txt_a) + txt_rgb * txt_a
                    ).astype(np.uint8)
                
                # Dash accent
                dash = render_text("—", size=(200, 80), fontsize=50, font_path=FONT_ITALIC, color=(200, 200, 200, int(180 * text_alpha)))
                dash_arr = np.array(dash)
                dah, daw = dash_arr.shape[:2]
                dy = HEIGHT - 250
                dx = (WIDTH - daw) // 2
                da = dash_arr[:, :, 3:] / 255.0 * text_alpha
                drgb = dash_arr[:, :, :3]
                frame[dy:dy+dah, dx:dx+daw] = (
                    frame[dy:dy+dah, dx:dx+daw] * (1 - da) + drgb * da
                ).astype(np.uint8)
            
            # Brand on last slide
            if slide_idx == len(quotes) - 1 and slide_t > 1.0:
                brand = render_text(BRAND, size=(300, 60), fontsize=28, font_path=FONT_SERIF, color=(255, 255, 255, 150))
                brand_arr = np.array(brand)
                by = HEIGHT - 80
                bx = (WIDTH - brand_arr.shape[1]) // 2
                bah, baw = brand_arr.shape[:2]
                ba = brand_arr[:, :, 3:] / 255.0 * 0.7
                brgb = brand_arr[:, :, :3]
                frame[by:by+bah, bx:bx+baw] = (
                    frame[by:by+bah, bx:bx+baw] * (1 - ba) + brgb * ba
                ).astype(np.uint8)
            
            Image.fromarray(frame).save(os.path.join(tmpdir, f"frame_{frame_idx:05d}.png"))
            
            if frame_idx % 24 == 0:
                print(f"    Frame {frame_idx}/{total_frames}", end="\r")
        
        print(f"\n    Encoding with ffmpeg...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        success = ffmpeg_encode(tmpdir, output_path)
        return success
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Generate reels using ffmpeg + PIL")
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
