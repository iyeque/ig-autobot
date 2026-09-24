#!/usr/bin/env python3
"""
Fast Reel Generator - PIL static frames + ffmpeg JPG clip concat.

Generates one JPG per text overlay, creates zoompan clips, concatenates.

Templates:
  - hook_blast: big bold text every ~1.2s (9s, 7 clips) - FAST (~30s)
  - cinematic_quote: elegant serif slides w/ xfade transitions (9s, 3 slides) - FAST (~15s)
  - word_ripple: words appear one-by-one (9s, 30 clips) - FAST (~45s)

Usage:
    python scripts/generate_reel.py --template hook_blast --post_id 296
    python scripts/generate_reel.py --template cinematic_quote --post_id 296
    python scripts/generate_reel.py --template word_ripple --post_id 296
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

FPS = 24
DURATION = 9
WIDTH, HEIGHT = 1080, 1920


def run_ffmpeg(cmd, **kwargs):
    """Run ffmpeg with FONTCONFIG_PATH set for Windows font loading."""
    env = os.environ.copy()
    env["FONTCONFIG_PATH"] = "C:/Windows/Fonts"
    return subprocess.run(cmd, env=env, capture_output=True, text=True, **kwargs)


def load_state(post_id=None):
    with open("state.json", "r", encoding="utf-8") as f:
        state = json.load(f)
    if post_id:
        for b in state.get("content_queue", []):
            if str(b.get("post_id")) == str(post_id):
                return state, b
        return state, {
            "post_id": post_id,
            "image": "images/post_20260922_072453.jpg",
            "master_reflection": "We live in systems designed to capture our attention.",
            "topic": "Bundle " + str(post_id),
            "captions": {"instagram": ""},
        }
    return state, state.get("active_bundle", {})


def get_text(bundle):
    text = bundle.get("master_reflection", "")
    if not text:
        text = bundle.get("captions", {}).get("instagram", "")
    if not text:
        text = "We live in systems designed to capture our attention and return us to the same loops."
    return text


def extract_hooks(bundle):
    """Extract hook phrases from master_reflection."""
    text = get_text(bundle)
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
    if not hooks:
        hooks = ["THE LOOP", "WE LIVE IN", "SYSTEMS SHAPE", "OUR ATTENTION", "RETURNS US"]
    if topic:
        hooks.insert(0, topic.upper()[:30])
    while len(hooks) < int(DURATION / 1.2):
        hooks.extend(hooks[:2])
    return hooks[:int(DURATION / 1.2)]


def extract_quotes(bundle):
    """Extract 3 cinematic quotes from master_reflection."""
    text = get_text(bundle)
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
    if not quotes:
        quotes = [
            "We live in systems designed to capture our attention",
            "and return us to the same loops",
            "every single day"
        ]
    while len(quotes) < 3:
        quotes.extend(quotes[:1])
    return quotes[:3]


def extract_words(bundle):
    """Extract individual words for word_ripple effect."""
    import re
    text = get_text(bundle)
    topic = bundle.get("topic", "")
    clean = re.sub(r'[^\w\s]', '', text)
    words = [w for w in clean.split() if len(w) > 2]
    if topic:
        words = topic.upper().split()[:3] + words
    min_words = int(DURATION * 1.8)
    max_words = int(DURATION / 0.3)
    if len(words) == 0:
        words = ["THE", "LOOP", "CONTINUES", "EVERY", "SINGLE", "DAY", "WE", "LIVE", "IN", "SYSTEMS", "THAT", "SHAPE", "OUR", "MINDS", "AND", "RETURN"]
    if len(words) > max_words:
        words = words[:max_words]
    while len(words) < min_words:
        words.extend(words[:10])
    return words


def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except (OSError, IOError):
        return ImageFont.load_default()


def render_text(text, size=(1000, 300), fontsize=86, font=FONT_BOLD,
                color=(255, 255, 255, 255), stroke=(0, 0, 0, 255), stroke_w=3):
    """Render text with stroke on RGBA image."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    f = get_font(font, fontsize)
    words = text.split()
    lines = []
    current = ""
    for w in words:
        test = current + " " + w if current else w
        bb = draw.textbbox((0, 0), test, font=f)
        if bb[2] - bb[0] <= size[0] - 40:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    lh = fontsize + 10
    total = len(lines) * lh
    y = (size[1] - total) // 2
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=f)
        tw = bb[2] - bb[0]
        x = (size[0] - tw) // 2
        for dx in range(-stroke_w, stroke_w + 1):
            for dy in range(-stroke_w, stroke_w + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), line, font=f, fill=stroke)
        draw.text((x, y), line, font=f, fill=color)
        y += lh
    return img


def make_clip(jpg_path, clip_path, duration, zoom, fade_in=0.0, fade_out=0.0):
    """Create a zoompan clip from a static image with optional fade."""
    d = int(duration * FPS)
    vf_parts = ["zoompan=z=" + str(zoom) + ":d=" + str(d) + ":s=" + str(WIDTH) + "x" + str(HEIGHT) + ":fps=" + str(FPS)]
    if fade_in > 0:
        vf_parts.append("fade=t=in:st=0:d=" + str(fade_in))
    if fade_out > 0:
        fade_out_st = round(duration - fade_out, 2)
        if fade_out_st < 0:
            fade_out_st = 0
        vf_parts.append("fade=t=out:st=" + str(fade_out_st) + ":d=" + str(fade_out))
    vf_chain = ",".join(vf_parts)
    cmd = ["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-t", str(duration), "-i", jpg_path,
           "-vf", vf_chain, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23", "-an", clip_path]
    return run_ffmpeg(cmd)


def generate_hook_blast(image_path, output_path, hooks):
    """Generate Hook-Text Blast: big bold text every ~1.2s."""
    print("  Hook-Text Blast: " + str(len(hooks)) + " hooks")
    sys.stdout.flush()
    if not os.path.exists(image_path):
        return False
    base = Image.open(image_path).convert("RGBA").resize((WIDTH, HEIGHT), Image.LANCZOS)
    tmpdir = tempfile.mkdtemp()
    try:
        clip_dur = 1.2
        clips = []
        for i, hook in enumerate(hooks):
            frame = base.copy()
            txt = render_text(hook, fontsize=90, font=FONT_BOLD)
            frame.paste(txt, ((WIDTH - txt.size[0]) // 2, (HEIGHT - txt.size[1]) // 2), txt)
            if i == len(hooks) - 1:
                brand = render_text(BRAND, size=(300, 60), fontsize=28, font=FONT_REG, color=(255,255,255,140))
                frame.paste(brand, (30, HEIGHT - 80), brand)
            jpg_path = os.path.join(tmpdir, "hook_" + str(i) + ".jpg")
            frame.convert("RGB").save(jpg_path, quality=90)
            clip_path = os.path.join(tmpdir, "hook_" + str(i) + ".mp4")
            zoom = round(1.0 + 0.015 * i, 4)
            result = make_clip(jpg_path, clip_path, clip_dur, zoom)
            if result.returncode != 0:
                print("Clip " + str(i) + " error: " + result.stderr[-200:])
                return False
            clips.append(clip_path)
            print("    Hook " + str(i+1) + "/" + str(len(hooks)))
            sys.stdout.flush()
        return concat_clips(clips, output_path, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def generate_cinematic_quote(image_path, output_path, quotes):
    """
    Generate Cinematic Quote: elegant serif, one quote per 3s slide.
    Uses ffmpeg xfade for smooth crossfade transitions (FAST on CPU).
    No zoompan — static slides with crossfade, like Instagram carousels.
    """
    print("  Cinematic Quote: " + str(len(quotes)) + " slides (xfade)")
    sys.stdout.flush()
    if not os.path.exists(image_path):
        return False
    base = Image.open(image_path).convert("RGBA").resize((WIDTH, HEIGHT), Image.LANCZOS)
    tmpdir = tempfile.mkdtemp()
    try:
        slide_dur = 3.0
        clips = []
        
        for i, quote in enumerate(quotes):
            frame = base.copy()
            
            # Dark overlay for cinematic feel
            overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 76))
            frame = Image.alpha_composite(frame, overlay)
            
            # Render quote text
            quote_text = '"' + quote + '"'
            txt = render_text(quote_text, size=(950, 400), fontsize=58, font=FONT_SERIF)
            frame.paste(txt, ((WIDTH - txt.size[0]) // 2, (HEIGHT - txt.size[1]) // 2), txt)
            
            # Brand on last slide
            if i == len(quotes) - 1:
                brand = render_text(BRAND, size=(300, 60), fontsize=28, font=FONT_SERIF, color=(255,255,255,100))
                frame.paste(brand, (WIDTH - 350, HEIGHT - 80), brand)
            
            jpg_path = os.path.join(tmpdir, "slide_" + str(i) + ".jpg")
            frame.convert("RGB").save(jpg_path, quality=90)
            
            # Create a plain video clip (no zoompan!)
            clip_path = os.path.join(tmpdir, "slide_" + str(i) + ".mp4")
            cmd = ["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-t", str(slide_dur), "-i", jpg_path,
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23", "-an", clip_path]
            result = run_ffmpeg(cmd)
            if result.returncode != 0:
                print("Slide " + str(i) + " error: " + result.stderr[-200:])
                return False
            clips.append(clip_path)
            print("    Slide " + str(i+1) + "/" + str(len(quotes)))
            sys.stdout.flush()
        
        # Use xfade for crossfade transitions between slides
        if len(clips) == 1:
            shutil.copy(clips[0], output_path)
            return True
        
        filter_parts = []
        offset = slide_dur - 0.5
        prev_label = "0:v"
        
        for i in range(1, len(clips)):
            curr_label = str(i) + ":v"
            out_label = "f" + str(i-1) if i < len(clips) - 1 else "out"
            filter_parts.append(
                "[" + prev_label + "][" + curr_label + "]xfade=transition=fade:duration=0.5:offset=" + str(round(offset, 2)) + "[" + out_label + "]"
            )
            prev_label = out_label
            offset += slide_dur - 0.5
        
        filter_str = ";".join(filter_parts)
        
        cmd = ["ffmpeg", "-y"]
        for clip in clips:
            cmd.extend(["-i", clip])
        cmd.extend(["-filter_complex", filter_str, "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-an", output_path])
        
        print("  Applying xfade transitions...")
        sys.stdout.flush()
        result = run_ffmpeg(cmd)
        if result.returncode != 0:
            print("xfade error: " + result.stderr[-300:])
            return False
        return True
        
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def generate_word_ripple(image_path, output_path, words):
    """Generate Word Ripple: words appear one-by-one with ripple fade."""
    print("  Word Ripple: " + str(len(words)) + " words")
    sys.stdout.flush()
    if not os.path.exists(image_path):
        return False
    base = Image.open(image_path).convert("RGBA").resize((WIDTH, HEIGHT), Image.LANCZOS)
    tmpdir = tempfile.mkdtemp()
    try:
        word_dur = round(DURATION / len(words), 2)
        clips = []
        for i, word in enumerate(words):
            frame = base.copy()
            txt = render_text(word.upper(), fontsize=86, font=FONT_BOLD)
            frame.paste(txt, ((WIDTH - txt.size[0]) // 2, (HEIGHT - txt.size[1]) // 2), txt)
            brand = render_text(BRAND, size=(300, 60), fontsize=28, font=FONT_REG, color=(255,255,255,140))
            frame.paste(brand, (30, HEIGHT - 80), brand)
            jpg_path = os.path.join(tmpdir, "word_" + str(i) + ".jpg")
            frame.convert("RGB").save(jpg_path, quality=90)
            clip_path = os.path.join(tmpdir, "word_" + str(i) + ".mp4")
            zoom = round(1.0 + 0.01 * i, 4)
            result = make_clip(jpg_path, clip_path, word_dur, zoom, fade_in=0.1, fade_out=0.1)
            if result.returncode != 0:
                print("Word " + str(i) + " error: " + result.stderr[-200:])
                return False
            clips.append(clip_path)
            if i % 5 == 0:
                print("    Word " + str(i+1) + "/" + str(len(words)))
                sys.stdout.flush()
        print("    All " + str(len(words)) + " words done, concatenating...")
        sys.stdout.flush()
        return concat_clips(clips, output_path, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def concat_clips(clips, output_path, tmpdir):
    """Concatenate clips using ffmpeg concat demuxer."""
    concat_file = os.path.join(tmpdir, "concat.txt")
    with open(concat_file, "w") as f:
        for clip in clips:
            f.write("file '" + clip + "'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", "-movflags", "+faststart", output_path]
    result = run_ffmpeg(cmd)
    if result.returncode != 0:
        print("Concat error: " + result.stderr[-200:])
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", choices=["hook_blast", "cinematic_quote", "word_ripple"], default="hook_blast")
    parser.add_argument("--post_id")
    parser.add_argument("--output")
    args = parser.parse_args()
    state, bundle = load_state(args.post_id)
    post_id = bundle.get("post_id", "unknown")
    print("Generating " + args.template + " for bundle " + str(post_id))
    sys.stdout.flush()
    image_path = bundle.get("image", "images/post_20260922_072453.jpg")
    if not os.path.exists(image_path):
        image_path = "images/post_20260922_072453.jpg"
    if args.output:
        output_path = args.output
    else:
        os.makedirs("reels", exist_ok=True)
        output_path = "reels/reel_" + str(post_id) + "_" + args.template + ".mp4"
    if args.template == "hook_blast":
        success = generate_hook_blast(image_path, output_path, extract_hooks(bundle))
    elif args.template == "cinematic_quote":
        success = generate_cinematic_quote(image_path, output_path, extract_quotes(bundle))
    else:
        success = generate_word_ripple(image_path, output_path, extract_words(bundle))
    if success:
        print("OK " + str(os.path.getsize(output_path) // 1024) + " KB")
    else:
        print("FAILED")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
