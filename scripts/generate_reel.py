#!/usr/bin/env python3
"""
Fast Reel Generator using ffmpeg-native filters.

Templates:
  - hook_blast: big bold text every ~1.2s, Ken Burns zoom
  - cinematic_quote: elegant serif, one quote per 3s slide

Usage:
    python scripts/generate_reel.py --template hook_blast --post_id 296
    python scripts/generate_reel.py --template cinematic_quote --post_id 296
"""
import os
import sys
import json
import argparse
import subprocess

BRAND = "@theninestitches"
FONT_BOLD = "C\\:/Windows/Fonts/arialbd.ttf"
FONT_REG = "C\\:/Windows/Fonts/arial.ttf"
FONT_SERIF = "C\\:/Windows/Fonts/georgia.ttf"

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


def escape_drawtext(text):
    """Escape for ffmpeg drawtext."""
    text = text.replace("'", "")
    text = text.replace(":", "\\\\:")
    text = text.replace("%", "%%")
    return text


def generate_hook_blast(image_path, output_path, hooks):
    """Generate Hook-Text Blast using ffmpeg filters."""
    print(f"  Generating Hook-Text Blast: {len(hooks)} hooks")
    
    # Build drawtext filters - one per hook with enable expression
    drawtext_filters = []
    
    for i, hook in enumerate(hooks):
        start = i * 1.2
        end = start + 1.2
        escaped = escape_drawtext(hook)
        
        # Simple enable expression
        enable = "between(t\\\\," + str(round(start, 2)) + "\\\\," + str(round(end, 2)) + ")"
    
    dt = "drawtext=fontfile=" + FONT_BOLD + ":text='" + escaped + "':"
    dt += "fontsize=86:fontcolor=white:bordercolor=black:borderw=3:"
    dt += "x=(w-text_w)/2:y=(h-text_h)/2:"
    dt += "enable='" + enable + "'"
    drawtext_filters.append(dt)
    
    # Brand watermark
    brand_esc = escape_drawtext(BRAND)
    dt_brand = "drawtext=fontfile=" + FONT_REG + ":text='" + brand_esc + "':"
    dt_brand += "fontsize=28:fontcolor=white@0.7:x=30:y=h-th-30"
    drawtext_filters.append(dt_brand)
    
    # Ken Burns zoom
    zoom_expr = "1+0.000694*n"
    vf_chain = "zoompan=z='" + zoom_expr + "':d=1:s=" + str(WIDTH) + "x" + str(HEIGHT) + ":fps=" + str(FPS)
    
    for dt in drawtext_filters:
        vf_chain += "," + dt
    
    # Add fade in/out
    vf_chain += ",fade=t=in:st=0:d=0.5,fade=t=out:st=" + str(DURATION - 0.5) + ":d=0.5"
        
        dt = "drawtext=fontfile=" + FONT_BOLD + ":text='" + escaped + "':"
        dt += "fontsize=86:fontcolor=white:bordercolor=black:borderw=3:"
        dt += "x=(w-text_w)/2:y=(h-text_h)/2:"
        dt += "enable='" + enable + "'"
        drawtext_filters.append(dt)
    
    # Brand watermark
    brand_esc = escape_drawtext(BRAND)
    dt_brand = "drawtext=fontfile=" + FONT_REG + ":text='" + brand_esc + "':"
    dt_brand += "fontsize=28:fontcolor=white@0.7:x=30:y=h-th-30"
    drawtext_filters.append(dt_brand)
    
    # Ken Burns zoom
    zoom_expr = "1+0.000694*n"
    vf_chain = "zoompan=z='" + zoom_expr + "':d=1:s=" + str(WIDTH) + "x" + str(HEIGHT) + ":fps=" + str(FPS)
    
    for dt in drawtext_filters:
        vf_chain += "," + dt
    
    # Add fade in/out
    vf_chain += ",fade=t=in:st=0:d=0.5,fade=t=out:st=" + str(DURATION - 0.5) + ":d=0.5"
    
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-t", str(DURATION),
        "-i", image_path,
        "-vf", vf_chain,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        output_path
    ]
    
    print("    Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("    ❌ ffmpeg error: " + result.stderr[-500:])
        return False
    
    return True


def generate_cinematic_quote(image_path, output_path, quotes):
    """Generate Cinematic Quote using ffmpeg filters."""
    print("  Generating Cinematic Quote: " + str(len(quotes)) + " slides")
    
    drawtext_filters = []
    
    for i, quote in enumerate(quotes):
        start = i * 3.0
        end = start + 3.0
        escaped = escape_drawtext('"' + quote + '"')
        
        enable = "between(t\\\\," + str(round(start, 2)) + "\\\\," + str(round(end, 2)) + ")"
    
    dt = "drawtext=fontfile=" + FONT_BOLD + ":text='" + escaped + "':"
    dt += "fontsize=86:fontcolor=white:bordercolor=black:borderw=3:"
    dt += "x=(w-text_w)/2:y=(h-text_h)/2:"
    dt += "enable='" + enable + "'"
    drawtext_filters.append(dt)
    
    # Brand watermark
    brand_esc = escape_drawtext(BRAND)
    dt_brand = "drawtext=fontfile=" + FONT_REG + ":text='" + brand_esc + "':"
    dt_brand += "fontsize=28:fontcolor=white@0.7:x=30:y=h-th-30"
    drawtext_filters.append(dt_brand)
    
    # Ken Burns zoom
    zoom_expr = "1+0.000694*n"
    vf_chain = "zoompan=z='" + zoom_expr + "':d=1:s=" + str(WIDTH) + "x" + str(HEIGHT) + ":fps=" + str(FPS)
    
    for dt in drawtext_filters:
        vf_chain += "," + dt
    
    # Add fade in/out
    vf_chain += ",fade=t=in:st=0:d=0.5,fade=t=out:st=" + str(DURATION - 0.5) + ":d=0.5"
        
        dt = "drawtext=fontfile=" + FONT_SERIF + ":text='" + escaped + "':"
        dt += "fontsize=58:fontcolor=white:bordercolor=#1a1a2e:borderw=2:"
        dt += "x=(w-text_w)/2:y=(h-text_h)/2:"
        dt += "enable='" + enable + "'"
        drawtext_filters.append(dt)
    
    # Brand on last slide
    brand_esc = escape_drawtext(BRAND)
    dt_brand = "drawtext=fontfile=" + FONT_SERIF + ":text='" + brand_esc + "':"
    dt_brand += "fontsize=28:fontcolor=white@0.6:x=(w-text_w)/2:y=h-th-80:"
    dt_brand += "enable='gte(t\\,6)'"
    drawtext_filters.append(dt_brand)
    
    # Ken Burns + dark overlay
    zoom_expr = "1+0.00037*n"
    vf_chain = "zoompan=z='" + zoom_expr + "':d=1:s=" + str(WIDTH) + "x" + str(HEIGHT) + ":fps=" + str(FPS)
    vf_chain += ",drawbox=x=0:y=0:w=" + str(WIDTH) + ":h=" + str(HEIGHT) + ":color=black@0.3:t=fill"
    
    for dt in drawtext_filters:
        vf_chain += "," + dt
    
    # Add fade in/out
    vf_chain += ",fade=t=in:st=0:d=0.5,fade=t=out:st=" + str(DURATION - 0.5) + ":d=0.5"
    
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-t", str(DURATION),
        "-i", image_path,
        "-vf", vf_chain,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        output_path
    ]
    
    print("    Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("    ❌ ffmpeg error: " + result.stderr[-500:])
        return False
    
    return True


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
