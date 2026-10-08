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
import re
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont

BRAND = "@theninestitches"

# Font paths: Windows (local dev) vs Linux (CI)
import sys
if sys.platform == "win32":
    FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
    FONT_REG = "C:/Windows/Fonts/arial.ttf"
    FONT_SERIF = "C:/Windows/Fonts/georgia.ttf"
else:
    # Common Linux fonts (Ubuntu CI runner)
    FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

FPS = 24
DURATION = 9
WIDTH, HEIGHT = 1080, 1920


def run_ffmpeg(cmd, **kwargs):
    """Run ffmpeg with FONTCONFIG_PATH set for Windows font loading (no-op on Linux)."""
    env = os.environ.copy()
    if sys.platform == "win32":
        env["FONTCONFIG_PATH"] = "C:/Windows/Fonts"
    return subprocess.run(cmd, env=env, capture_output=True, text=True, **kwargs)


def load_state(post_id=None):
    with open("state.json", "r", encoding="utf-8") as f:
        state = json.load(f)
    if post_id:
        # The bundle can be in any of three places depending on when this runs:
        # content_queue (before generation finishes), active_bundle (after
        # bot.py promotes it), or pending_bundle (mid-generation). Searching
        # only content_queue meant the lookup silently failed once bot.py had
        # promoted the bundle, and the caller fell back to a hardcoded image.
        candidates = []
        candidates.extend(state.get("content_queue", []) or [])
        for key in ("active_bundle", "pending_bundle"):
            b = state.get(key)
            if isinstance(b, dict):
                candidates.append(b)
        # A bundle that has already been published is no longer in the queue or
        # active_bundle — its content survives only in posted_bundle_content,
        # which shared_utils records at publish time. Without this lookup,
        # regenerating a reel for a shipped bundle silently fell back to the
        # placeholder topic ("Bundle 312") and a stock reflection.
        saved = (state.get("posted_bundle_content") or {}).get(str(post_id))
        if saved:
            return state, saved
        for b in candidates:
            if str(b.get("post_id")) == str(post_id):
                return state, b
        # Loud fallback: a silent hardcoded image hid this bug for weeks.
        print(
            f"⚠ WARNING: bundle {post_id} not found in content_queue, "
            "active_bundle or pending_bundle — falling back to placeholder image. "
            "The generated reel will NOT use this bundle's artwork."
        )
        sys.stdout.flush()
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


def render_text(text, size=None, fontsize=86, font=FONT_BOLD,
                color=(255, 255, 255, 255), stroke=(0, 0, 0, 255), stroke_w=3,
                panel=True):
    """Render text with stroke on a transparent RGBA image.

    The canvas auto-sizes to the wrapped text. It used to be a fixed
    (1000, 300), so a 4-line hook at fontsize 90 needed 400px and started at
    y=-50 — the first line was drawn above the canvas and lost, and successive
    hooks collided in the middle of the frame.

    A translucent panel is drawn behind the text so it stays readable over a
    busy photo, instead of relying on a thin stroke alone.
    """
    measure = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    mdraw = ImageDraw.Draw(measure)
    f = get_font(font, fontsize)

    max_w = (size[0] if size else WIDTH - 120)
    words = text.split()
    lines = []
    current = ""
    for w in words:
        test = current + " " + w if current else w
        bb = mdraw.textbbox((0, 0), test, font=f)
        if bb[2] - bb[0] <= max_w - 40:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)

    lh = fontsize + 10
    total = len(lines) * lh
    # Auto-size the canvas to the wrapped text (plus padding for the panel).
    pad_y = 40 if panel else 0
    canvas_w = size[0] if size else max_w
    canvas_h = max(size[1] if size else 0, total + pad_y * 2)

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if panel:
        # Rounded translucent panel behind the text for contrast on any photo.
        draw.rounded_rectangle(
            [(0, 0), (canvas_w - 1, canvas_h - 1)],
            radius=24, fill=(0, 0, 0, 165),
        )

    y = (canvas_h - total) // 2
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=f)
        tw = bb[2] - bb[0]
        x = (canvas_w - tw) // 2
        for dx in range(-stroke_w, stroke_w + 1):
            for dy in range(-stroke_w, stroke_w + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), line, font=f, fill=stroke)
        draw.text((x, y), line, font=f, fill=color)
        y += lh
    return img


def make_clip(jpg_path, clip_path, duration, zoom, fade_in=0.0, fade_out=0.0):
    """Create a zoompan clip from a static image with optional fade.

    zoompan's `d` is the number of OUTPUT frames produced per INPUT frame, not
    the total clip length. Combined with `-loop 1` (which feeds many input
    frames) the old `d=duration*FPS` multiplied the clip by that factor: a
    1.2s clip came out at 33.8s and a 7-clip reel at 236s instead of 9s.

    Fix: d=1 (one output frame per input frame) and bound the total with
    -frames:v so the clip is exactly duration*FPS frames long.
    """
    total_frames = max(1, int(round(duration * FPS)))
    vf_parts = ["zoompan=z=" + str(zoom) + ":d=1:s=" + str(WIDTH) + "x" + str(HEIGHT) + ":fps=" + str(FPS)]
    if fade_in > 0:
        vf_parts.append("fade=t=in:st=0:d=" + str(fade_in))
    if fade_out > 0:
        fade_out_st = round(duration - fade_out, 2)
        if fade_out_st < 0:
            fade_out_st = 0
        vf_parts.append("fade=t=out:st=" + str(fade_out_st) + ":d=" + str(fade_out))
    vf_chain = ",".join(vf_parts)
    cmd = ["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-t", str(duration), "-i", jpg_path,
           "-vf", vf_chain, "-frames:v", str(total_frames),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "26",
           "-maxrate", "4M", "-bufsize", "8M", "-an", clip_path]
    return run_ffmpeg(cmd)


def generate_hook_blast(image_path, output_path, hooks):
    """Generate Hook-Text Blast: big bold text every ~1.2s."""
    print("  Hook-Text Blast: " + str(len(hooks)) + " hooks")
    sys.stdout.flush()
    if not os.path.exists(image_path):
        return False
    # The published post_*.jpg already has the quote baked in by the image
    # generator. Building a reel from it double-bakes the text (the reel
    # overlays the same words on top), producing a ghosted double exposure.
    # Prefer the matching _clean.jpg, which is the bare photo.
    clean = re.sub(r"^(.*/post_\d+_\d+)\.jpg$", r"\1_clean.jpg", image_path)
    if os.path.exists(clean):
        print("  Using clean base (no baked-in text): " + os.path.basename(clean))
        sys.stdout.flush()
        image_path = clean
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
    # The published post_*.jpg already has the quote baked in by the image
    # generator. Building a reel from it double-bakes the text (the reel
    # overlays the same words on top), producing a ghosted double exposure.
    # Prefer the matching _clean.jpg, which is the bare photo.
    clean = re.sub(r"^(.*/post_\d+_\d+)\.jpg$", r"\1_clean.jpg", image_path)
    if os.path.exists(clean):
        print("  Using clean base (no baked-in text): " + os.path.basename(clean))
        sys.stdout.flush()
        image_path = clean
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
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "26",
                   "-maxrate", "4M", "-bufsize", "8M", "-an", clip_path]
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
        # -maxrate/-bufsize cap the output regardless of source complexity.
        # The 320 reel came out at 16.2 Mbps (17.9 MB for 8.7s) because the
        # final xfade encode had no rate control at all, versus 1.5-3.6 Mbps
        # for every other reel. A photographic source with grain defeats CRF
        # alone, so cap it explicitly.
        cmd.extend(["-filter_complex", filter_str, "-map", "[out]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "26",
                    "-maxrate", "4M", "-bufsize", "8M", "-an", output_path])
        
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
    # The published post_*.jpg already has the quote baked in by the image
    # generator. Building a reel from it double-bakes the text (the reel
    # overlays the same words on top), producing a ghosted double exposure.
    # Prefer the matching _clean.jpg, which is the bare photo.
    clean = re.sub(r"^(.*/post_\d+_\d+)\.jpg$", r"\1_clean.jpg", image_path)
    if os.path.exists(clean):
        print("  Using clean base (no baked-in text): " + os.path.basename(clean))
        sys.stdout.flush()
        image_path = clean
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


AUDIO_DIR = "audio"
# Reels are ~9s; these tracks are 70-200s, so the mix loops whatever it needs.
AUDIO_VOLUME = 0.18


def pick_audio_track(index=0):
    """Return a royalty-free track from audio/, rotating through what exists."""
    if not os.path.isdir(AUDIO_DIR):
        return None
    tracks = sorted(
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith((".mp3", ".wav", ".ogg", ".m4a", ".flac"))
    )
    if not tracks:
        return None
    return os.path.join(AUDIO_DIR, tracks[index % len(tracks)])


def mix_audio(video_path, audio_path, duration, volume=AUDIO_VOLUME):
    """Loop/trim a music track under the video and write the final reel.

    Reels posted silent get throttled on Instagram Reels and YouTube Shorts, so
    the track is mixed low (18%) under the video rather than replacing it.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-stream_loop", "-1", "-i", audio_path,
        "-filter_complex",
        "[1:a]volume=" + str(volume) + ",afade=t=in:st=0:d=0.5,"
        "afade=t=out:st=" + str(max(0.0, duration - 1.0)) + ":d=1.0[a]",
        "-map", "0:v", "-map", "[a]",
        "-t", str(duration),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
        video_path + ".tmp.mp4",
    ]
    result = run_ffmpeg(cmd)
    if result.returncode != 0:
        print("Audio mix error: " + result.stderr[-200:])
        return False
    os.replace(video_path + ".tmp.mp4", video_path)
    return True


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
    parser.add_argument("--audio", action="store_true",
                        help="mix a royalty-free track from audio/ under the reel")
    parser.add_argument("--audio-index", type=int, default=0,
                        help="which track in audio/ to use (rotates)")
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
        size_kb = os.path.getsize(output_path) // 1024
        if args.audio:
            track = pick_audio_track(args.audio_index)
            if track:
                probe = run_ffmpeg(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", output_path]
                )
                dur = 0.0
                try:
                    dur = float((probe.stdout or "").strip())
                except ValueError:
                    dur = 0.0
                print("  Adding audio: " + os.path.basename(track))
                sys.stdout.flush()
                if mix_audio(output_path, track, dur or DURATION):
                    size_kb = os.path.getsize(output_path) // 1024
                    print("  Audio mixed at " + str(int(AUDIO_VOLUME * 100)) + "% volume")
                else:
                    print("  WARNING: audio mix failed, keeping silent reel")
            else:
                print("  WARNING: no tracks found in " + AUDIO_DIR + ", keeping silent reel")
        print("OK " + str(size_kb) + " KB")
    else:
        print("FAILED")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
