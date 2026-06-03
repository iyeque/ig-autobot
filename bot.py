import os
import sys
import time
import json
import uuid
import argparse
import re
import textwrap
import random
import numpy as np
import requests
import base64
from typing import Any, Dict, Optional, List
import PyPDF2
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from dotenv import load_dotenv
from pathlib import Path

try:
    import json_repair
except ImportError:
    json_repair = None  # type: ignore[misc, assignment]

# Try MoviePy imports at top level for consistency, with fallback
try:
    from moviepy.editor import VideoClip, AudioFileClip
except ImportError:
    VideoClip = None  # type: ignore
    AudioFileClip = None  # type: ignore

# Load .env file
dotenv_path = Path(__file__).parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from {dotenv_path}")

# Environment / config
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY", "")

CAPTION_FILE = "caption.txt"
# Function to generate timestamped filename in 'images' folder
def get_output_path(ext="png"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.getcwd(), "images", f"post_{timestamp}.{ext}")

MAX_BOOK_CONTEXT_CHARS = 2000

# Book-specific constants
BOOK_TITLE = os.environ.get("BOOK_TITLE", "The Nine Stitches")
BOOK_AUTHOR = os.environ.get("BOOK_AUTHOR", "M.W.E. Wigman")

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Phase 1 / Step 4: Brand consistency controls
BRAND_MODE = _env_flag("BRAND_MODE", True)
STATIC_TEXT_OVERLAY = _env_flag("STATIC_TEXT_OVERLAY", False)

# Global quality and feeling (Grounded and Cinematic)
BRAND_BASE = (
    "hyper-realistic cinematic photography, dramatic natural lighting, deep shadows, "
    "sharp textures, professional composition, moody atmosphere, 8k resolution"
)

# Pillar-specific palettes and styles for variety
PILLAR_AESTHETICS = {
    "nature_metaphor": "macro photography of weathered ancient stone and moss, deep forest greens and slate grey, sharp detail, wet textures",
    "systems_psychology": "expansive mountain landscape at blue hour, dramatic peaks, silver and deep indigo palette, vast atmospheric perspective",
    "micro_philosophy": "sunlight piercing through heavy storm clouds (God rays), dramatic high contrast, gold and charcoal color palette",
    "author_voice": "dark academic still life, old ink-stained mahogany, candlelight, deep shadows, rich espresso and amber tones",
    "quote": "minimalist architectural nature, a single leaf on dark water, ripples, high contrast, monochromatic depth"
}

BRAND_SUFFIX = (
    "no humans, no faces, no text, hyper-realistic detail, cohesive color palette, 8k resolution, sharp focus"
)

# Brand-safe variations (replaces noisy/random wide modifiers)
BRAND_MODIFIERS = [
    "dramatic side-lighting",
    "sharp macro focus",
    "golden hour highlights",
    "cinematic fog and light rays",
    "intricate crystalline structures",
    "deep volcanic sand textures",
    "moody storm-light",
    "polished obsidian reflections",
]

GENERIC_MODIFIERS = [
    "macro photography, extreme detail",
    "wide angle landscape, atmospheric perspective",
    "representational nature, realistic lighting",
    "minimalist composition, high contrast",
    "soft focus, cinematic bokeh",
    "crisp textures, sharp focus",
]


def _write_output_jpg(src_path: str, out_path: str = "output.jpg") -> str:
    """Normalizes image to 1080x1350 JPEG for Instagram."""
    try:
        from PIL import Image
        img = Image.open(src_path).convert("RGB")
        target_w, target_h = 1080, 1350
        src_w, src_h = img.size
        
        # Calculate scaling to cover the target area
        scale = max(target_w / src_w, target_h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        
        try:
            resample = Image.Resampling.BICUBIC
        except Exception:
            resample = 3 # Fallback for older Pillow
            
        img = img.resize((new_w, new_h), resample)
        
        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        img = img.crop((left, top, right, bottom))
        
        print(f"DEBUG: Saving image with dimensions: {img.size}")
        img.save(out_path, format="JPEG", quality=90, optimize=True)
        return out_path
    except Exception as e:
        print(f"PIL processing failed for {src_path}: {e}")
        try:
            if not out_path.endswith(".jpg") and not out_path.endswith(".jpeg"):
                out_path += ".jpg"
            with open(src_path, "rb") as r, open(out_path, "wb") as w:
                w.write(r.read())
            return out_path
        except Exception:
            return ""


def _fetch_ambient_music(output_path: str = "reel_audio.mp3") -> tuple[str, str]:
    """
    Checks local 'audio/' folder first, then falls back to curated reliable URLs.
    Returns (path, title).
    """
    # 1. Check local audio directory
    local_dir = "audio"
    if os.path.exists(local_dir):
        try:
            local_files = [f for f in os.listdir(local_dir) if f.lower().endswith('.mp3')]
            if local_files:
                chosen = random.choice(local_files)
                print(f"Using local audio: {chosen}")
                return os.path.join(local_dir, chosen), chosen.rsplit('.', 1)[0]
        except Exception as e:
            print(f"Local audio access error: {e}")

    # 2. Curated Reliable Source (Verified 2026 Direct Links)
    FALLBACKS = [
        ("https://incompetech.com/music/royalty-free/mp3-royalty-free/Healing.mp3", "Healing"),
        ("https://incompetech.com/music/royalty-free/mp3-royalty-free/Relaxing.mp3", "Relaxing"),
        ("https://incompetech.com/music/royalty-free/mp3-royalty-free/Meditation.mp3", "Meditation"),
        ("https://incompetech.com/music/royalty-free/mp3-royalty-free/Peaceful%20Ponder.mp3", "Peaceful Ponder"),
        ("https://incompetech.com/music/royalty-free/mp3-royalty-free/Morning%20Prayer.mp3", "Morning Prayer"),
        ("https://incompetech.com/music/royalty-free/mp3-royalty-free/Garden%20Music.mp3", "Garden Music"),
        ("https://www.no-copyright-music.com/wp-content/uploads/2021/09/DeeperMeaning.mp3", "Deeper Meaning"),
    ]
    
    random.shuffle(FALLBACKS)
    
    for url, title in FALLBACKS:
        try:
            print(f"Attempting to fetch ambient music: {title}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            r = requests.get(url, headers=headers, timeout=45)
            r.raise_for_status()
            
            # Basic validation: ensure it's actually an MP3 or at least not small HTML
            if len(r.content) < 50000: # Smaller than 50KB is likely an error page
                continue
                
            with open(output_path, "wb") as f:
                f.write(r.content)
            return output_path, title
        except Exception as e:
            print(f"Failed to download {title}: {e}")
            continue

    return "", ""


def _clean_caption_formatting(text: str) -> str:
    """
    Aggressively strips numbering (1., 1), labels (HOOK:, Insight:), 
    and Markdown artifacts from LLM output.
    """
    import re
    # Remove Markdown bold/italic
    text = text.replace("**", "").replace("*", "").replace("__", "").replace("_", "")
    
    # Comprehensive list of structural 'negative' words and AI-isms to purge
    structural_labels = (
        r"(HOOK|INSIGHT|TAKEAWAY|BODY|CAPTION|POST|BRIDGE|OUTRO|STEP\s*\d+|"
        r"CTA/CLOSING|CTA|CLOSING|PUNCHY|RELATABLE|HOOK LINE|EYE-CATCHING|"
        r"EMOTIONAL|CURIOSITY PULL|PUNCHY LINE|CAPTIVATING|TITLE|THEME|METAPHOR|"
        r"PUNCHY BODY LINE|FINAL CTA|VISUAL|DESCRIPTION|PROMPT)"
    )

    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        l = line.strip()
        if not l:
            cleaned_lines.append("")
            continue
            
        # Double Filter 1: Remove standalone label lines or lines that are just labels
        if re.fullmatch(rf"(?i){structural_labels}[:\s\-]*", l):
            continue

        # Recursive-style stripping for multiple prefixes
        while True:
            old_l = l
            # Remove numbering and common AI-style labels as prefixes
            l = re.sub(r"^\(?\d+[\.\)\:]\s*", "", l)
            l = re.sub(rf"(?i)^{structural_labels}[:\s\-]*", "", l)
            # Remove leading dashes or bullets
            l = re.sub(r"^[\-\•\*\+]\s*", "", l)
            if l == old_l:
                break
        
        if l:
            cleaned_lines.append(l)
            
    # Double Filter 2: Final global scrub for any labels internal to the text
    final_text = "\n".join(cleaned_lines).strip()
    # Remove internal "Label: " artifacts but protect natural sentence structures
    final_text = re.sub(rf"(?i)\b{structural_labels}:\s*", "", final_text)
    
    # Final check for concatenated "CTA.HOOK" scenarios
    final_text = re.sub(rf"(?i)\.({structural_labels})", ". ", final_text)
    
    # Remove any line that starts with "HOOK", "BODY", etc.
    final_lines = []
    for line in final_text.splitlines():
        l = line.strip()
        if not l:
            final_lines.append("")
            continue
        if re.match(rf"(?i)^{structural_labels}[:\s\-]*", l):
            # If the line has more content after the label, keep the content
            l = re.sub(rf"(?i)^{structural_labels}[:\s\-]*", "", l).strip()
            if not l: continue
        final_lines.append(l)

    return "\n".join(final_lines).strip()


def generate_reel(image_path: str, text_overlay: str, output_path: str = "reel.mp4", duration_s: float = 8.0, is_custom_brand: bool = False) -> tuple[str, str]:
    """
    Create a professional Reel (1080x1920) with mirrored-blur background,
    cinematic 'slow-drift' zoom, and massive, high-impact animated text.
    """
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
    import textwrap

    if VideoClip is None or AudioFileClip is None:
        raise RuntimeError("MoviePy components not loaded correctly. Video generation failed.")

    audio_file, audio_title = _fetch_ambient_music("reel_audio.mp3")
    
    W, H = 1080, 1920
    fps = 30
    duration_s = float(max(6.0, min(10.0, duration_s)))
    base = Image.open(image_path).convert("RGB")

    def _load_font(size: int):
        # Professional font search with Linux fallbacks for GitHub Actions
        paths = [
            "DejaVuSans-Bold.ttf", 
            "Arial Bold.ttf", 
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf"
        ]
        for name in paths:
            try: return ImageFont.truetype(name, size=size)
            except: continue
        # Massive fallback if no font found: PIL default is too small, 
        # but we can't do much without a file. 
        # We'll at least warn or try a generic name.
        try: return ImageFont.truetype("arial.ttf", size=size)
        except: return ImageFont.load_default()

    # MASSIVE FONTS for YouTube/Reel impact - slightly reduced for better fit
    font_main = _load_font(75)
    font_sub = _load_font(48)
    font_cta = _load_font(38)

    overlay = (text_overlay or "").strip().replace("\n", " ")
    if len(overlay) > 110: overlay = overlay[:107] + "..."
    # Wider wrap for impact - reduced to 18 to avoid edge cropping
    text_lines = textwrap.wrap(overlay.upper(), width=18) if overlay else []

    # Cinematic vignette mask
    vignette = Image.new("L", (W, H), 255)
    v_draw = ImageDraw.Draw(vignette)
    for i in range(550):
        alpha = int(255 * (i / 550)**1.3)
        v_draw.ellipse([i-60, i-60, W-i+60, H-i+60], outline=255-alpha)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=60))

    def _compose_frame(t: float) -> np.ndarray:
        # 1. Background
        bg = base.copy()
        bg_scale = W / bg.width
        bg = bg.resize((W, int(bg.height * bg_scale)), Image.Resampling.LANCZOS)
        bg = bg.crop((0, (bg.height - H) // 2, W, (bg.height + H) // 2))
        bg = bg.filter(ImageFilter.GaussianBlur(radius=70))
        
        # 2. Main Image: Cinematic Drift
        zoom = 1.02 + 0.12 * (t / duration_s)
        fg_w, fg_h = 1080, 1350
        fg = base.copy()
        f_scale = fg_w / fg.width
        fg = fg.resize((int(fg.width * f_scale * zoom), int(fg.height * f_scale * zoom)), Image.Resampling.LANCZOS)
        
        drift_x = int(20 * np.sin(t * 0.4))
        l, top = (fg.width - fg_w) // 2 + drift_x, (fg.height - fg_h) // 2
        fg = fg.crop((l, top, l + fg_w, top + fg_h))
        
        y_offset = (H - fg_h) // 2
        bg.paste(fg, (0, y_offset))
        
        # 3. Grade
        black = Image.new("RGB", (W, H), (5, 8, 12))
        bg = Image.composite(bg, black, vignette)
        
        # 4. Light Atmosphere & Pattern Interrupt (3s Flash)
        leak = Image.new("RGBA", (W, H), (0,0,0,0))
        ldraw = ImageDraw.Draw(leak)
        pulse = 0.5 + 0.5 * np.sin(t * 0.7)
        
        # Standard ambient leak
        ldraw.ellipse([-300, -300, 700, 700], fill=(255, 230, 200, int(40 * pulse)))
        
        # --- PATTERN INTERRUPT AT 3 SECONDS ---
        # A quick high-contrast flash to reset viewer attention
        if 3.0 <= t <= 3.3:
            flash_intensity = int(100 * np.sin((t - 3.0) * np.pi / 0.3))
            ldraw.rectangle([0, 0, W, H], fill=(255, 255, 255, flash_intensity))
            
        bg.paste(leak, (0,0), leak)

        # 5. Animated Massive Text
        if text_lines:
            draw = ImageDraw.Draw(bg)
            line_height = 120
            # Centered layout: start y based on total height of block
            total_text_h = len(text_lines) * line_height
            start_y = (H - total_text_h) // 2
            
            for i, line in enumerate(text_lines):
                line_start = 0.5 + i * 0.4
                line_alpha = max(0, min(1, (t - line_start) / 0.7))
                if line_alpha <= 0: continue
                
                current_font = font_main
                lw = draw.textlength(line, font=current_font)
                
                lx, ly = (W - lw) // 2, start_y + i * line_height
                
                # High-contrast backing plate - now more transparent (120 alpha)
                plate_pad = 40
                plate = Image.new("RGBA", (int(lw + plate_pad*2), int(line_height - 20)), (0, 0, 0, int(120 * line_alpha)))
                bg.paste(plate, (int(lx - plate_pad), int(ly)), plate)
                
                draw.text((lx, ly), line, font=current_font, fill=(255, 255, 255, int(255 * line_alpha)))
            
        # 6. Footer
        footer_alpha = max(0, min(1, (t - (duration_s * 0.7)) / 0.8))
        if footer_alpha > 0:
            draw = ImageDraw.Draw(bg)
            if not is_custom_brand:
                footer_text = f"\"{BOOK_TITLE.upper()}\""
                author_text = f"by {BOOK_AUTHOR}"
                draw.text((W//2, H - 280), footer_text, font=font_sub, fill=(224, 205, 156, int(255 * footer_alpha)), anchor="mm")
                draw.text((W//2, H - 210), author_text, font=font_cta, fill=(200, 200, 200, int(200 * footer_alpha)), anchor="mm")
            
            pill_w, pill_h = 400, 90
            px, py = (W - pill_w) // 2, H - 150
            pill = Image.new("RGBA", (pill_w, pill_h), (0, 0, 0, 0))
            pdraw = ImageDraw.Draw(pill)
            pdraw.rounded_rectangle((0, 0, pill_w, pill_h), radius=45, fill=(224, 205, 156, int(200 * footer_alpha)))
            bg.paste(pill, (px, py), pill)
            draw.text((W//2, H - 105), "LINK IN BIO", font=font_cta, fill=(15, 24, 36, int(255 * footer_alpha)), anchor="mm")

        return np.array(bg)

    if VideoClip is None or AudioFileClip is None:
        raise RuntimeError("MoviePy components not loaded correctly. Video generation failed.")

    clip = VideoClip(_compose_frame, duration=duration_s)
    
    if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1000:
        try:
            audio = AudioFileClip(audio_file)
            if audio.duration > duration_s:
                start = random.uniform(0, audio.duration - duration_s)
                audio = audio.subclip(start, start + duration_s)
            else:
                audio = audio.set_duration(duration_s)
            
            try:
                if hasattr(audio, 'audio_fadeout'): 
                    audio = audio.audio_fadeout(1.5) # type: ignore
                else:
                    from moviepy.audio.fx.all import audio_fadeout # type: ignore
                    audio = audio_fadeout(audio, 1.5) # type: ignore
            except: pass
            
            clip = clip.set_audio(audio)
            print(f"✓ Audio attached to Reel.")
        except Exception as e:
            print(f"⚠ Audio attachment error: {e}")

    clip.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio=True,
        audio_codec="aac",
        temp_audiofile='temp-audio.m4a',
        remove_temp=True,
        logger=None
    )
    
    return output_path, audio_title


def add_logo_watermark(image_path: str, logo_path: str) -> str:
    """Adds the brand logo to the top right of the image."""
    try:
        from PIL import Image
        if not os.path.exists(logo_path):
            print(f"Logo skipped: {logo_path} not found.")
            return image_path
            
        img = Image.open(image_path).convert("RGBA")
        logo = Image.open(logo_path).convert("RGBA")
        
        # Resize logo to a reasonable size (e.g., 20% of image width)
        w, h = img.size
        logo_w = int(w * 0.18)
        logo_h = int(logo.height * (logo_w / logo.width))
        logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
        
        # Position: Top Right with padding
        padding = 40
        pos = (w - logo_w - padding, padding)
        
        # Composite
        img.paste(logo, pos, logo)
        img.convert("RGB").save(image_path, quality=95)
        return image_path
    except Exception as e:
        print(f"Logo watermark failed: {e}")
        return image_path

def add_static_text_overlay(image_path: str, text_overlay: str) -> str:
    """
    Bold, massive high-legibility text overlay for maximum impact.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        import textwrap
    except Exception as e:
        print(f"Static text overlay skipped (missing deps): {e}")
        return image_path

    overlay = (text_overlay or "").strip().replace("\n", " ")
    if not overlay:
        return image_path

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    def _load_font(size: int):
        # Professional font search for Windows and Linux (GitHub Actions)
        paths = [
            "DejaVuSans-Bold.ttf",
            "Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf", 
            "C:/Windows/Fonts/segoeuib.ttf",
            "Arial Bold.ttf"
        ]
        for path in paths:
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
        # If all fail, try to at least get a decent size even with default
        return ImageFont.load_default()

    # REFINED BOLD FONT for modern balance - reduced for better fit
    font_size = 75 if len(overlay) < 25 else 60
    font = _load_font(font_size)
    
    # Wrap text to be punchy but wider to save vertical space
    wrapped = "\n".join(textwrap.wrap(overlay.upper(), width=20))
    
    # Calculate text dimensions
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=20, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Sleeker padding
    pad_x, pad_y = 60, 40
    box_w = min(w - 80, tw + pad_x * 2)
    box_h = th + pad_y * 2
    
    # Center the box vertically and horizontally (higher center)
    box_x = int((w - box_w) // 2)
    box_y = int((h - box_h) // 2 - (h * 0.05)) # Shifted 5% higher

    # Create a semi-transparent sophisticated box
    overlay_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay_layer)
    
    # Sophisticated semi-transparent black (150 alpha for better contrast)
    odraw.rectangle((box_x, box_y, box_x + box_w, box_y + box_h), fill=(0, 0, 0, 150))

    img = Image.alpha_composite(img.convert("RGBA"), overlay_layer).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Center text strictly
    tx = (w - tw) // 2
    ty = box_y + pad_y
    
    draw.multiline_text((tx, ty), wrapped, font=font, fill=(255, 255, 255), spacing=20, align="center")
    
    img.save(image_path, format="JPEG", quality=95, optimize=True)
    return image_path


def should_make_story(total_done: int, make_reel: bool) -> str:
    """
    Decide story type for this run.
    We always publish a story (post amplifier baseline), then elevate to
    higher-priority strategic types on schedule.
    """
    if make_reel:
        return "reel_amplifier"
    if total_done % 7 == 0:
        return "author_voice"
    if total_done % 3 == 0:
        return "book_cta"
    return "post_amplifier"


def extract_hook_text(caption_core: str, fallback_text: str = "") -> str:
    """
    Robustly extract a short hook from generated caption text.
    Falls back to post title/fallback when caption formatting drifts.
    """
    text = (caption_core or "").strip()
    if not text:
        return (fallback_text or "").strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidate = lines[0] if lines else text

    # If first line is too long, use first sentence as a safer hook.
    if len(candidate) > 90 and "." in candidate:
        candidate = candidate.split(".", 1)[0].strip()

    # Keep overlay-friendly length.
    if len(candidate) > 80:
        candidate = candidate[:77].rstrip() + "..."

    candidate = candidate.strip(" -–—\"'`*")
    if not candidate:
        candidate = (fallback_text or "").strip()
    return candidate.replace("**", "").replace("*", "")


def generate_story_image(base_image: str, story_type: str, hook_text: str = "", output_path: str = "story.jpg") -> str:
    """
    Generate a branded 1080x1920 story image from the current post image.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        import textwrap
    except Exception as e:
        raise RuntimeError(f"Story generation requires pillow: {e}") from e

    W, H = 1080, 1920
    img = Image.open(base_image).convert("RGB")

    # Branded soft background
    bg = img.resize((W, H), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(radius=16))
    bg = Image.blend(bg, Image.new("RGB", (W, H), (15, 24, 36)), alpha=0.28)

    # Foreground card
    fg_w, fg_h = 860, 1160
    scale = max(fg_w / img.width, fg_h / img.height)
    fg = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    left = (fg.width - fg_w) // 2
    top = (fg.height - fg_h) // 2
    fg = fg.crop((left, top, left + fg_w, top + fg_h))

    # Soft shadow behind card
    card_x = (W - fg_w) // 2
    card_y = 280
    shadow = Image.new("RGBA", (fg_w + 24, fg_h + 24), (0, 0, 0, 85))
    bg.paste(shadow, (card_x - 12, card_y - 8), shadow)
    bg.paste(fg, (card_x, card_y))

    def _font(size: int):
        for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(name, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    title_font = _font(68)
    body_font = _font(44)
    draw = ImageDraw.Draw(bg)

    hook = (hook_text or "").strip().replace("\n", " ")
    if len(hook) > 70:
        hook = hook[:67].rstrip() + "..."

    if story_type == "reel_amplifier":
        title = "New Reel"
        body = hook or "Watch this."
        footer = "Tap to watch"
    elif story_type == "book_cta":
        title = "From The Nine Stitches"
        body = hook or "This idea lives deeper in the book."
        footer = "Read The Nine Stitches - link in bio"
    elif story_type == "author_voice":
        title = "From the desk of M.W.E. Wigman"
        body = hook or "A quiet reflection from the writing desk."
        footer = "More in stories and posts"
    else:
        title = "New Post"
        body = hook or "This one hits deep."
        footer = "Tap to read"

    wrapped_body = "\n".join(textwrap.wrap(body, width=28))

    draw.text((80, 90), title, font=title_font, fill=(245, 246, 248))

    panel_w, panel_h = 880, 280
    panel_x, panel_y = (W - panel_w) // 2, 1480
    panel = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 120))
    pdraw = ImageDraw.Draw(panel)
    try:
        pdraw.rounded_rectangle((0, 0, panel_w, panel_h), radius=28, fill=(0, 0, 0, 120))
    except Exception:
        pdraw.rectangle((0, 0, panel_w, panel_h), fill=(0, 0, 0, 120))
    bg.paste(panel, (panel_x, panel_y), panel)

    draw.multiline_text((panel_x + 40, panel_y + 38), wrapped_body, font=body_font, fill=(250, 250, 250), spacing=10)
    draw.text((panel_x + 40, panel_y + 210), footer, font=_font(36), fill=(224, 205, 156))

    bg.save(output_path, format="JPEG", quality=92, optimize=True)
    return output_path

def sanitize_image_prompt(prompt: str, pillar: str = "") -> str:
    """
    Sanitize prompt for better AI generation success.
    Removes problematic terms, simplifies complex concepts.
    """
    replacements = {
        "human skin": "organic texture",
        "human body": "organic form",
        "human silhouette": "abstract form",
        "flesh": "organic matter",
        "bioluminescent phytoplankton": "glowing blue microorganisms in water",
        "blood": "crimson liquid",
        "corpse": "still form",
        "face": "surface",
        "person": "figure",
        "people": "figures",
        "man": "figure",
        "woman": "figure",
        "crack": "fracture",  # 'crack' often triggers NSFW filters for vessel-shaped objects
        "cracked": "fractured"
    }
    
    clean_prompt = (prompt or "").strip()
    for old, new in replacements.items():
        clean_prompt = clean_prompt.replace(old, new)
        clean_prompt = clean_prompt.replace(old.title(), new.title())

    # Normalize whitespace artifacts from model output
    clean_prompt = " ".join(clean_prompt.split())

    # Apply brand DNA envelope when BRAND_MODE is enabled.
    if BRAND_MODE:
        # Get pillar-specific aesthetic or fallback to a general one
        aesthetic = PILLAR_AESTHETICS.get(pillar, "minimalist abstract nature, cohesive colors")
        clean_prompt = f"{aesthetic}, {BRAND_BASE}, {clean_prompt}, {BRAND_SUFFIX}"

    # Keep prompt size bounded for stable API behavior.
    if len(clean_prompt) > 700:
        clean_prompt = clean_prompt[:697] + "..."

    return clean_prompt


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a given PDF file."""
    if not os.path.exists(pdf_path):
        print(f"Warning: The PDF file '{pdf_path}' does not exist.")
        return ""

    full_text = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    full_text.append(text)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    
    return "\n".join(full_text)


def extract_book_insights(text: str) -> Dict[str, Any]:
    """Extract key themes and structure from book for better context."""
    # Base insights
    insights = {
        "central_question": "What happens if you try to fail and succeed?",
        "epigraph": "To become, be calm. To be calm, pretend to be calm.",
        "chapters": [],
        "key_concepts": [
            "intention vs outcome", "productive failure", "adversity-growth cycles",
            "antifragility", "wabi-sabi", "kintsugi", "keystone species"
        ]
    }
    
    # Simple dynamic extraction logic
    if text:
        # Try to find common chapter patterns
        import re
        chapter_matches = re.findall(r"(?:Chapter|CHAPTER)\s+(\d+)\s*[:.-]?\s*(.*)", text[:10000])
        for num, title in chapter_matches[:5]:
            insights["chapters"].append({"number": int(num), "title": title.strip()})
            
    # Fallback if no chapters found
    if not insights["chapters"]:
        insights["chapters"] = [
            {"number": 1, "title": "The One in Time", "theme": "Intention vs. Outcome"},
            {"number": 2, "title": "If you can't evade it, embrace it", "theme": "Adversity and Growth"}
        ]
        
    return insights


# -------------------------
# Persistence helpers
# -------------------------
def _read_state() -> Dict[str, Any]:
    try:
        if os.path.exists("state.json"):
            with open("state.json", "r", encoding="utf-8") as f:
                state = json.load(f)
                
                # Migration: if used_ids is a list, move it to used_ids.instagram
                if isinstance(state.get("used_ids"), list):
                    old_used = state["used_ids"]
                    state["used_ids"] = {
                        "instagram": old_used,
                        "linkedin": list(old_used), # Copy existing to avoid immediate repeats on new platforms
                        "pinterest": list(old_used),
                        "youtube": list(old_used),
                        "threads": list(old_used),
                        "bluesky": list(old_used)
                    }
                # Ensure new platforms exist in used_ids if it's already a dict
                if isinstance(state.get("used_ids"), dict):
                    for p in ["youtube", "threads", "bluesky"]:
                        if p not in state["used_ids"]:
                            state["used_ids"][p] = []
                return state
    except Exception as e:
        print(f"Error reading state.json: {e}")
    
    return {
        "used_ids": {
            "instagram": [],
            "linkedin": [],
            "pinterest": [],
            "youtube": [],
            "threads": [],
            "bluesky": []
        },
        "last_cta": "",
        "cta_history": [],
        "last_hashtag_cluster": "",
        "last_hashtags": [],
        "last_pillar": "",
        "pillar_history": [],
    }


def _write_state(state: Dict[str, Any]) -> None:
    try:
        with open("state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Error writing state.json: {e}")


def _read_posts() -> List[Dict[str, Any]]:
    try:
        if os.path.exists("posts.json"):
            with open("posts.json", "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading posts.json: {e}")
    return []


def _write_posts(posts: List[Dict[str, Any]]) -> None:
    try:
        with open("posts.json", "w", encoding="utf-8") as f:
            json.dump(posts, f, indent=2)
    except Exception as e:
        print(f"Error writing posts.json: {e}")


# -------------------------
# Caption generation
# -------------------------
PILLAR_WEIGHTS: Dict[str, float] = {
    "micro_philosophy": 0.30,
    "nature_metaphor": 0.25,
    "systems_psychology": 0.20,
    "author_voice": 0.15,
    "quote": 0.10,
}
PILLAR_HISTORY_WINDOW = 10
CTA_HISTORY_WINDOW = 8

CTA_BY_CATEGORY: Dict[str, List[str]] = {
    "engagement": [
        "What part of this speaks to you.",
        "Tell me how this lands for you.",
        "I am curious what this brings up for you.",
    ],
    "save": [
        "Save this for later.",
        "Keep this close for the days you need it.",
    ],
    "share": [
        "Share this with someone who needs it.",
        "Someone in your circle needs this today.",
    ],
    "book": [
        f"If this resonates, read \"{BOOK_TITLE}\" - link in bio.",
        f"This idea lives deeper in my book \"{BOOK_TITLE}\".",
        f"If this moved you, you will find more in \"{BOOK_TITLE}\".",
    ],
}

# Balanced rotation target across categories.
CTA_CATEGORY_WEIGHTS: Dict[str, float] = {
    "engagement": 0.30,
    "save": 0.25,
    "share": 0.20,
    "book": 0.25,
}

HASHTAG_CLUSTERS: Dict[str, List[str]] = {
    # Cluster 1 — Micro-Philosophy
    "micro_philosophy": [
        "#TheNineStitches",
        "#PhilosophyDaily",
        "#ModernPhilosophy",
        "#DeepThoughtsDaily",
        "#MindsetShift",
        "#InnerWorkJourney",
        "#ThoughtfulLiving",
        "#LifePhilosophy",
        "#WisdomOfTheDay",
        "#ReflectiveThoughts",
        "#MindfulReflections",
        "#ExistentialThoughts",
        "#PhilosophyCommunity",
        "#DailyPhilosophy",
    ],
    # Cluster 2 — Nature Metaphors
    "nature_metaphor": [
        "#TheNineStitches",
        "#NatureMetaphor",
        "#NatureWisdom",
        "#LessonsFromNature",
        "#EcoPhilosophy",
        "#Bioluminescence",
        "#SystemsInNature",
        "#NaturePatterns",
        "#FractalNature",
        "#MacrocosmMicrocosm",
        "#NatureIsTeacher",
        "#NatureInspiredWisdom",
        "#EarthBasedPhilosophy",
    ],
    # Cluster 3 — Systems Psychology
    "systems_psychology": [
        "#TheNineStitches",
        "#SystemsThinking",
        "#CognitiveBias",
        "#MindsetScience",
        "#PsychologyDaily",
        "#InnerWorkJourney",
        "#SelfAwarenessPractice",
        "#EmotionalResilience",
        "#ShadowWorkJourney",
        "#BehaviorPatterns",
        "#ThoughtPatterns",
        "#MindsetGrowth",
        "#SelfUnderstanding",
    ],
    # Cluster 4 — Author Voice
    "author_voice": [
        "#TheNineStitches",
        "#AuthorLife",
        "#WritersJourney",
        "#IndieAuthorLife",
        "#WritersOfInstagram",
        "#WritingWisdom",
        "#CreativeProcess",
        "#BookWritingJourney",
        "#AuthorThoughts",
        "#BehindTheBook",
        "#WritersCommunity",
        "#WritingPhilosophy",
    ],
    # Cluster 5 — Quotes
    "quote": [
        "#TheNineStitches",
        "#QuoteOfTheDay",
        "#PhilosophyQuotes",
        "#MindsetQuotes",
        "#DeepQuotesDaily",
        "#BookQuotes",
        "#WisdomQuotes",
        "#ThoughtProvokingQuotes",
        "#LiteraryQuotes",
        "#ModernWisdom",
        "#DailyWisdom",
        "#QuoteCollectors",
    ],
}


def _choose_hashtags(state: Dict[str, Any], pillar: str, platform: str = "instagram") -> List[str]:
    """
    SEO-optimized tag selection:
    - Instagram: 3-5 high-impact tags, focusing on keywords in caption text.
    - Others: 8-12 tags as before.
    - Bluesky: No hashtags.
    """
    if platform.lower() == "bluesky":
        return []

    pillar_key = pillar if pillar in HASHTAG_CLUSTERS else "micro_philosophy"
    cluster = list(HASHTAG_CLUSTERS.get(pillar_key, HASHTAG_CLUSTERS["micro_philosophy"]))
    state["last_hashtag_cluster"] = pillar_key

    # Ensure book tag is present
    canonical_book = "#TheNineStitches"
    if canonical_book not in cluster:
        cluster.insert(0, canonical_book)
    
    # Determine count: SEO-optimized for tight platforms (3-5), standard for others (8-12)
    tight_platforms = ["instagram", "threads", "bluesky", "pinterest"]
    k = 4 if platform.lower() in tight_platforms else random.randint(8, 12)
    
    pool = [t for t in cluster if t != canonical_book]
    k = max(1, min(k, 1 + len(pool)))

    # Basic tag rotation to avoid identical sets
    sampled = random.sample(pool, k=max(0, k - 1))
    chosen = [canonical_book] + sampled
    
    state["last_hashtags"] = chosen
    return chosen


def _weighted_post_choice(posts: List[Dict[str, Any]], state: Dict[str, Any], platform: str = "instagram") -> Dict[str, Any]:
    """
    Phase 2 / Step 5:
    Weighted pillar selection + repetition protection.
    Platform-aware to allow different queues for IG vs LinkedIn.
    Handles active series progression.
    """
    if not posts:
        raise RuntimeError(f"No posts available for weighted selection on {platform}.")

    # 1. Handle Active Series Progression
    active_series = state.get("active_series", {}).get(platform)
    if active_series:
        s_name = active_series.get("name")
        next_part = active_series.get("next_part", 1)

        print(f"DEBUG: Active series for {platform}: {s_name}, looking for part {next_part}")
        # Look for the exact next part in available posts
        series_match = None
        for p in posts:
            if p.get("series") == s_name and p.get("part") == next_part:
                series_match = p
                break

        if series_match:
            print(f"Continuing series '{s_name}' — Part {next_part}")
            return series_match
        else:
            print(f"DEBUG: Available series tags in posts pool: {[p.get('series') for p in posts if p.get('series')]}")
            print(f"Series '{s_name}' completed or next part ({next_part}) missing. Clearing active series.")
            state["active_series"][platform] = None
    # 2. Randomly start a new series (20% chance if not already in one)
    # Check if any new series exist in the available posts pool
    if not state.get("active_series", {}).get(platform):
        new_series_candidates = [p.get("series") for p in posts if p.get("series") and p.get("part") == 1]
        if new_series_candidates and random.random() < 0.20:
            chosen_s = random.choice(new_series_candidates)
            for p in posts:
                if p.get("series") == chosen_s and p.get("part") == 1:
                    print(f"Starting new series: {chosen_s}")
                    state.setdefault("active_series", {}).setdefault(platform, {})
                    state["active_series"][platform] = {"name": chosen_s, "next_part": 1}
                    return p

    # 3. Standard Weighted Pillar Selection
    # Group available posts by pillar
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for p in posts:
        pillar = str(p.get("pillar", "micro_philosophy") or "micro_philosophy").strip()
        grouped.setdefault(pillar, []).append(p)

    # Maintain rolling history for soft quota correction (shared or platform-specific?)
    # We'll share pillar history across platforms to keep the brand voice consistent,
    # but use platform-specific used_ids (handled in main).
    history_raw = state.get("pillar_history", [])
    if not isinstance(history_raw, list):
        history_raw = []
    history: List[str] = [str(x) for x in history_raw if isinstance(x, str)]
    if len(history) > PILLAR_HISTORY_WINDOW:
        history = history[-PILLAR_HISTORY_WINDOW:]
        state["pillar_history"] = history

    # Only keep pillars that actually have available posts.
    candidates = [pillar for pillar in PILLAR_WEIGHTS.keys() if grouped.get(pillar)]
    if not candidates:
        # Fallback for unknown/missing pillar metadata
        return random.choice(posts)

    # Soft quota correction over recent history:
    # if a pillar is underrepresented in the last N posts, boost it;
    # if overrepresented, reduce it (but do not zero it out).
    history_counts: Dict[str, int] = {p: 0 for p in PILLAR_WEIGHTS.keys()}
    for p in history:
        if p in history_counts:
            history_counts[p] += 1
    window = max(1, min(PILLAR_HISTORY_WINDOW, len(history)))

    def _corrected_weight(pillar: str) -> float:
        base = PILLAR_WEIGHTS[pillar]
        if len(history) == 0:
            return base
        expected = base * window
        actual = history_counts.get(pillar, 0)
        delta = expected - actual
        # 0.55 .. 1.45 multiplier keeps correction gentle/stable
        factor = max(0.55, min(1.45, 1.0 + (delta / max(1.0, window))))
        return max(0.001, base * factor)

    weights = [_corrected_weight(p) for p in candidates]
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]

    chosen_pillar = random.choices(candidates, weights=weights, k=1)[0]

    # Repetition protection: avoid same pillar back-to-back when alternatives exist.
    last_pillar = str(state.get("last_pillar", "") or "").strip()
    if len(candidates) > 1 and chosen_pillar == last_pillar:
        alt_candidates = [p for p in candidates if p != last_pillar]
        alt_weights = [PILLAR_WEIGHTS[p] for p in alt_candidates]
        alt_total = sum(alt_weights) or 1.0
        alt_weights = [w / alt_total for w in alt_weights]
        chosen_pillar = random.choices(alt_candidates, weights=alt_weights, k=1)[0]

    chosen_post = random.choice(grouped[chosen_pillar])

    # Update rolling history now so state can be persisted by caller.
    history.append(chosen_pillar)
    state["pillar_history"] = history[-PILLAR_HISTORY_WINDOW:]

    return chosen_post


def _choose_next_cta(state: Dict[str, Any]) -> str:
    """
    Step 6 CTA module:
    - category-aware CTA rotation (engagement/save/share/book)
    - never repeat the same CTA twice in a row
    - soft balancing over recent CTA history
    """
    # Build flat list + reverse lookup
    all_items: List[Dict[str, str]] = []
    for category, ctas in CTA_BY_CATEGORY.items():
        for cta in ctas:
            all_items.append({"category": category, "text": cta})
    if not all_items:
        return ""

    last_cta = str(state.get("last_cta", "") or "").strip()

    # Parse recent history
    raw_hist = state.get("cta_history", [])
    if not isinstance(raw_hist, list):
        raw_hist = []
    history: List[str] = [str(x) for x in raw_hist if isinstance(x, str)]
    if len(history) > CTA_HISTORY_WINDOW:
        history = history[-CTA_HISTORY_WINDOW:]
        state["cta_history"] = history

    # Count categories in recent history
    cta_to_cat: Dict[str, str] = {item["text"]: item["category"] for item in all_items}
    cat_counts: Dict[str, int] = {k: 0 for k in CTA_BY_CATEGORY.keys()}
    for cta in history:
        cat = cta_to_cat.get(cta)
        if cat in cat_counts:
            cat_counts[cat] += 1

    window = max(1, min(CTA_HISTORY_WINDOW, len(history)))

    # Soft correction by category deficit/surplus.
    categories = [c for c in CTA_CATEGORY_WEIGHTS.keys() if CTA_BY_CATEGORY.get(c)]
    if not categories:
        categories = list(CTA_BY_CATEGORY.keys())
    if not categories:
        return random.choice([item["text"] for item in all_items if item["text"] != last_cta] or [all_items[0]["text"]])

    cat_weights: List[float] = []
    for cat in categories:
        base = CTA_CATEGORY_WEIGHTS.get(cat, 0.25)
        if len(history) == 0:
            cat_weights.append(base)
            continue
        expected = base * window
        actual = cat_counts.get(cat, 0)
        delta = expected - actual
        factor = max(0.60, min(1.40, 1.0 + (delta / max(1.0, window))))
        cat_weights.append(max(0.001, base * factor))

    total = sum(cat_weights) or 1.0
    cat_weights = [w / total for w in cat_weights]
    chosen_category = random.choices(categories, weights=cat_weights, k=1)[0]

    # Choose CTA text from category, avoiding immediate repetition.
    options = list(CTA_BY_CATEGORY.get(chosen_category, []))
    filtered = [c for c in options if c != last_cta]
    if filtered:
        chosen_cta = random.choice(filtered)
    else:
        # Fallback: choose from any category excluding last_cta
        global_options = [item["text"] for item in all_items if item["text"] != last_cta]
        chosen_cta = random.choice(global_options if global_options else [options[0]])

    # Persist CTA state
    state["last_cta"] = chosen_cta
    history.append(chosen_cta)
    state["cta_history"] = history[-CTA_HISTORY_WINDOW:]

    return chosen_cta


def _generate_text_ai_horde(prompt: str, system_prompt: str = "", max_tokens: int = 512) -> str:
    """Generates text using the AI Horde (KoboldCPP) API with fallback logic."""
    api_key = os.environ.get("AI_HORDE_API_KEY", "0000000000")
    submit_url = "https://aihorde.net/api/v2/generate/text/async"
    
    # Combine system prompt and user prompt for Kobold/Horde style
    # Many Horde models respond better to a structured 'Instruction' format.
    full_prompt = f"### Instruction:\n{system_prompt}\n\n### Input:\n{prompt}\n\n### Response:\n"
    
    payload = {
        "prompt": full_prompt,
        "params": {
            "n": 1,
            "max_context_length": 4096,
            "max_length": max_tokens,
            "rep_pen": 1.1,
            "temperature": 0.75,
            "top_p": 0.9,
        },
        "models": [
            "KoboldCPP/Llama-3-70B-Instruct", "Midnight Miqu 70B v1.5", 
            "Goliath 120b", "Euryale-L3-70B", "Llama-3-1-70B-Instruct",
            "aphrodite/TheDrummer/Cydonia-24B-v4.3", 
            "aphrodite/TheDrummer/Behemoth-X-123B-v2.1", 
            "aphrodite/TheDrummer/Skyfall-31B-v4.1",
            "koboldcpp/TheDrummer/Magidonia-24B-v4.3",
            "koboldcpp/Rocinante-X-12B-v1"
        ],
    }
    
    headers = {"apikey": api_key, "Content-Type": "application/json"}
    
    try:
        r = requests.post(submit_url, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        job_id = r.json().get("id")
        if not job_id:
            raise RuntimeError("AI Horde text-gen did not return a job ID")
            
        status_url = f"https://aihorde.net/api/v2/generate/text/status/{job_id}"
        
        # Poll for completion (up to 3 minutes for large models/queues)
        for attempt in range(36): 
            time.sleep(5)
            res = requests.get(status_url, timeout=30)
            data = res.json()
            
            if data.get("done"):
                generations = data.get("generations", [])
                if generations:
                    return generations[0].get("text", "").strip()
                raise RuntimeError("AI Horde text-gen returned 'done' but no content")
            
            if attempt % 6 == 0:
                print(f"  AI Horde (Text) status: {data.get('queue_position', 'unknown')} in queue...")
                
        raise RuntimeError("AI Horde text generation timed out")
    except Exception as e:
        print(f"  AI Horde text generation failed: {e}")
        raise


def _ai_verify_caption(caption: str, platform: str, max_chars: int) -> str:
    """
    Uses Cerebras (GPT-OSS 120B) as an Active Editor.
    Always returns a string: either the original, a fixed version, or a truncated fallback.
    """
    if not CEREBRAS_API_KEY:
        return caption if len(caption) <= (max_chars + 10) else caption[:max_chars-3] + "..."

    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
    
    check_prompt = f"""You are a social media editor for {platform.upper()}. 
Limit: {max_chars} characters.

Instruction:
1. Strip all AI meta-talk, apologies, and technical chatter.
2. Ensure the persona is witty, cynical, and philosophical (M.W.E. Wigman style).
3. If the text is over {max_chars} chars, summarize it to fit perfectly.
4. Output ONLY the final cleaned caption. No prefixes like "FIXED:" or "VALID:".

INPUT TEXT:
---
{caption}
---
"""

    try:
        payload = {
            "model": "gpt-oss-120b",
            "messages": [{"role": "system", "content": "You are a professional editor. Output only the final text."},
                         {"role": "user", "content": check_prompt}],
            "temperature": 0.1,
            "max_tokens": 512
        }
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        resp_data = r.json()
        
        # Robust structure validation
        if "choices" in resp_data and len(resp_data["choices"]) > 0:
            fixed = resp_data["choices"][0]["message"]["content"].strip()
            if fixed:
                print(f"  AI Editor processed the caption.")
                return fixed
        
        print(f"  AI Editor returned unexpected structure, using raw/truncated.")
        return caption if len(caption) <= max_chars else caption[:max_chars-3] + "..."
    except Exception as e:
        print(f"  AI Editor check failed: {e}")
        return caption if len(caption) <= max_chars else caption[:max_chars-3] + "..."


def generate_caption(caption_prompt: str, platform: str = "instagram", system_prompt: Optional[str] = None, book_context: str = "", book_insights: Optional[Dict] = None) -> str:
    """
    Generates a caption exclusively via AI Horde with an AI-driven verification loop.
    """
    # Platform-specific limits
    limits = {"bluesky": 200, "threads": 400, "instagram": 1800, "linkedin": 2500, "pinterest": 400, "youtube": 3500}
    max_chars = limits.get(platform.lower(), 1800)

    if not system_prompt:
        system_prompt = f"""You are the 'Professional Failure Expert' persona for {BOOK_AUTHOR}, author of {BOOK_TITLE}.
Your vibe: Witty, self-deprecating, and philosophical. Write RELATABLE, HUMOROUS, and slightly cynical captions.
Sound like a smart friend who just realized life is a chaotic simulation."""

    # Add formatting requirements
    full_system_content = system_prompt + f"""
Hard requirements for {platform.upper()}:
- TOTAL CHARACTER LIMIT: {max_chars} characters. YOU MUST NOT EXCEED THIS.
- Structure: 1 Hook line, 2-3 short Body lines, 1 CTA.
- No Markdown (** or __). No hashtags in body. No labels like 'HOOK:'.
- IF YOU EXCEED THE CHARACTER LIMIT, THE POST WILL FAIL. BE CONCISE.
Output only the caption text."""

    context_prompt = f"Context: {book_context}\n\nPrompt: {caption_prompt}" if book_context else caption_prompt

    # Exclusive AI Horde Loop (3 attempts)
    for attempt in range(3):
        print(f"Attempting AI Horde caption generation (Attempt {attempt+1}/3)...")
        try:
            raw_caption = _generate_text_ai_horde(context_prompt, system_prompt=full_system_content)
            if not raw_caption: continue
            
            # Step 1: Intelligent AI Verification/Repair
            result = _ai_verify_caption(raw_caption, platform, max_chars)
            
            if result:
                # If Critic returns a summary or the original text, it's approved
                print(f"✓ AI Critic approved {'(Original)' if result == raw_caption else '(Summarized)'} caption.")
                return _process_caption_output(result, target_platform=platform)
            else:
                print(f"⚠ AI Critic rejected output as JUNK. Retrying...")
                
        except Exception as e:
            print(f"⚠ AI Horde generation attempt failed: {e}")
            if attempt < 2: time.sleep(10)

    raise RuntimeError("Failed to generate a high-quality caption via AI Horde after all retries.")


def _process_caption_output(caption: str, target_platform: str = "instagram") -> str:
    """Final surgical cleanup of markdown, hashtags, and leading/trailing junk symbols."""
    # 1. Initial strip of common AI artifacts and brackets
    text = caption.strip().strip('{}[]"\' ')
    
    # 2. Remove markdown artifacts
    final = text.replace("**", "").replace("*", "").replace("__", "").replace("_", "")
    
    # 3. Filter lines for hashtags and meta-chatter
    lines = final.split('\n')
    cleaned_lines = []
    
    # Patterns for lines to skip (intro/outro meta-talk)
    skip_patterns = [
        r'^here we go', r'^---', r'^word count:', r'^character limit:',
        r"^i'll write", r"^i will write", r"^here is a caption", r"^sure, here",
        r"^following your instructions"
    ]
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
            
        # Skip accidental hashtags
        if stripped.startswith('#') and ' ' not in stripped:
            continue
            
        # Skip intro/outro meta-lines
        if any(re.search(pat, stripped, re.IGNORECASE) for pat in skip_patterns):
            continue
            
        cleaned_lines.append(line)
    
    # Final rejoin and strip any remaining boundary junk
    return "\n".join(cleaned_lines).strip().strip('{}[]"\' ')


def _strip_json_fences(content: str) -> str:
    """Remove markdown ``` fences."""
    text = content.strip()
    if not text.startswith("```"):
        return text
    i = 3
    while i < len(text) and text[i] in " \t":
        i += 1
    # Content starts immediately (``` [ or ``` {)
    if i < len(text) and text[i] not in "[{":
        nl = text.find("\n", i)
        if nl != -1:
            i = nl + 1
        else:
            scan_start = i
            while i < len(text) and text[i] not in "[{":
                i += 1
            # No bracket on same line: keep text after opening fence for errors / logs (do not return "")
            if i >= len(text):
                i = scan_start
    text = text[i:]
    text = text.strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _extract_json_array(content: str) -> str:
    start = content.find("[")
    end = content.rfind("]")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]
    return content


def _parse_posts_json_array(raw: str) -> List[Dict[str, Any]]:
    """Parse JSON array from LLM output; use json-repair when stdlib fails."""
    text = _strip_json_fences(raw)
    text = _extract_json_array(text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        if json_repair is None:
            raise RuntimeError(f"Invalid JSON and json-repair not installed: {e}") from e
        try:
            data = json_repair.loads(text)
        except Exception as e2:
            raise RuntimeError(f"Invalid JSON: {e}; json-repair failed: {e2}") from e2

    if not isinstance(data, list):
        raise RuntimeError("Expected a JSON array of post objects")
    return data


def _repair_posts_json_via_llm(broken_text: str) -> List[Dict[str, Any]]:
    """Ask the model to emit valid JSON only (last resort)."""
    if not CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY is not set")

    url = "https://api.cerebras.ai/v1/chat/completions"
    model_name = "gpt-oss-120b"
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json",
    }
    snippet = broken_text.strip()
    if len(snippet) > 14000:
        snippet = snippet[:14000] + "\n... [truncated]"
    fix_prompt = f"""The following text was supposed to be a JSON array of objects with keys:
"pillar", "title", "image_prompt", "caption_prompt".

It is INVALID JSON (often unescaped quotes inside strings).

Rewrite it as ONE valid JSON array only. Rules:
- Use double quotes for all keys and string values.
- Inside string values, do not use raw double quotes; use single quotes or rephrase.
- No markdown fences, no commentary, no text before or after the array.

Broken input:
{snippet}
"""
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You output only valid JSON arrays. No markdown.",
            },
            {"role": "user", "content": fix_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    if not data.get("choices"):
        raise RuntimeError(f"Cerebras repair returned no choices: {data}")
    content = data["choices"][0].get("message", {}).get("content", "").strip()
    return _parse_posts_json_array(content)


def _generate_new_posts() -> List[Dict[str, Any]]:
    """Generates a new list of post prompts using the Cerebras API with book awareness."""
    if not CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY is not set in the environment for prompt generation.")

    url = "https://api.cerebras.ai/v1/chat/completions"
    model_name = "gpt-oss-120b"

    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }

    meta_prompt = f"""
    You are an AI assistant for {BOOK_AUTHOR}, author of {BOOK_TITLE}.
    
    The book explores themes of:
    - Productive failure and the paradox "What happens if you try to fail and succeed?"
    - Intention vs. outcome (Chapter 1: The One in Time)
    - Adversity as growth catalyst (Chapter 2: If you can't evade it, embrace it)
    - Elegance of flaws, wabi-sabi, kintsugi (Chapter 3)
    - Microcosm/macrocosm, keystone species, butterfly effect (Chapter 4)
    
    Generate a list of 20 new social media post ideas. Each post must be a JSON object with:
    - "pillar": one of ["micro_philosophy", "nature_metaphor", "systems_psychology", "author_voice", "quote"]
    - "title": short, evocative phrase referencing specific book concepts
    - "image_prompt": detailed description for AI image generation (avoid human figures, use abstract/nature imagery)
    - "caption_prompt": detailed instruction mentioning specific book concepts, ending with question and #{BOOK_TITLE.replace(' ', '')} hashtag
    
    CRITICAL: Every post MUST have a unique title. Do not repeat the same concepts (like 'The Art of Imperfection') in multiple items.
    
    JSON RULES (required for valid output):
    - Return ONLY a JSON array of 20 objects. No markdown, no commentary.
    - Do not put double-quote characters inside title, image_prompt, or caption_prompt. Use single quotes or paraphrase instead.
    - No trailing commas. Escape backslashes in strings as \\\\.

    Return ONLY a valid JSON list of 20 objects, no other text.
    """

    last_error: Optional[BaseException] = None

    for attempt in range(3):
        temperature = (0.75, 0.5, 0.35)[attempt]
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are a creative assistant that outputs ONLY valid JSON arrays for {BOOK_TITLE} automation bot. "
                        "Never use double quotes inside JSON string values."
                    ),
                },
                {"role": "user", "content": meta_prompt},
            ],
            "temperature": temperature,
            "max_tokens": 3500,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()

            if not data.get("choices"):
                last_error = RuntimeError(f"Cerebras returned no choices: {data}")
                print(f"Attempt {attempt + 1}/3: {last_error}")
                continue

            content = data["choices"][0].get("message", {}).get("content", "").strip()
            if not content:
                last_error = RuntimeError("Empty content from Cerebras")
                print(f"Attempt {attempt + 1}/3: {last_error}")
                continue

            try:
                new_posts = _parse_posts_json_array(content)
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1}/3 JSON parse failed: {e}")
                try:
                    new_posts = _repair_posts_json_via_llm(content)
                except Exception as repair_e:
                    last_error = repair_e
                    print(f"Attempt {attempt + 1}/3 repair call failed: {repair_e}")
                    continue

            if len(new_posts) > 0:
                print(f"Successfully generated {len(new_posts)} new posts.")
                return new_posts

            last_error = RuntimeError("Parsed list was empty")
            print(f"Attempt {attempt + 1}/3: empty list")

        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"Attempt {attempt + 1}/3 HTTP error: {e}")
            time.sleep(3)

    raise RuntimeError(f"Failed to generate new posts after retries. Last error: {last_error}")

def _is_image_censored(image_path: str) -> bool:
    """Checks if an image contains explicit censorship messages using OCR.space API."""
    if not OCR_SPACE_API_KEY:
        print("Warning: OCR_SPACE_API_KEY is not set. Skipping censorship check.")
        return False

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        if len(image_data) < 5000:
            print(f"Image {image_path} is too small, likely an error.")
            return True

        headers = {"apikey": OCR_SPACE_API_KEY}
        payload = {"OCREngine": 2, "scale": True}
        files = {"file": ("image.jpg", image_data, "image/jpeg")}

        response = requests.post("https://api.ocr.space/parse/image",
                                 headers=headers,
                                 data=payload,
                                 files=files,
                                 timeout=60)
        response.raise_for_status()
        result = response.json()

        parsed_text = ""
        if result.get("ParsedResults"):
            for pr in result["ParsedResults"]:
                if pr.get("ParsedText"):
                    parsed_text += pr["ParsedText"] + " "
        
        parsed_text = parsed_text.lower()
        if any(kw in parsed_text for kw in ["censored", "nsfw content detected", "blocked by client", "detected and the client"]):
            print(f"Censorship text detected in {image_path}")
            return True

    except Exception as e:
        print(f"OCR check failed: {e}. Assuming censored for safety (will retry).")
        return True 
    
    return False


def _generate_image_ai_horde(prompt: str) -> str:
    """Generates a high-quality cinematic image using the AI Horde API with SDXL models."""
    url = "https://stablehorde.net/api/v2/generate/async"
    api_key = os.environ.get("AI_HORDE_API_KEY", "0000000000")
    
    clean_prompt = sanitize_image_prompt(prompt)
    
    eye_candy_mod = (
        "shot on 35mm lens, f/1.8, cinematic lighting, ultra-detailed textures, "
        "natural bokeh, professional color grading, Kodak Portra 400 aesthetic, "
        "sharp focus, 8k resolution, incredible depth of field"
    )
    final_prompt = f"{clean_prompt}, {eye_candy_mod}"
    print(f"AI Horde (SDXL) prompt: {final_prompt[:120]}...")

    payload = {
        "prompt": final_prompt,
        "params": {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.0,
            "width": 1024,
            "height": 1280, 
            "steps": 30,
        },
        "models": [
            "Juggernaut XL", "RealVisXL_V4.0", "AlbedoBase XL", 
            "DreamShaper XL", "Animagine XL", 
            "ICBINP XL", "SDXL 1.0"
        ],
        "nsfw": False,
        "censor_nsfw": True
    }
    
    headers = {"apikey": api_key, "Content-Type": "application/json"}
    
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    request_id = response.json().get("id")

    if not request_id:
        raise RuntimeError("AI Horde did not return a request ID")

    check_url = f"https://stablehorde.net/api/v2/generate/check/{request_id}"
    status_url = f"https://stablehorde.net/api/v2/generate/status/{request_id}"
    
    # 120 attempts * 30s = 3600s (1 hour) max wait for large queues
    for i in range(120): 
        time.sleep(30)
        status_response = requests.get(check_url, timeout=30)
        status_data = status_response.json()
        
        if status_data.get("done"):
            status_response = requests.get(status_url, timeout=30)
            full_status = status_response.json()
            generations = full_status.get("generations", [])
            
            if generations and generations[0].get("state") == "ok":
                # Success logic remains unchanged
                img_data = generations[0].get("img")
                if not isinstance(img_data, str) or not img_data.strip():
                    raise RuntimeError("AI Horde returned ok state but missing image payload")
                final_path = get_output_path(ext="png")
                
                if img_data.startswith("http"):
                    img_res = requests.get(img_data, timeout=120)
                    with open(final_path, "wb") as f:
                        f.write(img_res.content)
                else:
                    if "," in img_data: img_data = img_data.split(",")[1]
                    img_bytes = base64.b64decode(img_data)
                    with open(final_path, "wb") as f:
                        f.write(img_bytes)
                
                return final_path
        
        if i % 4 == 0:
            # Enhanced Diagnostics
            q_pos = status_data.get('queue_position', 'unknown')
            wait_est = status_data.get('wait_time', 'unknown')
            kudos = status_data.get('kudos', 'unknown')
            print(f"  AI Horde Status [Poll {i+1}]: Pos={q_pos} | Est={wait_est}s | Kudos={kudos}")
            
    raise RuntimeError("AI Horde generation timed out")


def generate_image(prompt: str) -> str:
    """Generate image with retries and censorship checks."""
    MAX_RETRIES = 2 
    for attempt in range(MAX_RETRIES):
        try:
            image_path = _generate_image_ai_horde(prompt)
            if _is_image_censored(image_path):
                print(f"Image attempt {attempt + 1} was censored. Retrying...")
                continue
            return image_path
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                print("Waiting 15s before next attempt...")
                time.sleep(15)
    raise RuntimeError("Failed to generate a valid image after retries.")


def generate_images_batch(prompt: str, n: int) -> List[str]:
    """Generates a batch of images with varied prompts, gracefully handling failures."""
    paths: List[str] = []
    modifiers = list(BRAND_MODIFIERS if BRAND_MODE else GENERIC_MODIFIERS)
    random.shuffle(modifiers)
    
    for i in range(n):
        mod = modifiers[i % len(modifiers)]
        varied_prompt = f"{prompt}, {mod}"
        print(f"Generating image {i+1}/{n} with variation: {mod}")
        try:
            p = generate_image(varied_prompt)
            paths.append(p)
        except Exception as e:
            print(f"Skipping image {i+1} due to repeated failures: {e}")
            
    if not paths:
        raise RuntimeError("Failed to generate ANY images in the batch.")
    return paths


# -------------------------
# Main flow
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="ig-autobot Creator")
    parser.add_argument("--platform", type=str, default="instagram", 
                      choices=["instagram", "linkedin", "pinterest", "youtube", "threads", "bluesky"],
                      help="Target platform for single-post mode")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "generate_all"],
                      help="Mode: single (legacy) or generate_all (unified content creation)")
    args = parser.parse_args()
    
    if args.mode == "generate_all":
        platforms = ["instagram", "linkedin", "pinterest", "youtube", "threads", "bluesky"]
        print(f"🚀 UNIFIED GENERATION MODE: Creating daily assets for {len(platforms)} platforms.")
    else:
        platforms = [args.platform]

    pdf_file_path = os.environ.get("PDF_BOOK_FILENAME", "The-Nine-Stitches.pdf")
    print(f"Using PDF: {pdf_file_path}")
    
    book_raw_text = extract_text_from_pdf(pdf_file_path)
    book_context = book_raw_text[:MAX_BOOK_CONTEXT_CHARS] if book_raw_text else ""
    book_insights = extract_book_insights(book_raw_text) if book_raw_text else None

    all_posts = _read_posts()
    state = _read_state()
    
    # In generate_all mode, we pick ONE post and use it for all platforms
    # We use 'instagram' as the primary queue reference for variety
    primary_platform = "instagram"
    platform_used_ids = set(state.get("used_ids", {}).get(primary_platform, []))
    
    available_posts = [p for p in all_posts if p.get("id") not in platform_used_ids]
    if not available_posts:
        print(f"Queue empty. Generating new batch...")
        new_posts = _generate_new_posts()
        max_id = max((post.get("id", 0) for post in all_posts), default=0)
        for i, post in enumerate(new_posts):
            post["id"] = max_id + i + 1
            all_posts.append(post)
        _write_posts(all_posts)
        available_posts = new_posts

    post = _weighted_post_choice(available_posts, state, platform=primary_platform)
    post_id = post.get("id")
    print(f"Selected post {post_id}: {post.get('title', 'Untitled')}")

    # --- GENERATE MEDIA ONCE ---
    try:
        # 1. Clean up old assets
        for f in ["captions_bundle.json", "post_story.flag", "post_reel.flag", "reel.mp4", "story.jpg", "output.jpg", "caption.txt"]:
            if os.path.exists(f): os.remove(f)

        # 2. Generate and normalize Master Image
        raw_path = generate_image(post["image_prompt"])
        processed_path = _write_output_jpg(raw_path, "output.jpg")
        print(f"✓ Master image generated: {processed_path}")

        # 3. Generate Master Reel (Used by YT and IG)
        # We use a generic hook for the video overlay
        hook_raw = generate_caption(post["caption_prompt"], platform="instagram", book_context=book_context)
        media_hook = extract_hook_text(_clean_caption_formatting(hook_raw))
        
        print("Generating Master Reel (6s)...")
        generate_reel("output.jpg", media_hook, "reel.mp4", duration_s=6.0)
        with open("post_reel.flag", "w") as f: f.write("Ambient Reflection")

        # 4. Generate Story Image (Used by IG and Pinterest)
        generate_story_image("output.jpg", "post_amplifier", media_hook, "story.jpg")
        with open("post_story.flag", "w") as f: f.write("post_amplifier")

    except Exception as e:
        print(f"Media generation failed: {e}")
        raise

    # --- GENERATE PLATFORM-SPECIFIC CAPTIONS ---
    bundle = {}
    for p in platforms:
        print(f"Generating caption for {p.upper()}...")
        try:
            # Re-use limits logic
            limits = {"bluesky": 180, "threads": 350, "instagram": 1800, "linkedin": 2500, "pinterest": 350, "youtube": 3500}
            hard_total_limits = {"bluesky": 300, "threads": 500, "pinterest": 500}
            
            raw = generate_caption(post["caption_prompt"], platform=p, book_context=book_context)
            core = _clean_caption_formatting(raw)
            
            # Assembly
            cta = _choose_next_cta(state)
            tags = _choose_hashtags(state, post.get("pillar", ""), platform=p)
            
            final_cap = core.strip()
            if cta: final_cap += "\n\n" + cta
            if tags: final_cap += "\n\n" + " ".join(tags)

            # Final Truncation
            limit = hard_total_limits.get(p.lower())
            if limit and len(final_cap) > limit:
                final_cap = final_cap[:limit-3] + "..."
            
            bundle[p] = final_cap
            
            # If single mode, also write the legacy caption.txt
            if args.mode == "single":
                with open(CAPTION_FILE, "w", encoding="utf-8") as f: f.write(final_cap)

        except Exception as e:
            print(f"Caption failed for {p}: {e}")

    # Save the bundle for publishing jobs
    with open("captions_bundle.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    # Update state for all platforms
    for p in platforms:
        if post_id not in state["used_ids"][p]:
            state["used_ids"][p].append(post_id)
    
    state["last_pillar"] = str(post.get("pillar", "micro_philosophy")).strip()
    _write_state(state)
    print("✓ All assets bundled. Ready for syndication.")


if __name__ == "__main__":
    main()
