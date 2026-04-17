import os
import sys
import time
import json
import uuid
import argparse

try:
    import json_repair
except ImportError:
    json_repair = None  # type: ignore[misc, assignment]
import requests
import random
from typing import Any, Dict, Optional, List
import PyPDF2
import base64
from datetime import datetime

from dotenv import load_dotenv
from pathlib import Path

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
    short = uuid.uuid4().hex[:6]
    return os.path.join(os.getcwd(), "images", f"{timestamp}_{short}.{ext}")

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
    
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        l = line.strip()
        if not l:
            cleaned_lines.append("")
            continue
            
        # Recursive-style stripping for multiple prefixes (e.g. "1. HOOK: text")
        while True:
            old_l = l
            # Remove numbering like "1.", "1)", "(1)", "Step 1:"
            l = re.sub(r"^\(?\d+[\.\)\:]\s*", "", l)
            # Remove common AI-style labels
            l = re.sub(r"(?i)^(HOOK|INSIGHT|TAKEAWAY|BODY|CAPTION|POST|BRIDGE|OUTRO|STEP\s*\d+):\s*", "", l)
            # Remove leading dashes or bullets
            l = re.sub(r"^[\-\•\*\+]\s*", "", l)
            if l == old_l:
                break
        
        if l:
            cleaned_lines.append(l)
            
    # Remove leading/trailing empty lines
    return "\n".join(cleaned_lines).strip()


def generate_reel(image_path: str, text_overlay: str, output_path: str = "reel.mp4", duration_s: float = 8.0) -> tuple[str, str]:
    """
    Create a professional Reel (1080x1920) with mirrored-blur background,
    cinematic 'slow-drift' zoom, and animated text with conversion focus.
    """
    try:
        import numpy as np
        try:
            from moviepy.video.VideoClip import VideoClip
            from moviepy.audio.io.AudioFileClip import AudioFileClip
        except ImportError:
            from moviepy import VideoClip, AudioFileClip
        from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
        import textwrap
    except Exception as e:
        raise RuntimeError(f"Reel generation requires moviepy and pillow. Error: {e}") from e

    audio_file, audio_title = _fetch_ambient_music("reel_audio.mp3")
    
    W, H = 1080, 1920
    fps = 30
    duration_s = float(max(6.0, min(10.0, duration_s)))
    base = Image.open(image_path).convert("RGB")

    def _load_font(size: int):
        for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf", "DejaVuSans.ttf"):
            try: return ImageFont.truetype(name, size=size)
            except: continue
        return ImageFont.load_default()

    font_main = _load_font(78)
    font_sub = _load_font(44)
    font_cta = _load_font(38)

    overlay = (text_overlay or "").strip().replace("\n", " ")
    if len(overlay) > 110: overlay = overlay[:107] + "..."
    text_lines = textwrap.wrap(overlay, width=20) if overlay else []

    # Cinematic vignette mask (refined)
    vignette = Image.new("L", (W, H), 255)
    v_draw = ImageDraw.Draw(vignette)
    for i in range(480):
        alpha = int(255 * (i / 480)**1.5)
        v_draw.ellipse([i-50, i-50, W-i+50, H-i+50], outline=255-alpha)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=50))

    def _compose_frame(t: float) -> np.ndarray:
        # 1. Mirrored Blur Background
        bg = base.copy()
        bg_scale = W / bg.width
        bg = bg.resize((W, int(bg.height * bg_scale)), Image.Resampling.LANCZOS)
        bg = bg.crop((0, (bg.height - H) // 2, W, (bg.height + H) // 2))
        bg = bg.filter(ImageFilter.GaussianBlur(radius=60))
        
        # 2. Main Image: Cinematic Slow-Drift Zoom
        # Starts at 1.05 and slowly zooms in to 1.15 while slightly shifting
        zoom = 1.05 + 0.10 * (t / duration_s)
        fg_w, fg_h = 1080, 1350
        fg = base.copy()
        f_scale = fg_w / fg.width
        fg = fg.resize((int(fg.width * f_scale * zoom), int(fg.height * f_scale * zoom)), Image.Resampling.LANCZOS)
        
        # Slight horizontal drift
        drift_x = int(15 * np.sin(t * 0.5))
        l, top = (fg.width - fg_w) // 2 + drift_x, (fg.height - fg_h) // 2
        fg = fg.crop((l, top, l + fg_w, top + fg_h))
        
        y_offset = (H - fg_h) // 2
        bg.paste(fg, (0, y_offset))
        
        # 3. Vignette & Color Grade (Slightly cooler/moodier for 2026)
        black = Image.new("RGB", (W, H), (10, 15, 25))
        bg = Image.composite(bg, black, vignette)
        
        # 4. Light Leak / Atmosphere (Animated)
        leak = Image.new("RGBA", (W, H), (0,0,0,0))
        ldraw = ImageDraw.Draw(leak)
        pulse = 0.5 + 0.5 * np.sin(t * 0.8)
        ldraw.ellipse([-200, -200, 600, 600], fill=(255, 200, 150, int(35 * pulse)))
        bg.paste(leak, (0,0), leak)

        # 5. Animated Text Overlay (Line by Line with higher contrast)
        if text_lines:
            draw = ImageDraw.Draw(bg)
            line_height = 100
            start_y = 350
            
            for i, line in enumerate(text_lines):
                # Staggered entry
                line_start = 0.8 + i * 0.5
                line_alpha = max(0, min(1, (t - line_start) / 0.6))
                if line_alpha <= 0: continue
                
                lw, lh = draw.textlength(line, font=font_main), line_height
                lx, ly = (W - lw) // 2, start_y + i * line_height
                
                # Dynamic text shadow/glow
                shadow = Image.new("RGBA", (int(lw + 60), int(lh + 30)), (0, 0, 0, int(180 * line_alpha)))
                bg.paste(shadow.filter(ImageFilter.GaussianBlur(5)), (int(lx - 30), int(ly - 10)), shadow)
                
                draw.text((lx, ly), line, font=font_main, fill=(255, 255, 255, int(255 * line_alpha)))
            
        # 6. Conversion Footer (Appears in second half of video)
        footer_alpha = max(0, min(1, (t - (duration_s * 0.6)) / 1.0))
        if footer_alpha > 0:
            draw = ImageDraw.Draw(bg)
            footer_text = f"\"{BOOK_TITLE}\""
            author_text = f"by {BOOK_AUTHOR}"
            cta_text = "Link in Bio"
            
            # Draw with alpha
            draw.text((W//2, H - 240), footer_text, font=font_sub, fill=(224, 205, 156, int(255 * footer_alpha)), anchor="mm")
            draw.text((W//2, H - 180), author_text, font=font_cta, fill=(200, 200, 200, int(200 * footer_alpha)), anchor="mm")
            
            # Subtle CTA pill
            pill_w, pill_h = 320, 70
            px, py = (W - pill_w) // 2, H - 130
            pill = Image.new("RGBA", (pill_w, pill_h), (0, 0, 0, 0))
            pdraw = ImageDraw.Draw(pill)
            pdraw.rounded_rectangle((0, 0, pill_w, pill_h), radius=35, fill=(224, 205, 156, int(180 * footer_alpha)))
            bg.paste(pill, (px, py), pill)
            draw.text((W//2, H - 95), cta_text, font=font_cta, fill=(15, 24, 36, int(255 * footer_alpha)), anchor="mm")

        return np.array(bg)

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
                if hasattr(audio, 'audio_fadeout'): audio = audio.audio_fadeout(1.5)
                else:
                    from moviepy.audio.fx.all import audio_fadeout
                    audio = audio_fadeout(audio, 1.5)
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


def add_static_text_overlay(image_path: str, text_overlay: str) -> str:
    """
    Optional static-image text overlay for higher save/share potential.
    Kept intentionally minimal and brand-consistent.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import textwrap
    except Exception as e:
        print(f"Static text overlay skipped (missing deps): {e}")
        return image_path

    overlay = (text_overlay or "").strip().replace("\n", " ")
    if not overlay:
        return image_path
    if len(overlay) > 80:
        overlay = overlay[:77].rstrip() + "..."

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    def _load_font(size: int):
        for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(name, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    font = _load_font(58)
    wrapped = "\n".join(textwrap.wrap(overlay, width=22))
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=8, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    w, h = img.size
    pad_x, pad_y = 34, 20
    box_w = min(w - 80, tw + pad_x * 2)
    box_h = th + pad_y * 2
    box_x = (w - box_w) // 2
    box_y = max(80, int(h * 0.14))

    panel = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(panel)
    try:
        pdraw.rounded_rectangle((0, 0, box_w, box_h), radius=20, fill=(0, 0, 0, 140))
    except Exception:
        pdraw.rectangle((0, 0, box_w, box_h), fill=(0, 0, 0, 140))
    img.paste(panel, (box_x, box_y), panel)

    tx = box_x + (box_w - tw) // 2
    ty = box_y + pad_y
    draw.multiline_text((tx + 2, ty + 2), wrapped, font=font, fill=(0, 0, 0), spacing=8, align="center")
    draw.multiline_text((tx, ty), wrapped, font=font, fill=(255, 255, 255), spacing=8, align="center")
    img.save(image_path, format="JPEG", quality=92, optimize=True)
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
                        "pinterest": list(old_used)
                    }
                return state
    except Exception as e:
        print(f"Error reading state.json: {e}")
    
    return {
        "used_ids": {
            "instagram": [],
            "linkedin": [],
            "pinterest": []
        },
        "last_cta_index": -1,
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


def _choose_hashtags(state: Dict[str, Any], pillar: str, k_min: int = 8, k_max: int = 12) -> List[str]:
    """
    Phase 1 Step 2:
    - 8–12 niche, rotating, rankable tags
    - pillar-aware clusters
    - avoid repeating the same cluster twice in a row
    - always include #TheNineStitches first
    """
    pillar_key = pillar if pillar in HASHTAG_CLUSTERS else "micro_philosophy"
    cluster = list(HASHTAG_CLUSTERS.get(pillar_key, HASHTAG_CLUSTERS["micro_philosophy"]))
    state["last_hashtag_cluster"] = pillar_key

    # Ensure book tag is present and first
    canonical_book = "#TheNineStitches"
    if canonical_book not in cluster:
        cluster.insert(0, canonical_book)
    pool = [t for t in cluster if t != canonical_book]

    # Keep count between 8 and 12, bounded by available tags.
    k = random.randint(k_min, k_max)
    k = max(1, min(k, 1 + len(pool)))

    # Above-and-beyond: avoid near-identical hashtag sets across consecutive posts.
    last_hashtags_raw = state.get("last_hashtags", [])
    last_hashtags = [str(x) for x in last_hashtags_raw] if isinstance(last_hashtags_raw, list) else []
    last_set = set(h.lower() for h in last_hashtags if isinstance(h, str))

    best_pick: List[str] = []
    lowest_overlap = 10**9
    attempts = 8
    for _ in range(attempts):
        sampled = random.sample(pool, k=max(0, k - 1))
        candidate = [canonical_book] + sampled
        overlap = len(set(h.lower() for h in candidate) & last_set)
        if overlap < lowest_overlap:
            lowest_overlap = overlap
            best_pick = candidate
        if overlap <= 2:
            break

    chosen = best_pick if best_pick else [canonical_book] + (random.sample(pool, k=max(0, k - 1)) if k > 1 else [])
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

    # Keep old key updated so existing automation remains compatible.
    state["last_cta_index"] = -1
    return chosen_cta


def generate_caption(caption_prompt: str, book_context: str = "", book_insights: Optional[Dict] = None) -> str:
    """Generates a short, hook-driven caption using the Cerebras API with book-aware context."""
    if not CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY is not set in the environment")

    url = "https://api.cerebras.ai/v1/chat/completions"
    model_name = "llama3.1-8b"

    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }

    system_content = f"""You are {BOOK_AUTHOR}, author of {BOOK_TITLE}.

Your book explores:
- The paradox of productive failure: "{book_insights['central_question'] if book_insights else 'What happens if you try to fail and succeed?'}"
- The epigraph: "{book_insights['epigraph'] if book_insights else 'To become, be calm. To be calm, pretend to be calm.'}"
- Chapter themes: Intention vs. Outcome, Adversity & Growth, Elegance of Flaws, Microcosm/Macrocosm
- Key concepts: wabi-sabi, kintsugi, antifragility, keystone species, serotinous cones, bioluminescence

Write Instagram captions that are short, emotionally relatable, and tuned for attention on the feed.

Hard requirements:
- Total length: 40–80 words (not counting hashtags we add later).
- Use this structure EVERY time (no labels, just the structure):
  1) HOOK: exactly 1 line (max ~10 words). Impactful emotional/curiosity pull.
  2) INSIGHT: 2–3 short lines inspired by the book themes and nature metaphors.
  3) TAKEAWAY: 1–2 short lines connecting to everyday life.
- Add line breaks for readability (one sentence per line is okay).
- Keep the philosophical, poetic, grounded tone — never academic.
- Weave in specific concepts or imagery from the book where it feels natural.
- Do NOT use any Markdown formatting (no asterisks ** or underscores __).
- Do NOT include hashtags.
- Do NOT include CTAs like 'save', 'share', 'comment', or 'link in bio'. (We append a rotating CTA ourselves.)

Output only the caption text."""

    full_prompt = caption_prompt
    if book_context:
        full_prompt = f"Using the following context from '{BOOK_TITLE}':\n\n```\n{book_context}\n```\n\n{caption_prompt}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 240
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        if data.get("choices") and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            caption = message.get("content", "").strip()
            if caption:
                print(f"Successfully generated caption with model {model_name}")
                
                # Hashtags are appended outside based on rotating clusters (Phase 1 Step 2).
                # We still strip any hashtags the model accidentally included.
                selected_hashtags: List[str] = []
                
                # Append hashtags if not already in caption
                caption_lines = caption.split('\n')
                caption_without_hashtags = []
                existing_hashtags = set()

                for line in caption_lines:
                    # Very simple check for lines that are solely hashtags
                    if line.strip().startswith('#') and ' ' not in line.strip():
                        existing_hashtags.add(line.strip().lower())
                    else:
                        caption_without_hashtags.append(line)
                
                final_caption = "\n".join(caption_without_hashtags).strip()
                final_caption = final_caption.replace("**", "").replace("*", "")
                
                return final_caption

        raise RuntimeError(f"Cerebras API returned an unexpected response format: {data}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling Cerebras API: {e}")
        raise RuntimeError(f"Failed to generate caption with Cerebras. Last error: {e}")


def _strip_json_fences(content: str) -> str:
    """Remove markdown ``` fences.

    Info strings after the opening fence may include hyphens, +, #, etc. (e.g. json-ld, c++, c#).
    Multiline: skip the entire first line after ```. Same-line JSON: skip chars until '[' or '{'.
    If there is no bracket on that line, keep the remainder (after the opening fence) so callers
    can surface malformed LLM output instead of an empty string.
    """
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
    model_name = "llama3.1-8b"
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
    model_name = "llama3.1-8b"

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
    
    Generate a list of 20 new Instagram post ideas. Each post must be a JSON object with:
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
                        f"You are a creative assistant that outputs ONLY valid JSON arrays for {BOOK_TITLE} Instagram bot. "
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
        if any(kw in parsed_text for kw in ["censored", "nsfw content detected", "blocked by client"]):
            print(f"Censorship text detected in {image_path}")
            return True

    except Exception as e:
        print(f"OCR check failed: {e}")
    
    return False


def _generate_image_ai_horde(prompt: str) -> str:
    """Generates a high-quality cinematic image using the AI Horde API with SDXL models."""
    url = "https://stablehorde.net/api/v2/generate/async"
    api_key = os.environ.get("AI_HORDE_API_KEY", "0000000000")
    
    clean_prompt = sanitize_image_prompt(prompt)
    
    # Advanced photographic technicals for 'Eye Candy' appeal
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
            "height": 1280, # Closer to 4:5 for SDXL optimization
            "steps": 30,
        },
        # Specifically targeting high-end photorealistic models
        "models": [
            "Juggernaut XL", "RealVisXL_V4.0", "AlbedoBase XL", 
            "DreamShaper XL", "Animagine XL", "Pony Diffusion XL", 
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
    
    for i in range(40): # ~6.5 minutes
        time.sleep(10)
        status_response = requests.get(check_url, timeout=30)
        status_data = status_response.json()
        
        if status_data.get("done"):
            status_response = requests.get(status_url, timeout=30)
            full_status = status_response.json()
            generations = full_status.get("generations", [])
            
            if generations and generations[0].get("state") == "ok":
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
        
        if i % 6 == 0:
            print(f"Polling AI Horde... {i+1}")
            
    raise RuntimeError("AI Horde generation timed out")


def generate_image(prompt: str) -> str:
    """Generate image with retries and censorship checks."""
    MAX_RETRIES = 5 # Increased retries
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
    parser.add_argument("--platform", type=str, default="instagram", choices=["instagram", "linkedin", "pinterest"],
                      help="Target platform (determines post queue and scheduling logic)")
    args = parser.parse_args()
    platform = args.platform

    pdf_file_path = os.environ.get("PDF_BOOK_FILENAME", "The-Nine-Stitches.pdf")
    print(f"Running for platform: {platform}")
    print(f"Using PDF: {pdf_file_path}")
    print(f"Brand mode: {'ON' if BRAND_MODE else 'OFF'} | Static overlay: {'ON' if STATIC_TEXT_OVERLAY else 'OFF'}")
    
    book_raw_text = extract_text_from_pdf(pdf_file_path)
    book_context = book_raw_text[:MAX_BOOK_CONTEXT_CHARS] if book_raw_text else ""
    book_insights = extract_book_insights(book_raw_text) if book_raw_text else None

    all_posts = _read_posts()
    state = _read_state()
    
    # Platform-specific queue management
    platform_used_ids = set(state.get("used_ids", {}).get(platform, []))
    
    # Map used IDs to their titles for title-based filtering
    used_titles = set()
    for p in all_posts:
        if p.get("id") in platform_used_ids:
            t = p.get("title", "").strip().lower()
            if t: used_titles.add(t)

    # Available posts must have unique ID AND unique title for THIS platform
    available_posts = []
    for p in all_posts:
        p_id = p.get("id")
        p_title = p.get("title", "").strip().lower()
        if p_id not in platform_used_ids and p_title not in used_titles:
            available_posts.append(p)
            # Add to used_titles so we don't pick two duplicates in the same batch
            used_titles.add(p_title)

    if not available_posts:
        print(f"All unique posts used for {platform}. Generating new batch...")
        new_posts = _generate_new_posts()
        max_id = max((post.get("id", 0) for post in all_posts), default=0)
        for i, post in enumerate(new_posts):
            post["id"] = max_id + i + 1
            all_posts.append(post)
        _write_posts(all_posts)
        available_posts = new_posts

    post = _weighted_post_choice(available_posts, state, platform=platform)
    post_id = post.get("id")
    print(f"Selected post {post_id} for {platform}: {post.get('title', 'Untitled')}")
    print(f"Selected pillar: {post.get('pillar', 'micro_philosophy')}")
    if state.get("pillar_history"):
        print(f"Recent pillar history: {state.get('pillar_history')}")

    # Series handling
    is_series = bool(post.get("series") and post.get("part"))
    series_part = post.get("part")
    series_title = post.get("title", "")

    # Generate caption
    try:
        # Choose a CTA that avoids repeating the last one, then
        # let the caption generator focus purely on hook + body.
        cta_text = _choose_next_cta(state)
        caption_raw = generate_caption(post["caption_prompt"], book_context, book_insights)
        
        # Aggressively clean numbering and labels
        caption_core = _clean_caption_formatting(caption_raw)
        
        if is_series:
            # Prepend Part X info to caption
            caption_core = f"Part {series_part} — {series_title}\n\n{caption_core}"
            
        hook_text = extract_hook_text(caption_core, str(post.get("title", "") or ""))
        # If series, ensure the overlay doesn't just repeat the "Part X" line if it became the hook
        if is_series and hook_text.startswith(f"Part {series_part}"):
             # Try to find the actual hook in the next lines of caption_core
             lines = [l.strip() for l in caption_core.splitlines() if l.strip()]
             if len(lines) > 1:
                 hook_text = extract_hook_text("\n".join(lines[1:]), series_title)

        pillar = str(post.get("pillar", "") or "").strip()
        hashtag_list = _choose_hashtags(state, pillar)

        caption = caption_core.strip()
        if cta_text:
            # Ensure clean line-break formatting:
            # body, blank line, CTA on its own line.
            caption += "\n\n" + cta_text

        if hashtag_list:
            caption += "\n\n" + " ".join(hashtag_list)

        with open(CAPTION_FILE, "w", encoding="utf-8") as f:
            f.write(caption)
    except Exception as e:
        print(f"Caption generation failed: {e}")
        raise

    # Generate image(s)
    try:
        # Clean up old flags to prevent 'Groundhog Day' repetitions
        for f in ["carousel.json", "post_story.flag", "post_reel.flag", "reel.mp4", "story.jpg"]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Cleaned up old {f}")

        total_done = len(state["used_ids"][platform]) + 1
        
        # Determine overlay text for media (Series aware)
        media_overlay = f"Part {series_part}: {hook_text}" if is_series else hook_text

        # Platform-specific logic for extra media
        if platform == "instagram":
            # Every 3rd post is a Reel, every 5th post is a carousel.
            # Stories run every post with typed scheduling logic.
            make_reel = (total_done % 3 == 0)
            make_carousel = (total_done % 5 == 0)
        else:
            # LinkedIn and Pinterest usually favor single high-quality visuals over reels/carousels for this bot's style.
            make_reel = False
            make_carousel = False

        story_type = should_make_story(total_done, make_reel)
        
        if make_carousel:
            count = 3
            print(f"Generating carousel with {count} images.")
            raw_images = generate_images_batch(post["image_prompt"], count)
            jpg_images = []
            
            for i, raw_p in enumerate(raw_images):
                # Ensure each image is normalized to JPG
                jpg_p = get_output_path(ext="jpg")
                processed = _write_output_jpg(raw_p, jpg_p)
                if processed:
                    jpg_images.append(os.path.relpath(processed, os.getcwd()).replace('\\', '/'))
            
            with open("carousel.json", "w", encoding="utf-8") as f:
                json.dump(jpg_images, f)
            
            # For backward compatibility / verify_outputs.py
            if jpg_images:
                import shutil
                shutil.copy(os.path.join(os.getcwd(), jpg_images[0]), "output.jpg")
                
            print(f"Carousel saved: {jpg_images}")
        else:
            raw_path = generate_image(post["image_prompt"])
            processed_path = _write_output_jpg(raw_path, "output.jpg")
            if STATIC_TEXT_OVERLAY and processed_path:
                add_static_text_overlay(processed_path, media_overlay or post.get("title", "") or "")
            print(f"Image saved and normalized: {processed_path}")

        # Story generation (Instagram specific baseline, but Pinterest could use it too)
        if platform in ["instagram", "pinterest"]:
            story_path = generate_story_image("output.jpg", story_type, media_overlay or post.get("title", "") or "", "story.jpg")
            if story_path and os.path.exists(story_path):
                with open("post_story.flag", "w", encoding="utf-8") as f:
                    f.write(story_type)
                print(f"Story/Pin-Vertical saved: {story_path} (type={story_type})")

        # Generate Reel (Instagram only)
        if make_reel and platform == "instagram":
            print("Generating Reel (6s, 1080x1920)...")
            reel_path, audio_title = generate_reel("output.jpg", media_overlay or post.get("title", "") or "", "reel.mp4", duration_s=6.0)
            if reel_path and os.path.exists(reel_path):
                with open("post_reel.flag", "w", encoding="utf-8") as f:
                    # Save the audio title so publish.py can use it
                    f.write(audio_title or "Ambient Reflection")
                print(f"Reel saved: {reel_path}")

        # Persist state only after caption + image/reel/story generation succeed.
        state["used_ids"][platform].append(post_id)
        state["last_pillar"] = str(post.get("pillar", "micro_philosophy") or "micro_philosophy").strip()
        
        # Increment series part if active
        if is_series and state.get("active_series", {}).get(platform):
            state["active_series"][platform]["next_part"] = series_part + 1
            
        _write_state(state)
            
    except Exception as e:
        print(f"Image generation failed: {e}")
        # Persist non-post-consumption rotation state (CTA/hashtags/history),
        # but do NOT mark this post as used when media generation fails.
        _write_state(state)
        raise

    print("✓ Done.")


if __name__ == "__main__":
    main()
