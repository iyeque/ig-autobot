import os
import sys
import time
import json
import shutil
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

def sanitize_image_prompt(prompt: str) -> str:
    return (prompt or "").strip()

def _read_posts() -> list[dict]:
    return []

def _write_posts(_posts: list[dict]) -> None:
    return None

def _read_state() -> dict:
    return {"content_queue": [], "used_ids": {}, "last_pillar": "micro_philosophy"}

def _write_state(_state: dict) -> None:
    return None

def _weighted_post_choice(_posts: list[dict], _state: dict, platform: str = "instagram") -> dict:
    return _posts[0] if _posts else {"id": 0, "pillar": "micro_philosophy", "title": "", "image_prompt": "", "caption_prompt": ""}

def _try_resume_pending(_state: dict, _platforms: list[str]) -> bool:
    return False

def extract_hook_text(_text: str) -> str:
    text = (_text or "").strip()
    if not text:
        return ""
    return text.splitlines()[0][:100]

def generate_story_image(_source: str, _prefix: str, _text: str, _out: str) -> str:
    return _out

def _editor_fallback(caption: str, platform: str, max_chars: int) -> str:
    text = caption.strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text.strip()


# Function to generate timestamped filename in 'images' folder
def get_output_path(ext="png"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.getcwd(), "images", f"post_{timestamp}.{ext}")

MAX_BOOK_CONTEXT_CHARS = 2000

# Book-specific constants
BOOK_TITLE = os.environ.get("BOOK_TITLE", "The Nine Stitches")
BOOK_AUTHOR = os.environ.get("BOOK_AUTHOR", "M.W.E. Wigman")
BOOK_URL = os.environ.get("BOOK_URL", "")

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




def _caption_is_incomplete(text: str) -> bool:
    """
    Heuristic: flag captions that appear to end mid-sentence or mid-clause
    rather than at a natural boundary.
    """
    t = text.strip()
    if not t:
        return True
    # If the last line is only a hashtag-like tail and the line before it doesn't
    # end with sentence punctuation, consider it potentially incomplete.
    lines = t.splitlines()
    last = lines[-1].strip()
    if last.startswith('#'):
        meaningful = lines[:-1]
        if not meaningful:
            return True
        # Hashtags appended but no final sentence boundary above them
        return not meaningful[-1].rstrip().endswith(('.', '!', '?', '…', ':', ';'))
    # Final punctuation checks
    if t[-1] in '.!?…:;':
        return False
    # Ends mid-word or mid-phrase
    if t[-1] in (' ', '\t'):
        return True
    return False


def _has_mid_sentence_break(text: str) -> bool:
    """
    Detect paragraphs that end mid-clause (weak words, prepositions, articles,
    conjunctions, trailing gerunds) which typically indicate an AI truncation
    rather than a deliberate line break.
    """
    weak_endings = (
        'a ', 'an ', 'the ', 'and ', 'but ', 'or ', 'nor ', 'yet ', 'so ',
        'in ', 'on ', 'at ', 'by ', 'for ', 'from ', 'to ', 'with ', 'without ',
        'within ', 'upon ', 'among ', 'between ', 'because ', 'since ', 'although ',
        'though ', 'while ', 'if ', 'unless ', 'until ', 'whereas ', 'that ',
        'which ', 'who ', 'whom ', 'whose ',
    )
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) < 2:
        return False
    # Flag any paragraph before the last one that ends with a weak word or phrase
    for idx, para in enumerate(paragraphs[:-1]):
        lower = para.lower()
        # Check if paragraph ends on a weak standalone word or segment
        if any(lower.endswith(we) for we in weak_endings):
            return True
        # Check if last line of paragraph is extremely short and doesn't end in strong punctuation
        last_line = para.splitlines()[-1].strip()
        if len(last_line) < 60 and not last_line.endswith(('.', '!', '?', '…', ':', ';', ',”')):
            return True
    return False


def _ai_verify_caption(caption: str, platform: str, max_chars: int) -> str:
    """
    Uses Cerebras (GPT-OSS 120B) as an Active Editor.
    Always returns a string: either the original, a fixed version, or a truncated fallback.
    For Bluesky/Threads, it rewrites the caption into a short, witty, self-contained version.
    """
    if not CEREBRAS_API_KEY:
        result = caption if len(caption) <= max_chars else caption[:max_chars-3] + "..."
        return result.strip()

    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
    
    if platform.lower() in ("bluesky", "threads"):
        check_prompt = f"""You are a careful social media editor for {platform.upper()}.
Hard limit: {max_chars} characters MAX for the BODY TEXT ONLY.

CRITICAL RULES:
1. Output ONLY the body text. No prefixes, no quotes, no "FIXED:", no markdown, no explanations.
2. EXECUTION ORDER: If the FIRST line starts with "Ah", "Ah yes", "Ah, what a", or any "Ah..." variation, DELETE that line and replace it with a punchy, specific opener. The rest of the body stays intact.
3. REWRITE the content into a SHORT, WITTY, and COMPLETE {platform.upper()} caption body.
4. Summarize or condense the original content into a self-contained mini-version.
5. Keep the same voice and tone (witty, slightly cynical, smart-friend vibe) but make it punchy.
6. Do NOT add a CTA, hashtags, or closing line. A CTA will be appended automatically after this body.
7. Do NOT exceed {max_chars} characters. Be concise. If you must choose, cut examples, not the core insight.

INPUT TEXT:
---
{caption}
---

OUTPUT THE BODY TEXT NOW:
"""
    elif platform.lower() == "linkedin":
        check_prompt = f"""You are a careful social media editor for {platform.upper()}.
Hard limit: {max_chars} characters MAX for the BODY TEXT ONLY.

Platform Editor Rules:
1. Output ONLY the body text. No prefixes, no quotes, no "FIXED:", no markdown, no explanations.
2. REWRITE the content for LinkedIn's feed behavior: first ~140 characters must be a sharp hook that invites clicks.
3. EXECUTION ORDER: Before any other edits, check the FIRST line. If it starts with "Ah", "Ah yes", "Ah, what a", or any lazy "Ah..." variation, DELETE that line and replace it with a concrete observation, a counterintuitive claim, or a specific real-world example. The rest of the body stays intact.
4. Tighten paragraphs: use short lines, avoid wall-of-text blocks, preserve white space.
5. Rewrite the body into a grounded, warm, evidence-based digital wellness/parent voice. Like a real parent sharing lived experience—never an author pitching a book. Strip any book titles, author mentions, "subtle nod" remnants, purchase plugs, or brand plugs that don't belong to Digital Guardian digital wellness.
6. If hashtags are present in the input, keep only 3-5 targeted tags within the body. Remove hashtag soup. Final hashtags will be appended automatically.
7. Ensure the body ends at a natural boundary. A closing question/CTA will be appended automatically after this body.
8. Do NOT add markdown. Do NOT write in all caps.
9. Do NOT exceed {max_chars} characters.
10. Output ONLY the final cleaned body. No prefixes like "FIXED:" or "VALID:".

INPUT TEXT:
---
{caption}
---

OUTPUT THE BODY TEXT NOW:
"""
    else:
        check_prompt = f"""You are a careful social media editor for {platform.upper()}.
Hard limit: {max_chars} characters MAX for the BODY TEXT ONLY.

Instruction:
1. Output ONLY the body text. No prefixes, no quotes, no "FIXED:", no markdown, no explanations.
2. IMPORTANT: If the FIRST line starts with "Ah", "Ah yes", "Ah, what a", or any "Ah..." variation, rewrite that first line into a sharp, specific hook. Do not keep the lazy opener.
3. Keep the existing voice and tone. Do not rewrite, summarize, or shorten the content beyond this single-line fix.
4. CTA and hashtags will be appended automatically after this body. Do NOT include them here.
5. If the body exceeds {max_chars} chars, ONLY trim the excess from the end at a natural sentence or line boundary. Do NOT cut mid-word or mid-sentence.
6. Output ONLY the final cleaned body. No prefixes like "FIXED:" or "VALID:".

INPUT TEXT:
---
{caption}
---

OUTPUT THE BODY TEXT NOW:
"""
    # Allocate more output room for longer platforms so the editor can
    # actually complete the caption instead of cutting off mid-sentence.
    if platform.lower() in ('linkedin', 'instagram', 'youtube', 'facebook'):
        caps_max_tokens = 1024
    else:
        caps_max_tokens = max(512, max_chars)
    payload = {
        "model": "gpt-oss-120b",
        "messages": [{"role": "system", "content": "You are a professional editor. Output only the final text."},
                     {"role": "user", "content": check_prompt}],
        "temperature": 0.1,
        "max_tokens": caps_max_tokens
    }

    def _call_editor(attempt: int) -> str:
        prefix = "  "
        if attempt > 0:
            prefix = f"  [Retry {attempt}] "
            # On retry add stronger completion guidance at the end of prompt
            check_prompt_ = (
                check_prompt.rstrip()
                + "\n\nFINAL CHECK: If your output ends mid-sentence or mid-clause, rewrite the final sentence so it completes naturally. Do NOT leave trailing fragments or ellipsis mid-thought."
            )
        else:
            check_prompt_ = check_prompt

        payload["messages"][1]["content"] = check_prompt_
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        resp_data = r.json()

        if "choices" not in resp_data or not resp_data["choices"]:
            msg = resp_data.get("error", {}).get("message", "")
            print(f"{prefix}AI Editor empty response: {msg[:120]}")
            return ""

        msg = resp_data["choices"][0].get("message") or {}
        fixed = msg.get("content", "").strip()
        if not fixed:
            print(f"{prefix}AI Editor returned empty content.")
            return ""

        print(f"{prefix}AI Editor processed the caption.")
        if len(fixed) > max_chars:
            print(f"{prefix}Editor exceeded limit; retrying with summarization.")
            check_prompt_ = (
                check_prompt.rstrip()
                + f"\n\nFINAL COMPRESSION: The previous output was too long. "
                + f"Rewrite this into a COMPLETE, COHESIVE caption under {max_chars} characters. "
                + "DO NOT truncate or add ellipsis. Summarize the content while preserving the voice, hook, body, and CTA."
            )
            payload["messages"][1]["content"] = check_prompt_
            r2 = requests.post(url, headers=headers, json=payload, timeout=25)
            resp_data2 = r2.json()
            if "choices" in resp_data2 and resp_data2["choices"]:
                msg2 = resp_data2["choices"][0].get("message") or {}
                fixed2 = msg2.get("content", "").strip()
                if fixed2 and len(fixed2) <= max_chars:
                    fixed = fixed2
                elif fixed2:
                    # Third retry with maximum compression instead of hard truncation
                    print(f"{prefix}Second editor attempt exceeded limit; maximum compression retry.")
                    check_prompt_ = (
                        check_prompt.rstrip()
                        + f"\n\nMAXIMUM COMPRESSION: Your output MUST be under {max_chars} characters. "
                        + "Cut examples, not insights. Merge sentences. Remove every unnecessary word. "
                        + "Output only the essential message. NO ellipsis, NO truncation."
                    )
                    payload["messages"][1]["content"] = check_prompt_
                    r3 = requests.post(url, headers=headers, json=payload, timeout=25)
                    resp_data3 = r3.json()
                    if "choices" in resp_data3 and resp_data3["choices"]:
                        msg3 = resp_data3["choices"][0].get("message") or {}
                        fixed3 = msg3.get("content", "").strip()
                        if fixed3 and len(fixed3) <= max_chars:
                            fixed = fixed3
                        elif fixed3:
                            # Final soft fallback: trim to last complete sentence without ellipsis
                            trimmed = fixed3[:max_chars-3]
                            last_period = trimmed.rfind(".")
                            if last_period > max_chars * 0.6:
                                fixed = trimmed[:last_period+1]
                            else:
                                fixed = trimmed.rstrip()
                    else:
                        fixed = _editor_fallback(caption, platform, max_chars)
                else:
                    fixed = _editor_fallback(caption, platform, max_chars)
            else:
                fixed = _editor_fallback(caption, platform, max_chars)
        return fixed

    fixed = _call_editor(0)
    
    try:
        if fixed and not _caption_is_incomplete(fixed) and not _has_mid_sentence_break(fixed):
            return fixed
        if fixed:
            print(f"  Caption looks incomplete; retrying editor...")
        fixed_attempt2 = _call_editor(1)
        if fixed_attempt2 and not _caption_is_incomplete(fixed_attempt2) and not _has_mid_sentence_break(fixed_attempt2):
            return fixed_attempt2

        print(f"  AI Editor returned unexpected structure, using raw/truncated.")
        result = _editor_fallback(caption, platform, max_chars)
        return result.strip()
    except Exception as e:
        print(f"  AI Editor check failed: {e}")
        result = _editor_fallback(caption, platform, max_chars)
        return result.strip()


def generate_caption(caption_prompt: str, platform: str = "instagram", system_prompt: Optional[str] = None, book_context: str = "", book_insights: Optional[Dict] = None) -> str:
    """
    Generates a caption exclusively via AI Horde with an AI-driven verification loop.
    """
    # Platform-specific limits
    limits = {"bluesky": 250, "threads": 450, "instagram": 1400,
                      "linkedin": 1800, "pinterest": 450, "youtube": 400, "facebook": 500}
    max_chars = limits.get(platform.lower(), 1800)

    if not system_prompt:
        system_prompt = f"""You are the 'Professional Failure Expert' persona for {BOOK_AUTHOR}, author of {BOOK_TITLE}.
        Your vibe: Witty, self-deprecating, and philosophical. Write RELATABLE, HUMOROUS, and slightly cynical captions.
        Sound like a smart friend who just realized life is a chaotic simulation.

        Hook rules:
        - NEVER start with "Ah", "Ah yes", "Ah, what a", or any variation of "Ah...".
        - Open with a specific observation, a bold claim, or a relatable scenario.
        - The first line must pass the "so what?" test — if someone reads it and shrugs, rewrite it.
        """
    # Add formatting requirements
    full_system_content = system_prompt + f"""
Hard requirements for {platform.upper()}:
- TOTAL CHARACTER LIMIT: {max_chars} characters. YOU MUST NOT EXCEED THIS.
- Structure: 1 Hook line, 2-3 short Body lines, 1 CTA.
- No Markdown (** or __). No hashtags in body. No labels like 'HOOK:'.
- NEVER start a caption with "Ah", "Ah yes", "Ah, what a", or any variation.
- Open with a specific observation, a bold claim, or a relatable scenario.
- IF YOU EXCEED THE CHARACTER LIMIT, THE POST WILL FAIL. BE CONCISE.
"""
    if platform.lower() == "youtube":
        full_system_content += """
YouTube-specific rules:
- YouTube is visual-first. The audience watches, they don't read essays.
- Hook in the FIRST line or they scroll past.
- 1-2 sentences max for the body. Bullet points if needed, but keep them punchy.
- End with 1 short CTA: like, comment, or link in description.
- Do NOT write long paragraphs. Use line breaks for scanability.
- Output must feel like a quick voiceover note, not a blog post.
"""
    if platform.lower() == "linkedin":
        full_system_content += f"""
LinkedIn-specific rules:
- The feed shows ~140 chars before "see more". Your FIRST line must be a sharp hook that screams "click this."
- STRICTLY BANNED OPENERS: "Ah", "Ah yes", "Ah, what a", "Ah yes, wabi-sabi", or any lazy "Ah..." variation. If you catch yourself typing it, delete the whole line and start over.
- Open with a concrete observation, a counterintuitive claim, or a specific real-world example.
- LinkedIn rewards COMMENTS over likes. End with a specific, low-friction question that invites professionals to share their experience.
- Use 3-5 hashtags only. No hashtag soup.
- Voice: digital wellness first. Grounded, warm, evidence-based, slightly witty — like a parent or coach sharing a real insight, never an author pitching a book.
- Body can be longer (up to {max_chars} chars) — but optimal dwell-time performance is ~1400–1800 chars. Use short paragraphs and white space.
- Do NOT mention {BOOK_TITLE}, "out now", "link in bio", or any purchase/plug language. This is not a book ad.
- Do NOT use markdown. Do NOT write in all caps.
"""

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


def _sanitize_profanity(text: str) -> str:
    """Replace explicit profanity with cleaner alternatives."""
    replacements = {
        "fuck-ups": "mistakes",
        "fuck up": "mistake",
        "fucked up": "messed up",
        "fucking": "damn",
        "fuck": "freak",
        "shit": "junk",
        "bullshit": "nonsense",
        "bitch": "pain",
        "bastard": "rogue",
        "ass": "jerk",
        "damn": "darn",
        "hell": "heck",
        "crap": "mess",
        "screw-up": "mistake",
    }
    lowered = text.lower()
    for bad, clean in replacements.items():
        if bad in lowered:
            text = text.replace(bad, clean)
            text = text.replace(bad.title(), clean.title())
            text = text.replace(bad.upper(), clean.upper())
    return text


def _process_caption_output(caption: str, target_platform: str = "instagram") -> str:
    """Final surgical cleanup of markdown, hashtags, and leading/trailing junk symbols."""
    # 1. Initial strip of common AI artifacts and brackets
    text = caption.strip().strip('{}[]"\' ')
    
    # 2. Remove markdown artifacts
    final = text.replace("**", "").replace("*", "").replace("__", "").replace("_", "")
    
    # 2b. Sanitize profanity
    final = _sanitize_profanity(final)
    
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
        if any(kw in parsed_text for kw in ["nsfw content detected", "blocked by client", "nsfw", "sexually explicit", "censored"]):
            print(f"Censorship text detected in {image_path}")
            return True

    except requests.exceptions.Timeout:
        # Network timeout: cannot verify safety. Assume censored and force a retry.
        print(f"OCR check timed out for {image_path}. Assuming censored for safety (will retry).")
        return True
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else 0
        # Server or client error: assume censored and force a retry rather than posting unsafe content.
        print(f"OCR server error ({status}) for {image_path}. Assuming censored for safety (will retry).")
        return True
    except Exception as e:
        print(f"OCR check unexpected error: {e}. Assuming censored for safety (will retry).")
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


def generate_carousel(pillar: str, topic: str, timestamp: str) -> List[str]:
    """
    Generate a 5-slide LinkedIn carousel from a pillar/topic.
    Style: off-white/beige paper background, yellow highlighter behind
    the first two lines, black serif text, centered.
    Slides follow high-performing LinkedIn 2026 carousel anatomy:
    - 4:5 ratio (1080x1350)
    - Slide 1 = hook question
    - Slide 2 = quick context/lens
    - Slide 3 = reframe or paradox
    - Slide 4 = actionable system/framework
    - Slide 5 = CTA / announcement
    Returns list of 5 image paths.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import textwrap
    except Exception as e:
        print(f"Carousel generation skipped (missing PIL): {e}")
        return []

    pillar_title = pillar.replace('_', ' ').title()
    topic_clean = topic.strip().rstrip('.')
    slides = [
        f"What if {topic_clean}?",
        f"{pillar_title if pillar_title else topic_clean} is not what you think it is.",
        f"The {topic_clean} paradox: small inputs create massive outcomes.",
        "The Nine Stitches approach: intent plus system beats motivation.",
        "The Nine Stitches\nOut now"
    ]
    base_dir = "images"
    paths: List[str] = []

    BG_COLOR = (240, 240, 230)
    HIGHLIGHT_COLOR = (255, 255, 100, 160)
    TEXT_COLOR = (0, 0, 0)

    def _load_serif(size: int):
        font_paths = [
            "C:/Windows/Fonts/georgia.ttf",
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/-times.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "DejaVuSerif.ttf",
            "Georgia.ttf",
            "Times New Roman.ttf",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    for i, text in enumerate(slides):
        out_path = f"{base_dir}/carousel_{timestamp}_slide_{i+1}.jpg"
        os.makedirs(base_dir, exist_ok=True)

        w, h = 1080, 1350
        img = Image.new("RGB", (w, h), BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Larger font for readability
        font_size = 68 if len(text) < 30 else 56
        font = _load_serif(font_size)

        wrapped_lines = textwrap.wrap(text, width=26)
        line_spacing = 24

        # Measure each line individually
        line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in wrapped_lines]
        line_ws = [b[2] - b[0] for b in line_bboxes]
        line_hs = [b[3] - b[1] for b in line_bboxes]
        line_h = max(line_hs) if line_hs else 40

        # Total block height
        th = line_h * len(wrapped_lines) + line_spacing * (len(wrapped_lines) - 1)
        # Use max line width for block width
        tw = max(line_ws) if line_ws else 10

        margin_x = 100
        # Center the text block horizontally and vertically
        block_x = (w - tw) / 2
        block_y = (h - th) / 2

        # Yellow highlighter behind the first two lines only
        highlight_lines = wrapped_lines[:2]
        hline_bbox = None
        if len(wrapped_lines) == 0:
            hline_bbox = (block_x, block_y, block_x + 10, block_y + 10)
        else:
            # Highlight box matches the widest highlighted line, centered on page
            hl_w = max(line_ws[:len(highlight_lines)]) if highlight_lines else 10
            hl_x = (w - hl_w) / 2
            hl_h = line_h * len(highlight_lines) + line_spacing * (len(highlight_lines) - 1)
            hline_bbox = (hl_x, block_y, hl_x + hl_w, block_y + hl_h)

        highlight_pad = 18
        overlay_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay_layer)
        odraw.rectangle(
            (
                max(hline_bbox[0] - highlight_pad, margin_x - 20),
                hline_bbox[1] - highlight_pad,
                min(hline_bbox[2] + highlight_pad, w - margin_x + 20),
                hline_bbox[3] + highlight_pad,
            ),
            fill=HIGHLIGHT_COLOR,
        )
        img = Image.alpha_composite(img.convert("RGBA"), overlay_layer).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw each line centered individually for true center alignment
        for line_idx, line in enumerate(wrapped_lines):
            line_w = line_ws[line_idx]
            line_x = (w - line_w) / 2
            line_y = block_y + line_idx * (line_h + line_spacing)
            draw.text((line_x, line_y), line, font=font, fill=TEXT_COLOR)
        img.save(out_path, format="JPEG", quality=95, optimize=True)
        paths.append(out_path)

    return paths

# -------------------------
# Main flow
# -------------------------

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


def extract_book_insights(text: str) -> dict:
    """Extract key themes and structure from book for better context."""
    insights = {
        "central_question": "What happens if you try to fail and succeed?",
        "epigraph": "To become, be calm. To be calm, pretend to be calm.",
        "chapters": [],
        "key_concepts": [
            "intention vs outcome", "productive failure", "adversity-growth cycles",
            "antifragility", "wabi-sabi", "kintsugi", "keystone species"
        ]
    }
    
    if text:
        import re
        chapter_matches = re.findall(r"(?:Chapter|CHAPTER)\s+(\d+)\s*[:.-]?\s*(.*)", text[:10000])
        for num, title in chapter_matches[:5]:
            insights["chapters"].append({"number": int(num), "title": title.strip()})
            
    if not insights["chapters"]:
        insights["chapters"] = [
            {"number": 1, "title": "The One in Time", "theme": "Intention vs. Outcome"},
            {"number": 2, "title": "If you can't evade it, embrace it", "theme": "Adversity and Growth"}
        ]
        
    return insights


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
            except Exception: continue
        # Massive fallback if no font found: PIL default is too small, 
        # but we can't do much without a file. 
        # We'll at least warn or try a generic name.
        try: return ImageFont.truetype("arial.ttf", size=size)
        except Exception: return ImageFont.load_default()

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
            except Exception: pass
            
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


def apply_logo_watermark(image_path: str, logo_path: str = "wp logo.png") -> str:
    """
    Apply a small bottom-right logo watermark to an image for brand consistency.
    Only affects main bot assets; Wilma workflow does not call this.
    """
    try:
        from PIL import Image
    except Exception as e:
        print(f"Logo watermark skipped (missing PIL): {e}")
        return image_path

    if not os.path.exists(logo_path):
        print(f"Logo watermark skipped: {logo_path} not found")
        return image_path

    try:
        base = Image.open(image_path).convert("RGBA")
        logo = Image.open(logo_path).convert("RGBA")

        # Scale logo to ~160px wide
        logo_w = 160
        logo_h = int(logo.height * logo_w / logo.width)
        logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)

        padding = 30
        x = base.width - logo_w - padding
        y = base.height - logo_h - padding

        base.paste(logo, (x, y), logo)

        # Save back as JPEG (drop alpha)
        out = Image.new("RGB", base.size, (10, 14, 23))
        out.paste(base, mask=base.split()[3])
        out.save(image_path, format="JPEG", quality=95, optimize=True)
        return image_path
    except Exception as e:
        print(f"Logo watermark failed: {e}")
        return image_path


def _generate_text_ai_horde(prompt: str, system_prompt: str = "", max_tokens: int = 512) -> str:
    """Generates text using AI Horde with Cerebras fallback."""
    full_prompt = f"### Instruction:\n{system_prompt}\n\n### Input:\n{prompt}\n\n### Response:\n"
    available_text_models = _get_available_horde_text_models()
    if not available_text_models:
        print("  No AI Horde text models available. Using Cerebras fallback...")
        return _generate_text_cerebras(prompt, system_prompt, max_tokens)

    # Prefer models from your accessible list, in order.
    preferred_prefixes = [
        "aphrodite/TheDrummer/",
        "koboldcpp/",
        "coder3101/",
        "aphrodite/SicariusSicariiStuff/",
    ]
    preferred_models = []
    remaining_models = list(available_text_models)
    for prefix in preferred_prefixes:
        for m in available_text_models:
            if m.startswith(prefix) and m in remaining_models:
                preferred_models.append(m)
                remaining_models.remove(m)
    # Append in their original popularity order as last resort.
    preferred_models.extend(remaining_models)
    payload = {
        "prompt": full_prompt,
        "params": {"n": 1, "max_context_length": 4096, "max_length": max_tokens, "rep_pen": 1.1, "temperature": 0.75, "top_p": 0.9},
        "models": preferred_models[:10],
    }
    headers = {"apikey": os.environ.get("AI_HORDE_API_KEY", "0000000000"), "Content-Type": "application/json"}
    submit_url = "https://aihorde.net/api/v2/generate/text/async"

    try:
        r = requests.post(submit_url, headers=headers, json=payload, timeout=90)
        if r.status_code == 403:
            print("  AI Horde text 403. Falling back to Cerebras...")
            return _generate_text_cerebras(prompt, system_prompt, max_tokens)
        r.raise_for_status()
        job_id = r.json().get("id")
        if not job_id:
            raise RuntimeError("AI Horde text-gen did not return a job ID")

        status_url = f"https://aihorde.net/api/v2/generate/text/status/{job_id}"
        for _ in range(36):
            time.sleep(5)
            res = requests.get(status_url, timeout=30)
            data = res.json()
            if data.get("done"):
                generations = data.get("generations", [])
                if generations:
                    return generations[0].get("text", "").strip()
                raise RuntimeError("AI Horde text-gen returned 'done' but no content")
            if _ % 6 == 0:
                print(f"  AI Horde (Text) status: {data.get('queue_position', 'unknown')} in queue...")
        raise RuntimeError("AI Horde text generation timed out")
    except requests.exceptions.HTTPError as e:
        status_ = e.response.status_code if e.response else 0
        if status_ == 403:
            print("  AI Horde text 403 after submit. Falling back to Cerebras...")
            return _generate_text_cerebras(prompt, system_prompt, max_tokens)
        print(f"  AI Horde text generation failed: {e}")
        raise
    except Exception as e:
        print(f"  AI Horde text generation failed: {e}. Falling back to Cerebras...")
        return _generate_text_cerebras(prompt, system_prompt, max_tokens)


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
    
    # --- CONTENT QUEUE LOGIC ---
    # We want to maintain a buffer of at least 5 posts ready to go.
    target_buffer = 5
    current_buffer = len(state.get("content_queue", []))
    
    if args.mode == "generate_all":
        if current_buffer >= target_buffer:
            print(f"✅ Content buffer is full ({current_buffer}/{target_buffer}). Nothing to generate.")
            return
        
        to_generate = target_buffer - current_buffer
        print(f"🔄 Buffer status: {current_buffer}/{target_buffer}. Generating {to_generate} new bundles...")
        
        # Resume any pending bundle from a previous partial run first
        if _try_resume_pending(state, platforms):
            current_buffer = len(state.get("content_queue", []))
            to_generate = max(0, target_buffer - current_buffer)
            print(f"Buffer after resume: {current_buffer}/{target_buffer}. {to_generate} more to generate.")
    else:
        # Single mode: we just generate one and don't touch the queue (Legacy support)
        to_generate = 1

    for i in range(to_generate):
        print(f"\n📦 GENERATING BUNDLE {i+1}/{to_generate}...")
        
        # Update used IDs for this specific selection
        primary_platform = "instagram"
        platform_used_ids = set(state.get("used_ids", {}).get(primary_platform, []))
        
        available_posts = [p for p in all_posts if p.get("id") not in platform_used_ids]
        if not available_posts:
            print(f"Queue empty. Generating new batch...")
            new_posts = _generate_new_posts()
            max_id = max((post.get("id", 0) for post in all_posts), default=0)
            for j, post_item in enumerate(new_posts):
                post_item["id"] = max_id + j + 1
                all_posts.append(post_item)
            _write_posts(all_posts)
            available_posts = new_posts

        post = _weighted_post_choice(available_posts, state, platform=primary_platform)
        post_id = post.get("id")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Selected post {post_id}: {post.get('title', 'Untitled')}")

        # Unique paths for this specific bundle
        bundle_image = f"images/post_{timestamp}.jpg"
        bundle_reel = f"reels/reel_{timestamp}.mp4"
        bundle_story = f"images/story_{timestamp}.jpg"

        # Initialize pending bundle for this run
        pending = {
            "post_id": post_id,
            "timestamp": timestamp,
            "post": post,
            "platforms": platforms,
            "image": bundle_image,
            "reel": bundle_reel,
            "story": bundle_story,
            "carousel": [],
            "master_reflection": None,
            "captions": {},
        }

        # --- 1. MEDIA GENERATION (step-by-step with progress save) ---
        try:
            # Generate Master Image (CLEAN)
            raw_path = generate_image(post["image_prompt"])
            _write_output_jpg(raw_path, bundle_image)
            print(f"✓ Master image generated: {bundle_image}")
            _save_pending(state, pending)

            # --- THE MASTER REFLECTION (AI HORDE ONCE) ---
            print("Generating Master Reflection (AI Horde)...")
            master_system = f"""You are the 'Professional Failure Expert' persona for {BOOK_AUTHOR}. Write a deep, witty, and cynical reflection on the topic below. No length limit. Sound like a smart friend.

Style rules:
- If the content naturally connects to {BOOK_TITLE}, plant a subtle nod — never a hard sales pitch.
- Let ideas breathe. Do not summarize or truncate; the platform editor handles length later.
"""
            master_reflection = _generate_text_ai_horde(post["caption_prompt"], system_prompt=master_system)
            pending["master_reflection"] = master_reflection
            _save_pending(state, pending)
            print(f"✓ Master Reflection acquired.")

            # Generate Master Reel Hook from the Master Reflection
            media_hook = extract_hook_text(_ai_verify_caption(master_reflection, "instagram", 100))
            
            # --- VIDEO/STORY GENERATION (Uses CLEAN image) ---
            print("Generating Master Reel (6s)...")
            generate_reel(bundle_image, media_hook, bundle_reel, duration_s=6.0)

            print("Generating Story Image...")
            generate_story_image(bundle_image, "post_amplifier", media_hook, bundle_story)

            # --- STATIC OVERLAY (Finalizes the static bundle_image) ---
            print("Adding static text overlay to master image...")
            add_static_text_overlay(bundle_image, media_hook)
            print(f"✓ Final static asset prepared.")

            # --- LOGO WATERMARK (Main bot only; Wilma never reaches here) ---
            apply_logo_watermark(bundle_image)
            print(f"✓ Logo watermark applied.")

            # --- LINKEDIN CAROUSEL (Static background, no AI image cost) ---
            bundle_carousel = []
            if "linkedin" in [x.lower() for x in platforms]:
                # Carousel cadence: only on Wednesdays to keep 1 carousel per 4-post week
                is_carousel_day = datetime.now().weekday() == 2
                if is_carousel_day:
                    print("Generating LinkedIn carousel (5 slides)...")
                    bundle_carousel = generate_carousel(
                        pillar=post.get("pillar", "micro_philosophy"),
                        topic=post.get("title", post.get("caption_prompt", "productivity")),
                        timestamp=timestamp,
                    )
                    if bundle_carousel:
                        print(f"✓ Carousel ready: {len(bundle_carousel)} slides")
                else:
                    print("Single-image LinkedIn post (carousel reserved for Wednesdays).")
            pending["carousel"] = bundle_carousel
            _save_pending(state, pending)

        except Exception as e:
            print(f"❌ Media generation failed: {e}. Progress saved, will resume next run.")
            _save_pending(state, pending)
            return  # Exit cleanly instead of 'continue' — next run resumes this bundle

        # --- 2. GENERATE PLATFORM-SPECIFIC CAPTIONS (AI CRITIC EDITS) ---
        bundle_captions = {}
        for p in platforms:
            print(f"  Tailoring for {p.upper()}...")
            try:
                limits = {"bluesky": 250, "threads": 450, "instagram": 1400,
                          "linkedin": 1800, "pinterest": 450, "youtube": 400, "facebook": 500}
                hard_total_limits = {"bluesky": 300, "threads": 500, "pinterest": 500,
                                     "instagram": 1600, "linkedin": 2000, "youtube": 600, "facebook": 600}
                max_c = limits.get(p.lower(), 1800)

                cta = _choose_next_cta(state, preferred_category="engagement" if p.lower() == "linkedin" else None)
                cta = _render_cta(cta)
                tags = _choose_hashtags(state, post.get("pillar", ""), platform=p)
                linkedin_comment = random.choice(LINKEDIN_COMMENT_PROMPTS) if p.lower() == "linkedin" else ""

                # Reserve space for CTA, hashtags, and comment prompt so the editor does not eat them.
                _reserved = 0
                if p.lower() == "bluesky":
                    _reserved = len("\n\nWant to read more?... check out my LinkedIn")
                elif p.lower() == "linkedin":
                    _reserved += len("\n\n" + (cta or ""))
                    _reserved += len("\n\n" + " ".join(tags)) if tags else 0
                    _reserved += len("\n\n" + linkedin_comment) if linkedin_comment else 0
                elif p.lower() == "youtube":
                    if cta:
                        _reserved += len("\n\n" + cta)
                    if tags:
                        _reserved += len("\n\n" + " ".join(tags))
                else:
                    if cta:
                        _reserved += len("\n\n" + cta)
                    if p.lower() != "threads" and tags:
                        _reserved += len("\n\n" + " ".join(tags))
                max_c = max(100, max_c - _reserved)

                tailored_cap = _ai_verify_caption(master_reflection, p, max_c)
                if tailored_cap is None:
                    raise ValueError("AI editor returned None")
                final_cap = tailored_cap.strip()
                
                if p.lower() == "bluesky":
                    # Only the permanent LinkedIn CTA; rotating CTAs removed for Bluesky.
                    final_cap += "\n\nWant to read more?... check out my LinkedIn"
                elif p.lower() == "linkedin":
                    if cta: final_cap += "\n\n" + cta
                    if tags: final_cap += "\n\n" + " ".join(tags)
                    if linkedin_comment: final_cap += "\n\n" + linkedin_comment
                else:
                    if cta: final_cap += "\n\n" + cta
                    # Skip hashtags for Threads
                    if p.lower() != "threads" and tags:
                        final_cap += "\n\n" + " ".join(tags)

                final_cap = clean_caption_formatting(final_cap)
                bundle_captions[p] = final_cap
                pending["captions"][p] = final_cap
                _save_pending(state, pending)
                print(f"  ✓ Caption for {p}: {len(final_cap)} chars")

            except Exception as e:
                print(f"  Tailoring failed for {p}: {e}")
                bundle_captions[p] = f"[Caption generation failed: {e}]"
                pending["captions"][p] = bundle_captions[p]
                _save_pending(state, pending)

        # --- 3. ADD TO QUEUE ---
        new_bundle = {
            "post_id": post_id,
            "timestamp": timestamp,
            "image": bundle_image,
            "reel": bundle_reel,
            "story": bundle_story,
            "carousel": bundle_carousel,
            "captions": bundle_captions,
            "platforms_posted": []
        }
        
        if args.mode == "generate_all":
            state["content_queue"].append(new_bundle)
            # Mark post_id as used for THIS platform's selection logic
            for p in platforms:
                if post_id not in state["used_ids"][p]:
                    state["used_ids"][p].append(post_id)
            
            state["last_pillar"] = str(post.get("pillar", "micro_philosophy")).strip()
            _write_state(state) # Save after each bundle to prevent data loss on crash
            
            # Clear pending on success
            state.pop("pending_bundle", None)
            _write_state(state)
            
            print(f"✅ Bundle {post_id} added to queue. Queue size: {len(state['content_queue'])}")
        else:
            # Legacy single mode: write files directly to root for immediate consumption
            shutil.copy(bundle_image, "output.jpg")
            shutil.copy(bundle_reel, "reel.mp4")
            shutil.copy(bundle_story, "story.jpg")
            with open("captions_bundle.json", "w", encoding="utf-8") as f:
                json.dump(bundle_captions, f, indent=2)
            # Create flags for single mode
            for p in platforms:
                with open(f"{p}_ready.flag", "w") as f: f.write(timestamp)
            print("✓ Single mode assets prepared in root.")
            return # Exit after one

    print(f"✓ Generation cycle complete. Current buffer: {len(state['content_queue'])} items.")


if __name__ == "__main__":
    main()
