import os
import sys

# type: ignore[reportAttributeAccessIssue] — Pyright doesn't track io.TextIOWrapper.reconfigure on all platforms
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[reportAttributeAccessIssue]
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[reportAttributeAccessIssue]

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
from typing import Any, Dict, Optional, List, Union
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
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY", "")

CAPTION_FILE = "caption.txt"

def sanitize_image_prompt(prompt: str) -> str:
    return (prompt or "").strip()

def _read_posts() -> list[dict]:
    try:
        if os.path.exists("posts.json"):
            with open("posts.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                posts = data.get("posts", []) if isinstance(data, dict) else data
                return [p for p in posts if isinstance(p, dict)]
    except Exception as e:
        print(f"Error reading posts.json: {e}")
    return []

def _write_posts(posts: list[dict]) -> None:
    """Write the posts list back to posts.json (preserving the 'posts' key)."""
    try:
        with open("posts.json", "w", encoding="utf-8") as f:
            json.dump({"posts": posts}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing posts.json: {e}")

def extract_hook_text(_text: str) -> str:
    text = (_text or "").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    # Preserve full short topic/hook text; only truncate very long strings.
    if len(first_line) <= 120:
        return first_line
    return first_line[:117].rstrip() + "..."

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

HASHTAG_CLUSTERS = {
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

PILLAR_WEIGHTS = {
    "micro_philosophy": 0.30,
    "nature_metaphor": 0.25,
    "systems_psychology": 0.20,
    "author_voice": 0.15,
    "quote": 0.10,
}
PILLAR_HISTORY_WINDOW = 8

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
    """Deterministic local caption editor. No external API calls."""
    if not caption:
        return ""

    text = caption.strip()

    # Strip banned lazy openers deterministically.
    banned_openers = ("ah, ", "ah yes", "ah, what a", "ah—", "ah.")
    lines = text.splitlines()
    if lines:
        first_line = lines[0].strip().lower()
        if any(first_line.startswith(b) for b in banned_openers):
            rest = [ln for ln in lines[1:] if ln.strip()]
            if rest:
                text = "\n".join(rest).strip()

    # Strip markdown artifacts.
    text = text.replace("**", "").replace("*", "").replace("__", "").replace("_", "")

    # Sanitize profanity locally.
    text = _sanitize_profanity(text)

    # Remove accidental metadata lines.
    cleaned_lines = []
    skip_patterns = [
        r"^here we go", r"^---", r"^word count:", r"^character limit:",
        r"^i'll write", r"^i will write", r"^here is a caption", r"^sure, here",
        r"^following your instructions",
    ]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if any(__import__("re").search(pat, stripped, __import__("re").IGNORECASE) for pat in skip_patterns):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines).strip()

    # Enforce character limit by trimming at a natural boundary.
    if len(text) > max_chars:
        cut = text[: max_chars - 3]
        last_period = cut.rfind(".")
        if last_period > max_chars * 0.6:
            text = cut[: last_period + 1].strip()
        else:
            last_newline = cut.rfind("\n")
            if last_newline > max_chars * 0.6:
                text = cut[:last_newline].strip()
            else:
                text = cut.rstrip()

    return text.strip()


def _strip_trailing_cta(text: str) -> str:
    """Remove a trailing CTA / engagement question so the workflow can append its own."""
    if not text:
        return text
    lines = text.splitlines()
    cta_markers = [
        "want to read more",
        "check out my linkedin",
        "drop a note",
        "what's your take",
        "who else has seen",
        "agree or disagree",
        "follow for more",
        "save this",
        "share this",
        "comment below",
        "let me know below",
        "thoughts?",
        "what do you think?",
    ]
    # Strip from the last non-empty line backwards if it looks like a CTA.
    while lines:
        tail = lines[-1].strip().lower()
        if not tail:
            break
        if any(tail.startswith(m) or tail == m for m in cta_markers):
            lines.pop()
            continue
        if tail.endswith("?") and len(tail) < 120:
            lines.pop()
            continue
        break
    return "\n".join(lines).rstrip()


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
- Voice: Max Wigman's author voice. Grounded, philosophical, slightly literary, warm but not parental — like The Nine Stitches / Wabi-Sabi Wisdom energy: reflective, precise, occasionally wry. Never a wellness-parent plug. Strip any Digital Guardian phrasing, teen/game/family anecdotes, or book-pitching language.
- Body can be longer (up to {max_chars} chars) — but optimal dwell-time performance is ~1400–1800 chars. Use short paragraphs and white space.
- Do NOT mention {BOOK_TITLE}, "out now", "link in bio", or any purchase/plug language. This is not a book ad.
- Do NOT use markdown. Do NOT write in all caps.
- Include exactly one sentence of genuine warmth amid the cynicism. This is the moment the mask slips. It's the difference between "failure expert" and "person who failed." Not sentimental — just real.
"""
    if platform.lower() == "instagram":
        full_system_content += f"""
Instagram-specific rules:
- Caption is secondary to the visual. Strip the question close if it feels redundant; keep the hook.
- The image carries the narrative. Write a caption that opens the story, doesn't close it.
- Use 3-5 hashtags. Keep it clean.
"""
    if platform.lower() == "threads":
        full_system_content += """
Threads-specific rules:
- Cut to the point. Keep the hook and one body line. Drop the warmth sentence if needed for length.
- End with a short question or open thought.
"""
    if platform.lower() == "facebook":
        full_system_content += f"""
Facebook-specific rules:
- More personal than LinkedIn. Shift the anecdote one notch toward conversational tone.
- Keep the one warmth sentence if it fits. It reads naturally here.
"""
    if platform.lower() == "pinterest":
        full_system_content += """
Pinterest-specific rules:
- Pin title is the hook. Pin description expands it.
- No more than 2-3 sentences in the description. One question close.
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



# -------------------------
# Persistence helpers
# -------------------------
def _read_state() -> dict:
    try:
        if os.path.exists("state.json"):
            with open("state.json", "r", encoding="utf-8") as f:
                state = json.load(f)
                if isinstance(state.get("used_ids"), list):
                    old_used = state["used_ids"]
                    state["used_ids"] = {p: list(old_used) for p in ["instagram","linkedin","pinterest","youtube","threads","bluesky"]}
                if isinstance(state.get("used_ids"), dict):
                    for p in ["youtube", "threads", "bluesky"]:
                        state["used_ids"].setdefault(p, [])
                state.setdefault("content_queue", [])
                return state
    except Exception as e:
        print(f"Error reading state.json: {e}")
    return {
        "used_ids": {"instagram": [], "linkedin": [], "pinterest": [], "youtube": [], "threads": [], "bluesky": []},
        "last_cta": "", "cta_history": [], "last_hashtag_cluster": "", "last_hashtags": [], "last_pillar": "", "pillar_history": [],
    }

def _write_state(_state: dict) -> None:
    try:
        with open("state.json", "w", encoding="utf-8") as f:
            json.dump(_state, f, indent=4)
    except Exception as e:
        print(f"Error writing state.json: {e}")

# --- Pending-bundle helpers for mid-run failure recovery ---
def _save_pending(state, pending_data):
    state["pending_bundle"] = pending_data
    _write_state(state)

def _load_and_clear_pending(state):
    pending = state.pop("pending_bundle", None)
    if pending:
        _write_state(state)
    return pending

def _try_resume_pending(state, platforms):
    pending = _load_and_clear_pending(state)
    if not pending:
        return False
    emoji = "🔄"
    print("\nResuming pending bundle: post_id=%s" % pending.get("post_id"))
    media_paths = ["image", "reel", "story"]
    media_ok = all(pending.get(p) and os.path.exists(pending[p]) for p in media_paths)
    post = pending.get("post")
    master_reflection = pending.get("master_reflection")
    if not media_ok or not master_reflection or not post:
        print("  ❌ Cannot resume: incomplete pending bundle")
        return False
    print("  ✓ Pending bundle OK, continuing from last saved progress.")
    return True


# -------------------------
# Caption/CTA/hashtag helpers
# -------------------------
def clean_caption_formatting(text: str) -> str:
    """
    Aggressively strips numbering, labels, and Markdown artifacts from LLM output.
    """
    import re
    text = text.replace("**", "").replace("*", "").replace("__", "").replace("_", "")
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
        if re.fullmatch(rf"(?i){structural_labels}[:\\s\\-]*", l):
            continue
        while True:
            old_l = l
            l = re.sub(r"^\(?\d+[\.\)\:]\s*", "", l)
            l = re.sub(rf"(?i)^{structural_labels}[:\\s\\-]*", "", l)
            l = re.sub(r"^[\-\•\*\+]\s*", "", l)
            if l == old_l:
                break
        if l:
            cleaned_lines.append(l)
    final_text = "\n".join(cleaned_lines).strip()
    final_text = re.sub(rf"(?i)\b{structural_labels}:\s*", "", final_text)
    final_text = re.sub(rf"(?i)\.({structural_labels})", ". ", final_text)
    final_lines = []
    for line in final_text.splitlines():
        l = line.strip()
        if not l:
            final_lines.append("")
            continue
        if re.match(rf"(?i)^{structural_labels}[:\\s\\-]*", l):
            l = re.sub(rf"(?i)^{structural_labels}[:\\s\\-]*", "", l).strip()
            if not l:
                continue
        final_lines.append(l)
    return "\n".join(final_lines).strip()


# Caption/CTA/hashtag helpers
CTA_BY_CATEGORY = {
    "engagement": ["What do you think?", "Have you experienced this?", "Does this resonate?"],
    "save": ["Save this for later.", "Bookmark this insight."],
    "share": ["Share this with someone who needs to hear it.", "Re-post this if it got you thinking."],
    "book": [f"Grab your copy of {BOOK_TITLE} and see why readers say it changed how they see themselves.", "The next chapter is waiting. {BOOK_TITLE} is available now. {BOOK_URL}"],
}
CTA_CATEGORY_WEIGHTS = {"engagement": 0.5, "save": 0.2, "share": 0.2, "book": 0.1}
CTA_HISTORY_WINDOW = 8
LINKEDIN_COMMENT_PROMPTS = [
    "Drop a note in the comments if this sounds familiar.",
    "What’s your take on this?",
    "Who else has seen this pattern?",
    "Agree or disagree? Let’s discuss.",
]

def choose_next_cta(state: dict, preferred_category=None):
    all_items = []
    for category, ctas in CTA_BY_CATEGORY.items():
        for cta in ctas:
            all_items.append({"category": category, "text": cta})
    if not all_items:
        return ""
    last_cta = str(state.get("last_cta", "") or "").strip()
    raw_hist = state.get("cta_history", [])
    if not isinstance(raw_hist, list):
        raw_hist = []
    history = [str(x) for x in raw_hist if isinstance(x, str)]
    if len(history) > CTA_HISTORY_WINDOW:
        history = history[-CTA_HISTORY_WINDOW:]
        state["cta_history"] = history
    cta_to_cat = {item["text"]: item["category"] for item in all_items}
    cat_counts = {k: 0 for k in CTA_BY_CATEGORY.keys()}
    for cta in history:
        cat = cta_to_cat.get(cta)
        if cat in cat_counts:
            cat_counts[cat] += 1
    window = max(1, min(CTA_HISTORY_WINDOW, len(history)))
    categories = [c for c in CTA_CATEGORY_WEIGHTS.keys() if CTA_BY_CATEGORY.get(c)]
    if not categories:
        categories = list(CTA_BY_CATEGORY.keys())
    if not categories:
        return random.choice([item["text"] for item in all_items if item["text"] != last_cta] or [all_items[0]["text"]])
    cat_weights = []
    for cat in categories:
        base = CTA_CATEGORY_WEIGHTS.get(cat, 0.25)
        expected = base * window if history else base
        actual = cat_counts.get(cat, 0)
        delta = expected - actual
        factor = max(0.60, min(1.40, 1.0 + (delta / max(1.0, window))))
        cat_weights.append(max(0.001, base * factor))
    total = sum(cat_weights) or 1.0
    cat_weights = [w / total for w in cat_weights]
    if preferred_category and preferred_category in CTA_BY_CATEGORY and CTA_BY_CATEGORY.get(preferred_category):
        chosen_category = preferred_category
        options = list(CTA_BY_CATEGORY.get(chosen_category, []))
        filtered = [c for c in options if c != last_cta]
        chosen_cta = random.choice(filtered) if filtered else random.choice(options)
    else:
        chosen_category = random.choices(categories, weights=cat_weights, k=1)[0]
        options = list(CTA_BY_CATEGORY.get(chosen_category, []))
        filtered = [c for c in options if c != last_cta]
        if filtered:
            chosen_cta = random.choice(filtered)
        else:
            global_options = [item["text"] for item in all_items if item["text"] != last_cta]
            chosen_cta = random.choice(global_options if global_options else [options[0]])
    state["last_cta"] = chosen_cta
    history.append(chosen_cta)
    state["cta_history"] = history[-CTA_HISTORY_WINDOW:]
    return chosen_cta


def render_cta(text: str) -> str:
    url = os.environ.get("BOOK_URL", BOOK_URL)
    url_suffix = f" → {url}" if url else ""
    return (
        text.replace("{BOOK_TITLE}", BOOK_TITLE)
        .replace("{BOOK_AUTHOR}", BOOK_AUTHOR)
        .replace("{BOOK_URL}", url_suffix)
    )


def choose_hashtags(state: dict, pillar: str = "", platform: str = "instagram"):
    if platform.lower() == "bluesky":
        return []
    pillar_key = pillar if pillar in HASHTAG_CLUSTERS else "micro_philosophy"
    cluster = list(HASHTAG_CLUSTERS.get(pillar_key, HASHTAG_CLUSTERS["micro_philosophy"]))
    canonical_book = "#TheNineStitches"
    if canonical_book not in cluster:
        cluster.insert(0, canonical_book)
    if platform.lower() == "linkedin":
        k = random.randint(3, 5)
    elif platform.lower() in ["instagram", "threads", "pinterest", "youtube"]:
        k = 4
    else:
        k = random.randint(8, 12)
    pool = [t for t in cluster if t != canonical_book]
    k = max(1, min(k, 1 + len(pool)))
    sampled = random.sample(pool, k=max(0, k - 1))
    chosen = [canonical_book] + sampled
    state["last_hashtags"] = chosen
    return chosen


def _process_caption_output(caption: str, target_platform: str = "instagram") -> str:
    """Final surgical cleanup of markdown, hashtags, and leading/trailing junk symbols."""
    # 1. Initial strip of common AI artifacts and brackets
    text = caption.strip().strip('{}[]"\'' ' ')

    # 1a. Deterministic filter: strip banned lazy openers regardless of model output.
    banned_openers = ("ah, ", "ah yes", "ah, what a", "ah—", "ah.")
    first_line = text.splitlines()[0].strip().lower() if text.splitlines() else ""
    if any(first_line.startswith(b) for b in banned_openers):
        rest = [ln for ln in text.splitlines()[1:] if ln.strip()]
        if rest:
            text = "\n".join(rest).strip()
        else:
            if target_platform.lower() == "linkedin":
                text = "Here's the uncomfortable truth about stability: it dissolves faster than we admit.\n\n" + text
            else:
                text = "Stability is a story we tell ourselves.\n\n" + text

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
    """Attempt to parse broken JSON locally; no external LLM repair."""
    try:
        return _parse_posts_json_array(broken_text)
    except Exception:
        return []


def _generate_new_posts() -> List[Dict[str, Any]]:
    """Generate new post ideas deterministically from local templates."""
    pillars = ["micro_philosophy", "nature_metaphor", "systems_psychology", "author_voice", "quote"]
    templates = [
        ("micro_philosophy", "The Art of Starting Over", "abstract water droplets on stone, morning light"),
        ("nature_metaphor", "Ripples and Resonance", "calm lake surface at dawn, soft reflections"),
        ("systems_psychology", "The Feedback Loop We Live In", "minimalist circuit diagram merged with tree roots"),
        ("author_voice", "Notes from a Professional Failure Expert", "open notebook, handwritten, warm desk lamp"),
        ("quote", "Wabi-Sabi in the Age of Algorithms", "weathered ceramic, gold repair lines, textured paper"),
    ]
    posts = []
    for i in range(20):
        pillar, title, image_prompt = templates[i % len(templates)]
        posts.append({
            "pillar": pillar,
            "title": f"{title} #{i + 1}",
            "image_prompt": image_prompt,
            "caption_prompt": f"Reflect on {title.lower()}. What does it teach us about productive failure? #TheNineStitches",
        })
    return posts


# type: ignore[reportReturnType] — _is_image_censored can return "unknown" on OCR failures
def _is_image_censored(image_path: str) -> Union[bool, str]:
    """Checks if an image contains explicit censorship messages using OCR.space API.
    Returns True if censored, False if safe, or "unknown" if the check could not be completed.
    """
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
        # Network timeout: cannot verify safety. Mark as UNKNOWN and allow
        # the caller to decide — do NOT auto-reject the image on OCR timeout.
        print(f"OCR check timed out for {image_path}. Marking as unknown (will retry once).")
        return "unknown"
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else 0
        # 5xx server errors: unknown, not necessarily unsafe — allow retry.
        if 500 <= status < 600:
            print(f"OCR server error ({status}) for {image_path}. Marking as unknown (will retry once).")
            return "unknown"
        # 4xx client errors: assume safe and skip — the image is fine, OCR side is the problem.
        print(f"OCR client error ({status}) for {image_path}. Assuming safe, skipping check.")
        return False
    except Exception as e:
        # Unexpected errors: unknown, not necessarily unsafe.
        print(f"OCR check unexpected error: {e}. Marking as unknown (will retry once).")
        return "unknown"
    
    return False


def _generate_image_ai_horde(prompt: str) -> str:
    """Generates a high-quality cinematic image using the AI Horde API with SDXL models."""
    # Pre-check: is the horde reachable?
    if not _check_horde_health():
        raise RuntimeError("AI Horde health check failed — skipping image generation.")

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
    
    # Poll AI Horde every 5 minutes for up to 3 checks; avoids premature caption-only fallback
    for i in range(3):
        time.sleep(5 * 60)
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


def _weighted_post_choice(posts: list[dict], state: dict, platform: str = "instagram") -> dict:
    if not posts:
        raise RuntimeError(f"No posts available for weighted selection on {platform}.")
    active_series = state.get("active_series", {}).get(platform)
    if active_series:
        s_name = active_series.get("name")
        next_part = active_series.get("next_part", 1)
        series_match = None
        for post in posts:
            if post.get("series") == s_name and post.get("part") == next_part:
                series_match = post
                break
        if series_match:
            print(f"Continuing series '{s_name}' — Part {next_part}")
            return series_match
        state["active_series"][platform] = None
    if not state.get("active_series", {}).get(platform):
        new_series_candidates = [post.get("series") for post in posts if post.get("series") and post.get("part") == 1]
        if new_series_candidates and random.random() < 0.20:
            chosen_s = random.choice(new_series_candidates)
            for post in posts:
                if post.get("series") == chosen_s and post.get("part") == 1:
                    print(f"Starting new series: {chosen_s}")
                    state.setdefault("active_series", {}).setdefault(platform, {})
                    state["active_series"][platform] = {"name": chosen_s, "next_part": 1}
                    return post
    grouped: dict = {}
    for post in posts:
        pillar = str(post.get("pillar", "micro_philosophy") or "micro_philosophy").strip()
        grouped.setdefault(pillar, []).append(post)
    history_raw = state.get("pillar_history", [])
    if not isinstance(history_raw, list):
        history_raw = []
    history = [str(x) for x in history_raw if isinstance(x, str)]
    if len(history) > PILLAR_HISTORY_WINDOW:
        history = history[-PILLAR_HISTORY_WINDOW:]
        state["pillar_history"] = history
    candidates = [pillar for pillar in PILLAR_WEIGHTS.keys() if grouped.get(pillar)]
    if not candidates:
        return random.choice(posts)
    history_counts = {pillar: 0 for pillar in PILLAR_WEIGHTS.keys()}
    for pillar in history:
        if pillar in history_counts:
            history_counts[pillar] += 1
    window = max(1, min(PILLAR_HISTORY_WINDOW, len(history)))
    def _corrected_weight(pillar: str) -> float:
        base = PILLAR_WEIGHTS[pillar]
        if not history:
            return base
        expected = base * window
        actual = history_counts.get(pillar, 0)
        delta = expected - actual
        factor = max(0.55, min(1.45, 1.0 + (delta / max(1.0, window))))
        return max(0.001, base * factor)
    weights = [_corrected_weight(pillar) for pillar in candidates]
    total = sum(weights) or 1.0
    weights = [weight / total for weight in weights]
    chosen_pillar = random.choices(candidates, weights=weights, k=1)[0]
    last_pillar = str(state.get("last_pillar", "") or "").strip()
    if len(candidates) > 1 and chosen_pillar == last_pillar:
        alt_candidates = [pillar for pillar in candidates if pillar != last_pillar]
        alt_weights = [PILLAR_WEIGHTS[pillar] for pillar in alt_candidates]
        alt_total = sum(alt_weights) or 1.0
        alt_weights = [weight / alt_total for weight in alt_weights]
        chosen_pillar = random.choices(alt_candidates, weights=alt_weights, k=1)[0]
    chosen_post = random.choice(grouped[chosen_pillar])
    history.append(chosen_pillar)
    state["pillar_history"] = history[-PILLAR_HISTORY_WINDOW:]
    return chosen_post


def generate_image(prompt: str) -> str:
    """Generate image with retries and censorship checks."""
    MAX_RETRIES = 3
    unknown_retry_done = False
    for attempt in range(MAX_RETRIES):
        try:
            image_path = _generate_image_ai_horde(prompt)
            censorship = _is_image_censored(image_path)
            if censorship is True:
                print(f"Image attempt {attempt + 1} was censored. Retrying...")
                continue
            if censorship == "unknown":
                if unknown_retry_done:
                    # Second "unknown" — accept the image rather than looping forever.
                    print(f"OCR check unknown for {image_path} on retry; accepting image.")
                    return image_path
                unknown_retry_done = True
                print(f"OCR check unknown for {image_path}; one retry remaining before accepting.")
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


def generate_carousel(pillar: str, topic: str, timestamp: str, footer_text: str = "M.W.E. WIGMAN | THE NINE STITCHES", slides: Optional[List[str]] = None) -> List[str]:
    """
    Generate a 5-slide LinkedIn/Instagram carousel from a pillar/topic.
    Style: dark minimalist quote card — pure black framing bars, dark charcoal
    center, soft blurred orb, white sans-serif text centered, attribution footer.
    Slides follow high-performing carousel anatomy:
    - 4:5 ratio (1080x1350)
    - Slide 1 = hook question
    - Slide 2 = quick context/lens
    - Slide 3 = reframe or pivot
    - Slide 4 = actionable system/framework
    - Slide 5 = CTA / announcement
    Returns list of 5 image paths.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        import textwrap
    except Exception as e:
        print(f"Carousel generation skipped (missing PIL): {e}")
        return []

    pillar_title = pillar.replace('_', ' ').title()
    topic_clean = topic.strip().rstrip('.')
    if slides is None:
        slides = [
            f"What if {topic_clean}?",
            f"{pillar_title if pillar_title else topic_clean} is not what you think it is.",
            f"The {topic_clean} effect: small inputs create massive outcomes.",
            "The Nine Stitches approach: intent plus system beats motivation.",
            "The Nine Stitches\nOut now",
        ]
    base_dir = "images"
    paths: List[str] = []

    # Palette
    BG_TOP_BOTTOM = (8, 8, 8)
    BG_CENTER = (28, 30, 36)
    INSET_BORDER = (60, 64, 72)
    ORB_COLOR = (50, 58, 80)
    TEXT_COLOR = (240, 240, 240)
    FOOTER_COLOR = (160, 165, 175)

    def _load_sans(size: int):
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "DejaVuSans.ttf",
            "Arial.ttf",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _load_sans_bold(size: int):
        font_paths = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/tahomabd.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "DejaVuSans-Bold.ttf",
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
        img = Image.new("RGB", (w, h), BG_TOP_BOTTOM)

        # Central panel
        margin_x = 70
        margin_y = 170
        panel = (margin_x, margin_y, w - margin_x, h - margin_y)
        draw = ImageDraw.Draw(img)
        draw.rectangle(panel, fill=BG_CENTER)

        # Inset border
        inset = 18
        draw.rectangle(
            (
                panel[0] + inset,
                panel[1] + inset,
                panel[2] - inset,
                panel[3] - inset,
            ),
            outline=INSET_BORDER,
            width=2,
        )

        # Soft orb behind text
        orb_size = min(w, h) // 3
        orb_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        od = ImageDraw.Draw(orb_layer)
        od.ellipse(
            (
                (w - orb_size) / 2 - orb_size * 0.15,
                (h - orb_size) / 2 - orb_size * 0.05,
                (w + orb_size) / 2 + orb_size * 0.15,
                (h + orb_size) / 2 + orb_size * 0.15,
            ),
            fill=(*ORB_COLOR, 120),
        )
        orb_layer = orb_layer.filter(ImageFilter.GaussianBlur(radius=70))
        img = Image.alpha_composite(img.convert("RGBA"), orb_layer).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Font sizing
        is_cta = i == 4
        header_size = 68 if is_cta else 62
        footer_size = 28
        font = _load_sans(header_size)
        font_bold = _load_sans_bold(header_size)
        font_footer = _load_sans(footer_size)

        # Wrap text
        max_text_width = panel[2] - panel[0] - inset * 2 - 40
        raw = text.split("\n")[0]
        wrap_width = max(16, int(max_text_width / (header_size * 0.48)))
        wrapped_lines = textwrap.wrap(raw, width=wrap_width)

        # Recalculate metrics
        line_hs = []
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_hs.append(bbox[3] - bbox[1])
        line_h = max(line_hs) if line_hs else 40
        line_spacing = 26
        th = line_h * len(wrapped_lines) + line_spacing * max(0, len(wrapped_lines) - 1)

        # Place text block in the middle of the panel, above footer
        footer_space = 80 if i == 4 else 110
        available_h = (panel[3] - footer_space) - (panel[1] + 40)
        block_y = panel[1] + 40 + (available_h - th) / 2
        block_x = panel[0] + 40

        # Draw main quote
        for line_idx, line in enumerate(wrapped_lines):
            lf = font_bold if is_cta else font
            line_w = draw.textbbox((0, 0), line, font=lf)[2] - draw.textbbox((0, 0), line, font=lf)[0]
            line_x = block_x + (max_text_width - line_w) / 2
            line_y = block_y + line_idx * (line_h + line_spacing)
            draw.text((line_x, line_y), line, font=lf, fill=TEXT_COLOR)

        # Footer attribution (slides 1-4); slide 5 keeps the CTA text itself prominent
        if i < 4:
            footer_text = footer_text or "M.W.E. WIGMAN | THE NINE STITCHES"
            fbbox = draw.textbbox((0, 0), footer_text, font=font_footer)
            fw = fbbox[2] - fbbox[0]
            fh = fbbox[3] - fbbox[1]
            fx = panel[0] + (panel[2] - panel[0] - fw) / 2
            fy = panel[3] - inset - 20 - fh
            draw.text((fx, fy), footer_text, font=font_footer, fill=FOOTER_COLOR)

        img.save(out_path, format="JPEG", quality=95, optimize=True)
        paths.append(out_path)

    return paths


def generate_wilma_carousel(
    pillar: str,
    topic: str,
    timestamp: str,
    footer_text: str = "DIGITAL GUARDIAN | WILMA",
    slides: Optional[List[str]] = None,
) -> List[str]:
    """
    Wilma-specific carousel: textured cream paper background, centered serif text,
    no highlight, footnote footer. 1080x1350, 5 slides.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        import textwrap, random
    except Exception as e:
        print(f"Wilma carousel generation skipped (missing PIL): {e}")
        return []

    pillar_title = pillar.replace('_', ' ').title()
    topic_clean = topic.strip().rstrip('.')
    if slides is None:
        slides = [
            f"What if {topic_clean}?",
            f"{pillar_title if pillar_title else topic_clean} is not what you think it is.",
            f"The {topic_clean} effect: small inputs create massive outcomes.",
            "Build a routine, not a wall. Small consistency beats big restrictions.",
            "DIGITAL GUARDIAN | WILMA",
        ]

    base_dir = "images"
    paths: List[str] = []

    BG = (245, 242, 235)
    TEXT_COLOR = (18, 18, 18)
    FOOTER_COLOR = (90, 85, 78)

    def _load_serif(size: int):
        font_paths = [
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/timesbd.ttf",
            "C:/Windows/Fonts/georgia.ttf",
            "C:/Windows/Fonts/georgiab.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "DejaVuSerif.ttf",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _load_serif_bold(size: int):
        font_paths = [
            "C:/Windows/Fonts/timesbd.ttf",
            "C:/Windows/Fonts/georgiab.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "DejaVuSerif-Bold.ttf",
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
        img = Image.new("RGB", (w, h), BG)

        # subtle paper grain
        if i < 4:
            grain = Image.new("RGB", (w, h), BG)
            gdraw = ImageDraw.Draw(grain)
            for _ in range(18000):
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                shade = random.randint(-12, 12)
                px = tuple(max(0, min(255, c + shade)) for c in BG)
                gdraw.point((x, y), fill=px)
            grain = grain.filter(ImageFilter.GaussianBlur(radius=1.2))
            img = Image.blend(img, grain, alpha=0.18)

        draw = ImageDraw.Draw(img)

        is_cta = i == 4
        header_size = 66 if is_cta else 60
        footer_size = 26
        font = _load_serif(header_size)
        font_bold = _load_serif_bold(header_size)
        font_footer = _load_serif(footer_size)

        panel_x = 80
        panel_y = 160
        max_text_width = w - panel_x * 2
        raw = text.split("\n")[0]
        wrap_width = max(14, int(max_text_width / (header_size * 0.52)))
        wrapped_lines = textwrap.wrap(raw, width=wrap_width)

        line_hs = []
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font_bold if is_cta else font)
            line_hs.append(bbox[3] - bbox[1])
        line_h = max(line_hs) if line_hs else 38
        line_spacing = 24
        th = line_h * len(wrapped_lines) + line_spacing * max(0, len(wrapped_lines) - 1)

        footer_space = 90 if is_cta else 110
        available_h = (h - panel_y - footer_space) - (panel_y + 40)
        block_y = panel_y + 40 + (available_h - th) / 2
        block_x = panel_x

        for line_idx, line in enumerate(wrapped_lines):
            lf = font_bold if is_cta else font
            line_w = draw.textbbox((0, 0), line, font=lf)[2] - draw.textbbox((0, 0), line, font=lf)[0]
            line_x = block_x + (max_text_width - line_w) / 2
            line_y = block_y + line_idx * (line_h + line_spacing)
            draw.text((line_x, line_y), line, font=lf, fill=TEXT_COLOR)

        if is_cta:
            footer_text = footer_text or "DIGITAL GUARDIAN | WILMA"
            fbbox = draw.textbbox((0, 0), footer_text, font=font_footer)
            fw = fbbox[2] - fbbox[0]
            fh = fbbox[3] - fbbox[1]
            fx = (w - fw) / 2
            fy = h - panel_y - 20 - fh
            draw.text((fx, fy), footer_text, font=font_footer, fill=FOOTER_COLOR)

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
        return ImageFont.load_default()

    font_size = 75 if len(overlay) < 25 else 60
    font = _load_font(font_size)

    # Wrap by actual pixel width so the panel never exceeds image bounds.
    words = overlay.upper().split()
    max_text_width = w - 140
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        test_bbox = draw.textbbox((0, 0), test, font=font)
        if (test_bbox[2] - test_bbox[0]) > max_text_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    wrapped = "\n".join(lines)

    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=20, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # If text is too tall, reduce font and rewrap until it fits safe bounds.
    safe_max_h = h - 180
    while th > safe_max_h and font_size > 28:
        font_size -= 4
        font = _load_font(font_size)
        lines = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            test_bbox = draw.textbbox((0, 0), test, font=font)
            if (test_bbox[2] - test_bbox[0]) > max_text_width and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
        wrapped = "\n".join(lines)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=20, align="center")
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

    pad_x, pad_y = 60, 50
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2
    max_box_w = w - 40
    max_box_h = h - 40
    if box_w > max_box_w:
        box_w = max_box_w
    if box_h > max_box_h:
        box_h = max_box_h
    box_x = int((w - box_w) // 2)
    box_y = int((h - box_h) // 2 - (h * 0.05))
    box_x = max(20, min(box_x, w - box_w - 20))
    box_y = max(20, min(box_y, h - box_h - 20))

    overlay_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay_layer)
    odraw.rectangle((box_x, box_y, box_x + box_w, box_y + box_h), fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img.convert("RGBA"), overlay_layer).convert("RGB")
    draw = ImageDraw.Draw(img)

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

        # Save back as JPEG while preserving original background
        out = base.convert("RGB")
        out.save(image_path, format="JPEG", quality=95, optimize=True)
        return image_path
    except Exception as e:
        print(f"Logo watermark failed: {e}")
        return image_path


def _check_horde_health() -> bool:
    """Quick check whether the AI Horde is reachable and accepting requests.
    Returns True if the horde appears healthy, False otherwise.
    """
    try:
        r = requests.get("https://aihorde.net/api/v2/status/heartbeat", timeout=10)
        if r.status_code == 200:
            return True
        print(f"  AI Horde heartbeat returned status {r.status_code} — horde may be degraded.")
        return False
    except Exception as e:
        print(f"  AI Horde heartbeat failed: {e} — horde may be offline.")
        return False


def _get_available_horde_text_models() -> list[str]:
    api_key = os.environ.get("AI_HORDE_API_KEY", "0000000000")
    url = "https://aihorde.net/api/v2/status/models?type=text"
    headers = {"apikey": api_key}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
        models = [m.get("name") for m in data if m.get("name")]
        if models:
            print(f"  AI Horde text models detected: {len(models)} available")
        return models
    except Exception as e:
        print(f"  AI Horde text model discovery failed: {e}")
        return []


def _generate_text_ai_horde(prompt: str, system_prompt: str = "", max_tokens: int = 512) -> str:
    """Generates text using AI Horde."""
    # Pre-check: is the horde even reachable?
    if not _check_horde_health():
        print("  AI Horde health check failed. Skipping AI Horde text generation.")
        return ""

    full_prompt = f"### Instruction:\n{system_prompt}\n\n### Input:\n{prompt}\n\n### Response:\n"
    available_text_models = _get_available_horde_text_models()
    if not available_text_models:
        print("  No AI Horde text models available.")
        return ""

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
            print("  AI Horde text 403. Skipping text generation.")
            return ""
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
            print("  AI Horde text 403 after submit. Skipping text generation.")
            return ""
        print(f"  AI Horde text generation failed: {e}")
        raise
    except Exception as e:
        print(f"  AI Horde text generation failed: {e}.")
        return ""


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
    # We want to maintain a buffer of at least 3 posts ready to go.
    # With 4x/week generation, we create fewer bundles per run.
    target_buffer = 3
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

            # --- LINKEDIN CAROUSEL REMOVED: now handled by separate workflow ---
            bundle_carousel = []
            pending["carousel"] = bundle_carousel
            _save_pending(state, pending)

        except Exception as e:
            print(f"❌ Media generation failed: {e}. Progress saved, will resume next run.")
            _save_pending(state, pending)
            return  # Exit cleanly — main bot platforms need image/video

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

                try:
                    cta = choose_next_cta(state, preferred_category="engagement" if p.lower() == "linkedin" else None)
                    cta = render_cta(cta)
                    tags = choose_hashtags(state, post.get("pillar", ""), platform=p)
                    linkedin_comment = random.choice(LINKEDIN_COMMENT_PROMPTS) if p.lower() == "linkedin" else ""
                except Exception as _cta_exc:
                    print(f"  ⚠ CTA/hashtag setup failed: {_cta_exc}")
                    cta = ""
                    tags = []
                    linkedin_comment = ""

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
                final_cap = _strip_trailing_cta(tailored_cap.strip())

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
