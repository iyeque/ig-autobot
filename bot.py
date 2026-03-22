import os
import sys
import time
import json
import uuid

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

def sanitize_image_prompt(prompt: str) -> str:
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
    
    clean_prompt = prompt
    for old, new in replacements.items():
        clean_prompt = clean_prompt.replace(old, new)
        clean_prompt = clean_prompt.replace(old.title(), new.title())
    
    if len(clean_prompt) > 500:
        clean_prompt = clean_prompt[:497] + "..."
    
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
def _read_posts() -> List[Dict[str, Any]]:
    try:
        if os.path.exists("posts.json"):
            with open("posts.json", "r", encoding="utf-8") as f:
                posts = json.load(f)
                
                # Deduplicate existing posts by title to prevent 'Groundhog Day'
                unique_posts = []
                seen_titles = set()
                for p in posts:
                    title_norm = p.get("title", "").strip().lower()
                    if title_norm and title_norm not in seen_titles:
                        unique_posts.append(p)
                        seen_titles.add(title_norm)
                    elif not title_norm:
                        unique_posts.append(p) # Keep if no title for some reason
                
                if len(unique_posts) < len(posts):
                    print(f"Deduplicated posts.json: {len(posts)} -> {len(unique_posts)}")
                    # We don't write here to avoid side effects during read, 
                    # but the in-memory list is now clean.
                return unique_posts
    except Exception as e:
        print(f"Error reading posts.json: {e}")
    return []


def _read_state() -> Dict[str, Any]:
    try:
        if os.path.exists("state.json"):
            with open("state.json", "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading state.json: {e}")
    return {"used_ids": []}


def _write_state(state: Dict[str, Any]) -> None:
    try:
        with open("state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Error writing state.json: {e}")


def _write_posts(posts: List[Dict[str, Any]]) -> None:
    try:
        with open("posts.json", "w", encoding="utf-8") as f:
            json.dump(posts, f, indent=2)
    except Exception as e:
        print(f"Error writing posts.json: {e}")


# -------------------------
# Caption generation
# -------------------------
def generate_caption(caption_prompt: str, book_context: str = "", book_insights: Optional[Dict] = None) -> str:
    """Generates a caption using the Cerebras API with book-aware context."""
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

Write concise Instagram captions (150-300 words) that blend philosophical depth with accessibility.
Use nature metaphors, reference specific book concepts when relevant, and always end with an engaging question.
Include 3-5 hashtags with #{BOOK_TITLE.replace(' ', '')} always first."""

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
        "max_tokens": 500
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
                
                # --- Dynamic Hashtag Generation ---
                DEFAULT_HASHTAGS = [
                    "#AmWriting", "#AmReading", "#WritersOfInstagram", 
                    "#LiteraryLife", "#Bookstagram", "#IndieAuthor"
                ]
                
                POTENTIAL_HASHTAGS = [
                    "#Bookworm", "#Booklover", "#WritersCommunity",
                    "#ProductiveFailure", "#IntentionVsOutcome", "#AdversityAndGrowth",
                    "#Antifragility", "#WabiSabi", "#Kintsugi", "#PhilosophyOfLife",
                    "#DeepThoughts", "#BookishThoughts"
                ]

                # Use defaults
                selected_hashtags = list(DEFAULT_HASHTAGS)
                
                # Add 2-3 random ones from the remaining potential list
                remaining_hashtags = [h for h in POTENTIAL_HASHTAGS if h not in selected_hashtags]
                selected_hashtags.extend(random.sample(remaining_hashtags, k=min(len(remaining_hashtags), random.randint(2, 3))))
                
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
                
                # Add only new selected hashtags that aren't already present (case-insensitive check)
                new_hashtags_to_add = [h for h in selected_hashtags if h.lower() not in existing_hashtags]
                
                if new_hashtags_to_add:
                    final_caption += "\n\n" + " ".join(new_hashtags_to_add)

                return final_caption

        raise RuntimeError(f"Cerebras API returned an unexpected response format: {data}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling Cerebras API: {e}")
        raise RuntimeError(f"Failed to generate caption with Cerebras. Last error: {e}")


def _strip_json_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
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
    """Generates an image using the AI Horde API."""
    url = "https://stablehorde.net/api/v2/generate/async"
    api_key = os.environ.get("AI_HORDE_API_KEY", "0000000000")
    
    clean_prompt = sanitize_image_prompt(prompt)
    print(f"AI Horde prompt: {clean_prompt[:100]}...")

    payload = {
        "prompt": clean_prompt,
        "params": {
            "sampler_name": "k_dpm_2_a",
            "cfg_scale": 7.5,
            "width": 1088,
            "height": 1344,
            "steps": 25,
        },
        "models": ["stable_diffusion"],
        "nsfw": False
    }
    
    headers = {"apikey": api_key, "Content-Type": "application/json"}
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
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
            time.sleep(5)
    raise RuntimeError("Failed to generate a valid image after retries.")


def generate_images_batch(prompt: str, n: int) -> List[str]:
    """Generates a batch of images with varied prompts, gracefully handling failures."""
    paths: List[str] = []
    modifiers = [
        "macro photography, extreme detail",
        "wide angle, atmospheric perspective",
        "abstract interpretation, ethereal lighting",
        "minimalist composition, high contrast",
        "soft focus, cinematic bokeh",
        "long exposure, dreamlike quality"
    ]
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
    pdf_file_path = os.environ.get("PDF_BOOK_FILENAME", "The-Nine-Stitches.pdf")
    print(f"Using PDF: {pdf_file_path}")
    
    book_raw_text = extract_text_from_pdf(pdf_file_path)
    book_context = book_raw_text[:MAX_BOOK_CONTEXT_CHARS] if book_raw_text else ""
    book_insights = extract_book_insights(book_raw_text) if book_raw_text else None

    all_posts = _read_posts()
    state = _read_state()
    used_ids = set(state.get("used_ids", []))
    
    # Map used IDs to their titles for title-based filtering
    used_titles = set()
    for p in all_posts:
        if p.get("id") in used_ids:
            t = p.get("title", "").strip().lower()
            if t: used_titles.add(t)

    # Available posts must have unique ID AND unique title
    available_posts = []
    for p in all_posts:
        p_id = p.get("id")
        p_title = p.get("title", "").strip().lower()
        if p_id not in used_ids and p_title not in used_titles:
            available_posts.append(p)
            # Add to used_titles so we don't pick two duplicates in the same batch
            used_titles.add(p_title)

    if not available_posts:
        print("All unique posts used. Generating new batch...")
        new_posts = _generate_new_posts()
        max_id = max((post.get("id", 0) for post in all_posts), default=0)
        for i, post in enumerate(new_posts):
            post["id"] = max_id + i + 1
            all_posts.append(post)
        _write_posts(all_posts)
        available_posts = new_posts

    post = random.choice(available_posts)
    post_id = post.get("id")
    print(f"Selected post {post_id}: {post.get('title', 'Untitled')}")

    state["used_ids"].append(post_id)
    _write_state(state)

    # Generate caption
    try:
        caption = generate_caption(post["caption_prompt"], book_context, book_insights)
        with open(CAPTION_FILE, "w", encoding="utf-8") as f:
            f.write(caption)
    except Exception as e:
        print(f"Caption generation failed: {e}")
        raise

    # Generate image(s)
    try:
        # Clean up old flags to prevent 'Groundhog Day' repetitions
        for f in ["carousel.json", "post_story.flag"]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Cleaned up old {f}")

        total_done = len(state["used_ids"])
        # Every 5th post is a carousel, every 7th post is a story-ready image
        make_carousel = (total_done % 5 == 0)
        make_story = (total_done % 7 == 0)
        
        # Signal to workflow via file existence
        if make_story:
            with open("post_story.flag", "w") as f: f.write("true")

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
            print(f"Image saved and normalized: {processed_path}")
            
    except Exception as e:
        print(f"Image generation failed: {e}")
        raise

    print("✓ Done.")


if __name__ == "__main__":
    main()
