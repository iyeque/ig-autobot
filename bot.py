import os
import sys
import time
import json
import uuid
import requests
from typing import Any, Dict, Optional, List
import PyPDF2
import base64
import dashscope
from dashscope import ImageSynthesis
import random
import io # Added for BytesIO

# Environment / config
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
DEEPAI_API_KEY = os.environ.get("DEEPAI_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY", "") # Added OCR_SPACE_API_KEY

CAPTION_FILE = "caption.txt"
OUTPUT_IMAGE = "output.jpg"
MAX_BOOK_CONTEXT_CHARS = 2000

# Book-specific constants
BOOK_TITLE = os.environ.get("BOOK_TITLE", "The Nine Stitches")
BOOK_AUTHOR = os.environ.get("BOOK_AUTHOR", "M.W.E. Wigman")


def sanitize_image_prompt(prompt: str) -> str:
    """
    Sanitize prompt for better AI generation success.
    Removes problematic terms, simplifies complex concepts.
    """
    # Replace problematic biological terms with safer alternatives
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
        "woman": "figure"
    }
    
    clean_prompt = prompt
    for old, new in replacements.items():
        clean_prompt = clean_prompt.replace(old, new)
        clean_prompt = clean_prompt.replace(old.title(), new.title())
    
    # Ensure it's not too long (max 500 chars for most APIs)
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
    insights = {
        "central_question": "What happens if you try to fail and succeed?",
        "epigraph": "To become, be calm. To be calm, pretend to be calm.",
        "chapters": [
            {"number": 1, "title": "The One in Time", "theme": "Intention vs. Outcome"},
            {"number": 2, "title": "If you can't evade it, embrace it", "theme": "Adversity and Growth"},
            {"number": 3, "title": "The Elegance of Flaws", "theme": "Imperfection and Creativity"},
            {"number": 4, "title": "Microcosm and Macrocosm", "theme": "Individual and Collective"}
        ],
        "key_concepts": [
            "intention vs outcome",
            "productive failure",
            "adversity-growth cycles",
            "antifragility",
            "wabi-sabi",
            "kintsugi",
            "keystone species",
            "butterfly effect",
            "bioluminescent defense",
            "lizard autotomy",
            "serotinous cones"
        ]
    }
    return insights


# -------------------------
# Persistence helpers
# -------------------------
def _read_posts() -> List[Dict[str, Any]]:
    try:
        if os.path.exists("posts.json"):
            with open("posts.json", "r", encoding="utf-8") as f:
                return json.load(f)
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
                BOOK_HASHTAG = f"#{BOOK_TITLE.replace(' ', '')}"
                
                POTENTIAL_HASHTAGS = [
                    "#Bookstagram", "#AmReading", "#Bookworm", "#Booklover", 
                    "#WritersOfInstagram", "#AmWriting", "#WritersCommunity",
                    "#ProductiveFailure", "#IntentionVsOutcome", "#AdversityAndGrowth",
                    "#Antifragility", "#WabiSabi", "#Kintsugi", "#PhilosophyOfLife",
                    "#DeepThoughts", "#BookishThoughts", "#LiteraryLife", "#IndieAuthor"
                ]

                selected_hashtags = [BOOK_HASHTAG]
                # Ensure unique hashtags and add up to 4 more
                remaining_hashtags = [h for h in POTENTIAL_HASHTAGS if h != BOOK_HASHTAG]
                selected_hashtags.extend(random.sample(remaining_hashtags, k=min(len(remaining_hashtags), random.randint(3, 4))))
                
                # Append hashtags if not already in caption
                caption_lines = caption.split('\n')
                caption_without_hashtags = []
                existing_hashtags = set()

                for line in caption_lines:
                    # Very simple check for lines that are solely hashtags
                    if line.strip().startswith('#') and ' ' not in line.strip():
                        existing_hashtags.add(line.strip())
                    else:
                        caption_without_hashtags.append(line)
                
                final_caption = "\n".join(caption_without_hashtags).strip()
                
                # Add only new selected hashtags that aren't already present
                new_hashtags_to_add = [h for h in selected_hashtags if h not in existing_hashtags]
                
                if new_hashtags_to_add:
                    final_caption += "\n\n" + " ".join(new_hashtags_to_add)

                return final_caption

        raise RuntimeError(f"Cerebras API returned an unexpected response format: {data}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling Cerebras API: {e}")
        raise RuntimeError(f"Failed to generate caption with Cerebras. Last error: {e}")


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
    
    Generate a list of 30 new Instagram post ideas. Each post must be a JSON object with:
    - "pillar": one of ["micro_philosophy", "nature_metaphor", "systems_psychology", "author_voice", "quote"]
    - "title": short, evocative phrase referencing specific book concepts
    - "image_prompt": detailed description for AI image generation (avoid human figures, use abstract/nature imagery)
    - "caption_prompt": detailed instruction mentioning specific book concepts, ending with question and #{BOOK_TITLE.replace(' ', '')} hashtag
    
    Return ONLY a valid JSON list of these 10 objects, no other text.
    """

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": f"You are a creative assistant that generates JSON data for {BOOK_TITLE} Instagram bot."},
            {"role": "user", "content": meta_prompt}
        ],
        "temperature": 0.8,
        "max_tokens": 2500
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()

        if data.get("choices") and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            content = message.get("content", "").strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            new_posts = json.loads(content)
            if isinstance(new_posts, list) and len(new_posts) > 0:
                print(f"Successfully generated {len(new_posts)} new posts.")
                return new_posts

        raise RuntimeError(f"Cerebras API for new posts returned an unexpected format: {data}")

    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Error generating new posts with Cerebras: {e}")
        raise RuntimeError(f"Failed to generate new posts. Last error: {e}")

def _is_image_censored(image_path: str) -> bool:
    """
    Checks if an image contains explicit censorship messages using OCR.space API.
    Returns True if censored content is detected, False otherwise.
    """
    if not OCR_SPACE_API_KEY:
        print("Warning: OCR_SPACE_API_KEY is not set. Cannot check for censored images.")
        return False

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Check image size as a preliminary heuristic. Very small images are likely placeholders/errors.
        if len(image_data) < 5000: # 5KB threshold, adjust if necessary
            print(f"Image {image_path} is very small ({len(image_data)} bytes), likely a placeholder or error.")
            return True

        headers = {"apikey": OCR_SPACE_API_KEY}
        payload = {
            "isOverlayRequired": False,
            "detectOrientation": False,
            "scale": True,
            "OCREngine": 2 # 2 for better accuracy on digital text
        }
        files = {"file": ("image.jpg", image_data, "image/jpeg")}

        print(f"Sending {image_path} to OCR.space for censorship check...")
        response = requests.post("https://api.ocr.space/parse/image",
                                 headers=headers,
                                 data=payload,
                                 files=files,
                                 timeout=60)
        response.raise_for_status()
        result = response.json()

        if result.get("IsErroredOnProcessing"):
            print(f"OCR.space processing error: {result.get('ErrorMessage')}")
            return False # Treat OCR error as not censored for now

        parsed_text = ""
        if result.get("ParsedResults"):
            for parsed_result in result["ParsedResults"]:
                if parsed_result.get("ParsedText"):
                    parsed_text += parsed_result["ParsedText"] + " "
        
        parsed_text = parsed_text.lower()
        print(f"OCR detected text: {parsed_text[:200]}...") # Log first 200 chars

        # Keywords to look for in censored images
        if "censored" in parsed_text or \
           "nsfw content detected" in parsed_text or \
           "blocked by client request" in parsed_text:
            print(f"Censorship text detected in {image_path}")
            return True

    except requests.exceptions.RequestException as e:
        print(f"Error calling OCR.space API: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during OCR check: {e}")
    
    return False

# -------------------------
# Image generation - FOUR FALLBACKS
# -------------------------
def _generate_image_qwen(prompt: str) -> str:
    """Generates an image using the Qwen (DashScope) API."""
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY not set")

    dashscope.api_key = DASHSCOPE_API_KEY
    
    clean_prompt = sanitize_image_prompt(prompt)
    print(f"Qwen prompt (sanitized): {clean_prompt[:100]}...")

    try:
        response = ImageSynthesis.call(
            model='qwen-image-max',
            prompt=clean_prompt,
            n=1,
            size='1080*1350'
        )

        if response.status_code == 200:
            image_url = response.output.results[0].url
            print(f"Image generated successfully: {image_url}")
            
            img_response = requests.get(image_url, timeout=120)
            img_response.raise_for_status()
            with open(OUTPUT_IMAGE, "wb") as f:
                f.write(img_response.content)
            print("Saved Qwen image")
            return OUTPUT_IMAGE
        else:
            raise RuntimeError(f"Failed to generate image. Status Code: {response.status_code}, Message: {response.message}")

    except Exception as e:
        raise RuntimeError(f"Qwen failed: {e}")


def _generate_image_ai_horde(prompt: str) -> str:
    """Generates an image using the AI Horde API."""
    url = "https://stablehorde.net/api/v2/generate/async"
    api_key = os.environ.get("AI_HORDE_API_KEY", "0000000000")
    
    clean_prompt = sanitize_image_prompt(prompt)
    print(f"AI Horde prompt (sanitized): {clean_prompt[:100]}...")

    payload = {
        "prompt": clean_prompt,
        "params": {
            "sampler_name": "k_dpm_2_a",
            "cfg_scale": 7.5,
            "width": 1024,
            "height": 1280,
            "steps": 20,  # Reduced for speed
        },
        "models": ["stable_diffusion"],  # More common model
        "nsfw": False
    }
    
    headers = {"apikey": api_key, "Content-Type": "application/json"}
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    request_id = response.json().get("id")

    if not request_id:
        raise RuntimeError("AI Horde did not return a request ID")

    print(f"AI Horde request submitted: {request_id}")

    check_url = f"https://stablehorde.net/api/v2/generate/check/{request_id}"
    status_url = f"https://stablehorde.net/api/v2/generate/status/{request_id}"
    
    max_checks = 30  # 5 minutes max
    checks_after_done = 0
    max_after_done = 3

    for i in range(max_checks):
        time.sleep(10)
        status_response = requests.get(check_url, timeout=30)
        status_response.raise_for_status()
        status_data = status_response.json()
        
        if status_data.get("done"):
            print(f"Generation complete (check {i+1})")
            
            status_response = requests.get(status_url, timeout=30)
            status_response.raise_for_status()
            full_status = status_response.json()
            
            generations = full_status.get("generations", [])
            
            if generations:
                for gen in generations:
                    if gen.get("state") == "ok":
                        img_data = gen.get("img")
                        if not img_data:
                            continue
                        
                        try:
                            if img_data.startswith("http"):
                                img_response = requests.get(img_data, timeout=120)
                                img_response.raise_for_status()
                                with open(OUTPUT_IMAGE, "wb") as f:
                                    f.write(img_response.content)
                                print(f"Saved image from URL")
                                return OUTPUT_IMAGE
                            else:
                                if img_data.startswith("data:"):
                                    img_data = img_data.split(",", 1)[1]
                                img_bytes = base64.b64decode(img_data)
                                with open(OUTPUT_IMAGE, "wb") as f:
                                    f.write(img_bytes)
                                print(f"Saved decoded image")
                                return OUTPUT_IMAGE
                        except Exception as e:
                            print(f"Failed to process: {e}")
                            continue
                
                raise RuntimeError("No valid image in generations")
            
            elif checks_after_done < max_after_done:
                checks_after_done += 1
                print(f"Empty generations, retry {checks_after_done}/{max_after_done}")
                continue
            else:
                raise RuntimeError("No image URL after retries")
        
        if i % 6 == 0:
            print(f"Polling... {i+1}/{max_checks}")
            
    raise RuntimeError("AI Horde generation timed out")


def _generate_image_deep_ai(prompt: str) -> str:
    """Generates an image using DeepAI with retry logic."""
    if not DEEPAI_API_KEY:
        raise RuntimeError("DEEPAI_API_KEY not set")

    url = "https://api.deepai.org/api/text2img"
    headers = {"api-key": DEEPAI_API_KEY}
    
    clean_prompt = sanitize_image_prompt(prompt)
    print(f"DeepAI prompt (sanitized): {clean_prompt[:100]}...")
    
    data = {
        "text": clean_prompt,
        "width": 1024,
        "height": 1280
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"DeepAI attempt {attempt + 1}/{max_retries}")
            response = requests.post(url, headers=headers, data=data, timeout=120)
            response.raise_for_status()
            result = response.json()

            image_url = result.get("output_url")
            if not image_url:
                raise RuntimeError(f"No output_url: {result}")

            for dl_attempt in range(3):
                try:
                    img_response = requests.get(image_url, timeout=120)
                    img_response.raise_for_status()
                    with open(OUTPUT_IMAGE, "wb") as f:
                        f.write(img_response.content)
                    print(f"Saved DeepAI image")
                    return OUTPUT_IMAGE
                except requests.exceptions.RequestException as e:
                    if dl_attempt < 2:
                        wait = 2 ** dl_attempt
                        print(f"Download failed, retry in {wait}s")
                        time.sleep(wait)
                    else:
                        raise

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Request failed: {e}, retry in {wait_time}s")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed after {max_retries}: {e}")
        except Exception as e:
            raise RuntimeError(f"DeepAI error: {e}")

    raise RuntimeError("DeepAI failed all attempts")


def _generate_image_pollinations(prompt: str) -> str:
    """
    THIRD FALLBACK: Pollinations.ai - free, fast, reliable.
    Replaces Craiyon (better quality, more reliable).
    """
    from urllib.parse import quote  # Local import to avoid circular issues
    
    print("=== Attempt 3: Pollinations.ai ===")
    
    clean_prompt = sanitize_image_prompt(prompt)
    # Pollinations works better with simpler prompts
    if len(clean_prompt) > 300:
        clean_prompt = clean_prompt[:297] + "..."
    
    # URL encode the prompt
    encoded = quote(clean_prompt)
    
    # Build URL with parameters for consistency
    seed = int(time.time()) % 10000
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1350&seed={seed}&nologo=true&enhance=false"
    
    print(f"URL: https://image.pollinations.ai/prompt/[encoded]?...")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Verify we got an image, not an error page
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            # Might be JSON error
            try:
                error_data = response.json()
                raise RuntimeError(f"Pollinations API error: {error_data}")
            except:
                raise RuntimeError(f"Unexpected content type: {content_type}")
        
        with open(OUTPUT_IMAGE, "wb") as f:
            f.write(response.content)
        
        file_size = os.path.getsize(OUTPUT_IMAGE)
        if file_size < 1000:
            raise RuntimeError(f"Image too small ({file_size} bytes), likely error")
        
        print(f"Saved Pollinations image ({file_size} bytes)")
        return OUTPUT_IMAGE
        
    except requests.exceptions.Timeout:
        raise RuntimeError("Pollinations timeout (60s)")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Pollinations request failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Pollinations failed: {e}")


def generate_image(prompt: str) -> str:
    """
    Generate image with FOUR fallbacks, incorporating retry logic for censored images.
    1. Qwen (primary)
    2. AI Horde (best quality, slow)
    3. DeepAI (good quality, needs payment)
    4. Pollinations.ai (free, fast, reliable)
    """
    errors = []
    MAX_RETRIES_PER_API = 2 # Allow each API to retry once if it generates a censored image

    api_funcs = [
        (_generate_image_qwen, "Qwen"),
        (_generate_image_ai_horde, "AI Horde"),
        (_generate_image_deep_ai, "DeepAI"),
        (_generate_image_pollinations, "Pollinations.ai")
    ]

    for api_func, api_name in api_funcs:
        for attempt in range(MAX_RETRIES_PER_API):
            try:
                print(f"=== Attempt {api_name} (retry {attempt + 1}/{MAX_RETRIES_PER_API}) ===")
                image_path = api_func(prompt) # This function saves the image to OUTPUT_IMAGE

                if _is_image_censored(image_path):
                    print(f"!!! {api_name} generated a censored image. Retrying...")
                    # Optional: Modify prompt slightly for retry, though for now we rely on API randomness
                    continue # Try again with the same API
                
                print(f"Generated uncensored image with {api_name}.")
                return image_path # Successfully generated and uncensored image

            except Exception as e:
                errors.append(f"{api_name} (attempt {attempt + 1}): {str(e)[:100]}")
                print(f"{api_name} failed on attempt {attempt + 1}: {e}")
                break # Break retry loop for this API, move to next fallback

    # All failed
    error_msg = "All image services failed after retries:\n" + "\n".join(errors)
    raise RuntimeError(error_msg)


# -------------------------
# Main flow
# -------------------------
def main():
    pdf_file_path = os.environ.get("PDF_BOOK_FILENAME", "The-Nine-Stitches.pdf")
    print(f"Using PDF: {pdf_file_path}")
    
    book_raw_text = extract_text_from_pdf(pdf_file_path)
    book_context = ""
    book_insights = None
    
    if book_raw_text:
        book_context = book_raw_text[:MAX_BOOK_CONTEXT_CHARS]
        book_insights = extract_book_insights(book_raw_text)
        print(f"Loaded {len(book_context)} chars of context.")
        print(f"Book: {book_insights['central_question']}")
    else:
        print("No PDF context loaded.")

    all_posts = _read_posts()
    state = _read_state()
    used_ids = set(state.get("used_ids", []))
    available_posts = [post for post in all_posts if post.get("id") not in used_ids]

    if not available_posts:
        print("All posts used. Generating new batch...")
        new_posts = _generate_new_posts()
        
        max_id = max((post.get("id", 0) for post in all_posts), default=0)
        
        for i, post in enumerate(new_posts):
            post["id"] = max_id + i + 1
            all_posts.append(post)

        _write_posts(all_posts)
        print(f"Added {len(new_posts)} new posts.")

        state["used_ids"] = []
        used_ids = set()
        available_posts = new_posts

    if not available_posts:
        raise RuntimeError("No posts available.")

    import random
    post = random.choice(available_posts)
    post_id = post.get("id")

    if post_id is None:
        raise RuntimeError(f"Selected post has no ID: {post}")

    print(f"Selected post {post_id}: {post.get('title', 'Untitled')}")
    print(f"Image prompt: {post['image_prompt'][:80]}...")
    print(f"Caption prompt: {post['caption_prompt'][:80]}...")

    state["used_ids"].append(post_id)
    _write_state(state)
    print(f"Updated state with used_id: {post_id}")

    # Generate caption
    try:
        caption = generate_caption(post["caption_prompt"], book_context, book_insights)
        print(f"\nCaption ({len(caption)} chars):\n{caption[:150]}...")
        with open(CAPTION_FILE, "w", encoding="utf-8") as f:
            f.write(caption)
    except Exception as e:
        print(f"Caption generation failed: {e}")
        raise

    # Generate image
    try:
        image_path = generate_image(post["image_prompt"])
        print(f"Image saved: {image_path}")
    except Exception as e:
        print(f"Image generation failed: {e}")
        raise

    print("✓ Done.")


if __name__ == "__main__":
    main()
