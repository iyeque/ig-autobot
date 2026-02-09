#!/usr/bin/env python3
"""
Robust IG autobot using Cerebras AI for captions and AI Horde (with DeepAI fallback) for images.
"""

import os
import sys
import time
import json
import requests
from typing import Any, Dict, Optional, List
import PyPDF2 # New import

# Environment / config
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
DEEPAI_API_KEY = os.environ.get("DEEPAI_API_KEY", "")

CAPTION_FILE = "caption.txt"
OUTPUT_IMAGE = "output.jpg"
MAX_BOOK_CONTEXT_CHARS = 2000 # Define max characters from book to use as context

# PDF Text Extraction
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a given PDF file.
    """
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
def generate_caption(caption_prompt: str, book_context: str = "") -> str:
    """
    Generates a caption using the Cerebras API, optionally with book context.
    """
    if not CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY is not set in the environment")

    url = "https://api.cerebras.ai/v1/chat/completions"
    model_name = "llama3.1-8b"

    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }

    # Prepend book context to the caption prompt if available
    full_prompt = caption_prompt
    if book_context:
        full_prompt = f"Using the following context from the book 'The Nine Stitches':\n\n```\n{book_context}\n```\n\n{caption_prompt}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a concise Instagram caption writer. Generate captions relevant to the provided book context, if any."},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 180
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
                return caption

        raise RuntimeError(f"Cerebras API returned an unexpected response format: {data}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling Cerebras API: {e}")
        raise RuntimeError(f"Failed to generate caption with Cerebras. Last error: {e}")


def _generate_new_posts() -> List[Dict[str, Any]]:
    """
    Generates a new list of post prompts using the Cerebras API.
    """
    if not CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY is not set in the environment for prompt generation.")

    url = "https://api.cerebras.ai/v1/chat/completions"
    model_name = "llama3.1-8b"

    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }

    meta_prompt = """
    You are an AI assistant for an author whose work explores themes of nature as a metaphor, systems thinking, duality, and human psychology.
    Generate a list of 10 new Instagram post ideas. Each post must be a JSON object with the following fields: "pillar", "title", "image_prompt", and "caption_prompt".
    The "pillar" can be one of: "micro_philosophy", "nature_metaphor", "systems_psychology", or "author_voice".
    The "title" should be a short, evocative phrase.
    The "image_prompt" should be a description for an AI image generator to create a minimal, philosophical, and aesthetic image.
    The "caption_prompt" should be a detailed instruction for an AI caption writer.
    Return ONLY a valid JSON list of these 10 objects, with no other text before or after the list.
    """

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a creative assistant that generates JSON data for an Instagram bot."},
            {"role": "user", "content": meta_prompt}
        ],
        "temperature": 0.8,
        "max_tokens": 2048
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()

        if data.get("choices") and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            content = message.get("content", "").strip()
            
            # Clean the response to ensure it's valid JSON
            # The model might sometimes include markdown ```json ... ```
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




# -------------------------
# Image generation
# -------------------------
def _generate_image_ai_horde(prompt: str) -> str:
    """
    Generates an image using the AI Horde API.
    """
    url = "https://stablehorde.net/api/v2/generate/async"
    api_key = "0000000000"  # Anonymous key

    payload = {
        "prompt": prompt,
        "params": {
            "sampler_name": "k_dpm_2_a",
            "cfg_scale": 7.5,
            "width": 512,
            "height": 512,
            "steps": 25,
        }
    }
    
    headers = {"apikey": api_key, "Content-Type": "application/json"}
    
    # 1. Submit the request
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    request_id = response.json().get("id")

    if not request_id:
        raise RuntimeError("AI Horde did not return a request ID")

    print(f"AI Horde request submitted with ID: {request_id}")

    # 2. Poll for the result
    check_url = f"https://stablehorde.net/api/v2/generate/check/{request_id}"
    for _ in range(30):  # Poll for up to 5 minutes (30 * 10s)
        time.sleep(10)
        status_response = requests.get(check_url, timeout=30)
        status_response.raise_for_status()
        status_data = status_response.json()
        
        if status_data.get("done"):
            print("AI Horde generation is complete.")
            generations = status_data.get("generations", [])
            if generations:
                image_url = generations[0].get("img")
                if image_url:
                    # 3. Download the image
                    img_response = requests.get(image_url, timeout=120)
                    img_response.raise_for_status()
                    with open(OUTPUT_IMAGE, "wb") as f:
                        f.write(img_response.content)
                    return OUTPUT_IMAGE
            raise RuntimeError("AI Horde generation finished but no image URL was found.")

    raise RuntimeError("AI Horde generation timed out.")


def _generate_image_deep_ai(prompt: str) -> str:
    """
    Generates an image using the DeepAI API.
    """
    if not DEEPAI_API_KEY:
        raise RuntimeError("DEEPAI_API_KEY is not set in the environment")

    import deepai
    deepai.set_api_key(DEEPAI_API_KEY)
    
    try:
        response = deepai.api.text2image(prompt)
        image_url = response.output_url
        
        # Download the image
        img_response = requests.get(image_url, timeout=120)
        img_response.raise_for_status()
        with open(OUTPUT_IMAGE, "wb") as f:
            f.write(img_response.content)
        return OUTPUT_IMAGE

    except Exception as e:
        raise RuntimeError(f"DeepAI API call failed: {e}")


def generate_image(prompt: str) -> str:
    """
    Generate an image using AI Horde, with DeepAI as a fallback.
    Returns path to saved image (OUTPUT_IMAGE) or raises.
    """
    try:
        print("Attempting to generate image with AI Horde...")
        return _generate_image_ai_horde(prompt)
    except Exception as e_horde:
        print(f"AI Horde failed: {e_horde}")
        print("Falling back to DeepAI...")
        try:
            return _generate_image_deep_ai(prompt)
        except Exception as e_deepai:
            print(f"DeepAI also failed: {e_deepai}")
            raise RuntimeError("All image generation services failed.")


# -------------------------
# Main flow (example)
# -------------------------
def main():
    pdf_file_path = os.environ.get("PDF_BOOK_FILENAME", "The-Nine-Stitches.pdf")
    print(f"Using PDF book: {pdf_file_path}")
    # Extract book content for context
    book_raw_text = extract_text_from_pdf(pdf_file_path)
    book_context = ""
    if book_raw_text:
        book_context = book_raw_text[:MAX_BOOK_CONTEXT_CHARS]
        print(f"Loaded {len(book_context)} characters of book context.")
    else:
        print("No book context loaded from PDF.")

    # Load all posts and the current state
    all_posts = _read_posts()
    state = _read_state()

    # Get IDs of posts that have already been used
    used_ids = set(state.get("used_ids", []))

    # Filter out posts that have already been used
    available_posts = [post for post in all_posts if post.get("id") not in used_ids]

    # If all posts have been used, generate new ones
    if not available_posts:
        print("All posts have been used. Generating new posts...")
        new_posts = _generate_new_posts()
        
        # Find the highest existing ID to ensure new IDs are unique
        max_id = 0
        if all_posts:
            max_id = max(post.get("id", 0) for post in all_posts)

        # Assign new IDs and append to the main post list
        for i, post in enumerate(new_posts):
            post["id"] = max_id + i + 1
            all_posts.append(post)

        # Write the updated full list of posts back to posts.json
        _write_posts(all_posts)
        print(f"Appended {len(new_posts)} new posts to posts.json")

        # Reset the state and make the new posts available for selection
        state["used_ids"] = []
        used_ids = set()
        available_posts = new_posts

    if not available_posts:
        raise RuntimeError("No posts available in posts.json even after attempting to generate new ones. Exiting.")

    # Select a post (e.g., randomly)
    import random
    post = random.choice(available_posts)
    post_id = post.get("id")

    if post_id is None:
        raise RuntimeError(f"Selected post has no 'id' field: {post}. Exiting.")

    print(f"Selected post ID: {post_id} with caption prompt: {post['caption_prompt']}")

    # Add the selected post's ID to used_ids and save the state
    state["used_ids"].append(post_id)
    _write_state(state)
    print(f"Updated state.json with new used_id: {post_id}")

    # Generate caption
    try:
        caption = generate_caption(post["caption_prompt"], book_context)
        print("Generated caption:\n", caption)
        with open(CAPTION_FILE, "w", encoding="utf-8") as f:
            f.write(caption)
    except Exception as e:
        print("Caption generation failed:", e)
        raise

    # Generate image
    try:
        image_path = generate_image(post["image_prompt"])
        print("Image saved to:", image_path)
    except Exception as e:
        print("Image generation failed:", e)
        raise

    print("Done.")


if __name__ == "__main__":
    main()