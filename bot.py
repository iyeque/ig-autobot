import os
import json
import random
import time
import base64
import requests
from pathlib import Path

POSTS_FILE = Path("posts.json")
STATE_FILE = Path("state.json")
WORKING_MODEL_FILE = Path("working_model.txt")

# Inline default model (no secret required). Change this string to try a different model.
DEFAULT_HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Environment token (must be set in GitHub Actions or locally)
HF_TOKEN = os.getenv("HF_TOKEN")

# Fallback models to try if the primary model is unavailable
FALLBACK_MODELS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/flan-t5-large",
    "bigscience/bloomz-7b1",
    "tiiuae/falcon-7b-instruct"
]


def load_posts():
    with open(POSTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state():
    if not STATE_FILE.exists():
        return {"used_ids": []}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def pick_next_post(posts, state):
    used_ids = set(state.get("used_ids", []))
    remaining = [p for p in posts if p["id"] not in used_ids]
    if not remaining:
        state["used_ids"] = []
        save_state(state)
        remaining = posts
    post = random.choice(remaining)
    state["used_ids"].append(post["id"])
    save_state(state)
    return post


def _read_persisted_model():
    if WORKING_MODEL_FILE.exists():
        try:
            txt = WORKING_MODEL_FILE.read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return None


def _persist_working_model(model_name: str):
    try:
        WORKING_MODEL_FILE.write_text(model_name, encoding="utf-8")
    except Exception:
        # Non-fatal: persistence is best-effort
        pass


def _post_to_hf(model, payload, timeout=120):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    return requests.post(url, headers=headers, json=payload, timeout=timeout)


def generate_caption(caption_prompt: str) -> str:
    """
    Resilient caption generator using Hugging Face Inference API.
    - Uses a persisted working model if available.
    - Falls back to DEFAULT_HF_MODEL and FALLBACK_MODELS.
    - Retries transient errors with exponential backoff.
    - Persists the first model that succeeds for future runs.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set in the environment")

    # Determine model order: persisted -> inline default -> fallbacks
    models_to_try = []
    persisted = _read_persisted_model()
    if persisted:
        models_to_try.append(persisted)
    if DEFAULT_HF_MODEL not in models_to_try:
        models_to_try.append(DEFAULT_HF_MODEL)
    for m in FALLBACK_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    payload = {
        "inputs": (
            "Write a reflective, philosophical Instagram caption in the voice of "
            "M.W.E. Wigman. Blend nature, systems thinking, and introspection.\n\n"
            f"Caption prompt: {caption_prompt}"
        ),
        "parameters": {
            "max_new_tokens": 180,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    last_error = None
    for model in models_to_try:
        if not model:
            continue
        print(f"Trying model: {model}")
        attempt = 0
        while attempt < 3:
            try:
                r = _post_to_hf(model, payload, timeout=120)
            except requests.RequestException as e:
                last_error = e
                attempt += 1
                backoff = 2 ** attempt
                print(f"Network error for model {model}: {e}. Retrying in {backoff}s (attempt {attempt})")
                time.sleep(backoff)
                continue

            status = r.status_code
            print(f"Model {model} returned status {status}")

            # Model removed or not found: try next model
            if status in (410, 404):
                print(f"Model {model} not available (status {status}). Trying next model.")
                break

            # Rate limit or server error: retry
            if status in (429,) or status >= 500:
                attempt += 1
                backoff = 2 ** attempt
                print(f"Transient error (status {status}) for model {model}. Retrying in {backoff}s (attempt {attempt})")
                time.sleep(backoff)
                continue

            # Success
            if status == 200:
                try:
                    data = r.json()
                except Exception as e:
                    last_error = e
                    print(f"Failed to parse JSON from model {model}: {e}")
                    break

                # common HF formats:
                # 1) list of dicts with "generated_text"
                if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                    text = data[0]["generated_text"].strip()
                    _persist_working_model(model)
                    print(f"Model {model} succeeded and persisted as working model.")
                    return text

                # 2) dict with "generated_text"
                if isinstance(data, dict) and "generated_text" in data:
                    text = data["generated_text"].strip()
                    _persist_working_model(model)
                    print(f"Model {model} succeeded and persisted as working model.")
                    return text

                # 3) OpenAI-like structure: {"choices":[{"text": "..."}]} or {"choices":[{"message":{"content":"..."}}]}
                if isinstance(data, dict) and "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    text = choice.get("text") or (choice.get("message") or {}).get("content")
                    if text:
                        text = text.strip()
                        _persist_working_model(model)
                        print(f"Model {model} succeeded and persisted as working model.")
                        return text

                # 4) Some endpoints return {"generated_text": "..."} nested differently or list with "text"
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    for k in ("generated_text", "text"):
                        if k in data[0]:
                            text = data[0][k].strip()
                            _persist_working_model(model)
                            print(f"Model {model} succeeded and persisted as working model.")
                            return text

                # Unexpected format
                last_error = ValueError(f"Unexpected HuggingFace response format from model {model}: {data}")
                print(last_error)
                break

            # Other client errors: stop retrying this model
            last_error = RuntimeError(f"Model {model} returned status {status}")
            break

    # If we reach here, no model worked
    raise RuntimeError(f"All HuggingFace models failed or are unavailable. Last error: {last_error}")


def generate_image(image_prompt: str, output_path: str = "output.jpg") -> str:
    """
    Generate an image using Hugging Face Inference API for Stable Diffusion XL.
    Handles both binary responses and JSON with base64 images.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set in the environment")

    sd_model = "stabilityai/stable-diffusion-xl-base-1.0"
    url = f"https://api-inference.huggingface.co/models/{sd_model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": image_prompt}

    r = requests.post(url, headers=headers, json=payload, timeout=300)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "")
    # If HF returns JSON with base64 images
    if "application/json" in content_type:
        data = r.json()
        # Common HF image response: {"images": ["<base64>"]}
        if isinstance(data, dict) and "images" in data and data["images"]:
            b64 = data["images"][0]
            image_bytes = base64.b64decode(b64)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            return output_path
        # Some endpoints return {"artifacts": [{"base64": "..."}]}
        if isinstance(data, dict) and "artifacts" in data and data["artifacts"]:
            b64 = data["artifacts"][0].get("base64")
            if b64:
                image_bytes = base64.b64decode(b64)
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                return output_path
        # Unexpected JSON format: write raw content fallback
        with open(output_path, "wb") as f:
            f.write(r.content)
        return output_path

    # Otherwise assume binary image content
    with open(output_path, "wb") as f:
        f.write(r.content)
    return output_path


def main():
    posts = load_posts()
    state = load_state()
    post = pick_next_post(posts, state)

    print(f"Selected post: {post['id']} - {post['title']}")

    caption = generate_caption(post["caption_prompt"])
    print("Generated caption:\n", caption)

    image_path = generate_image(post["image_prompt"])
    print(f"Generated image at: {image_path}")

    # Save caption for GitHub Actions to read
    with open("caption.txt", "w", encoding="utf-8") as f:
        f.write(caption)


if __name__ == "__main__":
    main()