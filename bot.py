import os
import json
import random
import time
import base64
import requests
from pathlib import Path
from typing import Tuple, Any
from huggingface_hub import InferenceClient
from PIL import Image
import io

POSTS_FILE = Path("posts.json")
STATE_FILE = Path("state.json")
WORKING_MODEL_FILE = Path("working_model.txt")

# Default model (can be overridden with HF_MODEL env var)
DEFAULT_HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

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


def _post_to_hf(model: str, payload: dict, timeout: int = 120, use_router: bool = True) -> Tuple[int, Any, str]:
    """
    Unified POST helper.
    - If use_router is True, POST to router v1 chat/completions (OpenAI-compatible).
    - Otherwise POST to api-inference.huggingface.co/models/{model}.
    Returns: (status_code:int, body:Union[dict,str,bytes], endpoint:str)
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    # Router chat endpoint (OpenAI-compatible)
    if use_router:
        url = "https://router.huggingface.co/v1/chat/completions"
        body = dict(payload)  # copy
        if "model" not in body:
            body["model"] = model
        headers["Content-Type"] = "application/json"
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
            try:
                return r.status_code, r.json(), "router"
            except Exception:
                return r.status_code, r.text, "router"
        except requests.RequestException as e:
            return 0, str(e), "router"

    # Fallback to classic api-inference
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers["Content-Type"] = "application/json"
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        try:
            return r.status_code, r.json(), "api-inference"
        except Exception:
            return r.status_code, r.content, "api-inference"
    except requests.RequestException as e:
        return 0, str(e), "api-inference"


def generate_caption(caption_prompt: str) -> str:
    """
    Resilient caption generator using Hugging Face router + api-inference.
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

    # Prepare both chat-style and classic payloads
    chat_payload = {
        "messages": [
            {"role": "system", "content": "You are a concise Instagram caption writer."},
            {"role": "user", "content": caption_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 180
    }
    # Classic HF payload (kept for api-inference)
    hf_payload = {
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
            attempt += 1
            # First try router (chat) — include model in payload
            status, body, endpoint = _post_to_hf(model, {**chat_payload, "model": model}, timeout=120, use_router=True)
            print(f"Attempt {attempt} endpoint={endpoint} status={status}")

            # Router success (200) — parse chat-style response
            if status == 200 and endpoint == "router":
                try:
                    choices = body.get("choices") if isinstance(body, dict) else None
                    if choices and len(choices) > 0:
                        msg = choices[0].get("message") or {}
                        text = msg.get("content") or choices[0].get("text")
                        if text:
                            text = text.strip()
                            _persist_working_model(model)
                            print(f"Model {model} succeeded via router and persisted.")
                            return text
                except Exception as e:
                    last_error = e
                    print("Failed to parse router response:", e)
                    # fall through to try api-inference for this model

            # Router returned 404 -> try api-inference for this model
            if status == 404 or (status == 200 and endpoint == "router" and not isinstance(body, dict)):
                status2, body2, endpoint2 = _post_to_hf(model, hf_payload, timeout=120, use_router=False)
                print(f"Fallback endpoint={endpoint2} status={status2}")
                if status2 == 200:
                    try:
                        data = body2
                        # parse classic HF formats
                        if isinstance(data, list) and data and isinstance(data[0], dict):
                            for k in ("generated_text", "text"):
                                if k in data[0]:
                                    text = data[0][k].strip()
                                    _persist_working_model(model)
                                    print(f"Model {model} succeeded via api-inference and persisted.")
                                    return text
                        if isinstance(data, dict):
                            if "generated_text" in data:
                                text = data["generated_text"].strip()
                                _persist_working_model(model)
                                return text
                            if "choices" in data and data["choices"]:
                                choice = data["choices"][0]
                                text = choice.get("text") or (choice.get("message") or {}).get("content")
                                if text:
                                    text = text.strip()
                                    _persist_working_model(model)
                                    return text
                    except Exception as e:
                        last_error = e
                        print("Failed to parse api-inference response:", e)
                # if api-inference failed, break to try next model
                break

            # Transient errors: retry with backoff
            if status in (429,) or status >= 500 or status == 0:
                backoff = 2 ** attempt
                print(f"Transient error (status {status}). Retrying in {backoff}s (attempt {attempt})")
                time.sleep(backoff)
                continue

            # Other client errors: stop retrying this model
            last_error = RuntimeError(f"Model {model} returned status {status} (endpoint={endpoint})")
            print(last_error)
            break

    raise RuntimeError(f"All HuggingFace models failed or are unavailable. Last error: {last_error}")


def generate_image(image_prompt: str, output_path: str = "output.jpg") -> str:
    """
    Generate an image using Hugging Face InferenceClient routed to a provider.
    Falls back to api-inference if provider call fails. Returns saved image path.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set in the environment")

    sd_model = os.getenv("SD_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

    # Try provider via InferenceClient first
    try:
        print(f"Trying provider-backed InferenceClient for model: {sd_model}")
        # If the provider requires a separate key, replace api_key with the provider key from env
        client = InferenceClient(provider="replicate", api_key=os.environ["HF_TOKEN"])
        image = client.text_to_image(image_prompt, model=sd_model)
        # PIL.Image returned
        if isinstance(image, Image.Image):
            image.save(output_path, format="JPEG", quality=95)
            print(f"Saved image from provider to {output_path}")
            return output_path
        # Bytes returned
        if isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image)).convert("RGB")
            img.save(output_path, format="JPEG", quality=95)
            print(f"Saved image bytes from provider to {output_path}")
            return output_path
        print("Provider returned unexpected type; falling back to api-inference.")
    except Exception as e:
        print("Provider InferenceClient call failed:", e)

    # Fallback: try api-inference endpoints
    print(f"Falling back to api-inference for model: {sd_model}")
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": image_prompt}
    url = f"https://api-inference.huggingface.co/models/{sd_model}"
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=300)
    except requests.RequestException as e:
        raise RuntimeError(f"Network error calling api-inference for {sd_model}: {e}")

    if r.status_code in (404, 410):
        raise RuntimeError(f"Image model {sd_model} not available via api-inference (status {r.status_code}).")
    if r.status_code >= 400:
        raise RuntimeError(f"Image model {sd_model} returned status {r.status_code}: {r.text[:400]}")

    content_type = r.headers.get("Content-Type", "")
    if "application/json" in content_type:
        data = r.json()
        if isinstance(data, dict) and "images" in data and data["images"]:
            b64 = data["images"][0]
            image_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.save(output_path, format="JPEG", quality=95)
            return output_path
        if isinstance(data, dict) and "artifacts" in data and data["artifacts"]:
            b64 = data["artifacts"][0].get("base64")
            if b64:
                image_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img.save(output_path, format="JPEG", quality=95)
                return output_path
        with open(output_path, "wb") as f:
            f.write(r.content)
        return output_path
    else:
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