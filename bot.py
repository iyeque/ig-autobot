#!/usr/bin/env python3
"""
Robust IG autobot using Hugging Face Router (router.huggingface.co).
- Router-first for captions (chat -> text-generation fallback).
- Router-first for images (InferenceClient.text_to_image with router-compatible models).
- No provider=... usage and no Replicate keys required.
"""

import os
import sys
import time
import json
import requests
from typing import Any, Dict, Optional, List

# Environment / config
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = os.environ.get("HF_MODEL", "") or None
SD_MODEL = os.environ.get("SD_MODEL", "") or None

# Defaults and fallbacks
DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FALLBACK_HF_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]

# Image model candidates (router-compatible, commonly accessible)
IMAGE_MODEL_CANDIDATES = [
    # If SD_MODEL is set it will be tried first by generate_image()
    "stabilityai/stable-diffusion-xl-base-1.0",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    # community fallback (replace with a model you have accepted on HF if needed)
    "stabilityai/stable-diffusion-3-medium-diffusers",
]

WORKING_MODEL_FILE = "working_model.txt"
CAPTION_FILE = "caption.txt"
OUTPUT_IMAGE = "output.jpg"


# -------------------------
# Persistence helpers
# -------------------------
def _read_persisted_model() -> Optional[str]:
    try:
        if os.path.exists(WORKING_MODEL_FILE):
            with open(WORKING_MODEL_FILE, "r", encoding="utf-8") as f:
                m = f.read().strip()
                return m or None
    except Exception:
        pass
    return None


def _persist_working_model(model: str) -> None:
    try:
        with open(WORKING_MODEL_FILE, "w", encoding="utf-8") as f:
            f.write(model)
    except Exception:
        pass


# -------------------------
# Router HTTP helpers
# -------------------------
def _router_post_json(url: str, token: str, body: dict, timeout: int = 120) -> Any:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    # Try to return JSON, else raise with text
    try:
        return r.json()
    except Exception:
        r.raise_for_status()
        return r.text


def _router_chat_call(client: Any, model: str, messages: list, temperature: float, max_tokens: int, token: str, timeout: int = 120):
    """
    Try dynamic client.chat.completions.create via getattr; if that fails, call the Router HTTP endpoint.
    Returns parsed response (dict/object) or raises Exception.
    """
    try:
        client_any = client  # dynamic access
        chat = getattr(client_any, "chat")
        completions = getattr(chat, "completions")
        create = getattr(completions, "create")
        return create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    except Exception:
        url = "https://router.huggingface.co/v1/chat/completions"
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        return _router_post_json(url, token, body, timeout=timeout)


def _text_generation_router_call(client: Any, model: str, inputs: str, params: dict, token: str, timeout: int = 120):
    """
    Try dynamic client.text_generation.create via getattr; if that fails, call the Router HTTP endpoint.
    Returns parsed response (dict/object) or raises Exception.
    """
    try:
        client_any = client  # dynamic access
        text_gen = getattr(client_any, "text_generation")
        create = getattr(text_gen, "create")
        return create(model=model,
                      inputs=inputs,
                      max_new_tokens=params.get("max_new_tokens"),
                      temperature=params.get("temperature"))
    except Exception:
        url = "https://router.huggingface.co/v1/text-generation"
        body = {
            "model": model,
            "inputs": inputs,
            "parameters": {
                "max_new_tokens": params.get("max_new_tokens"),
                "temperature": params.get("temperature")
            }
        }
        return _router_post_json(url, token, body, timeout=timeout)


# -------------------------
# Caption generation
# -------------------------
def generate_caption(caption_prompt: str) -> str:
    """
    Robust caption generation:
    - Try persisted model, DEFAULT_HF_MODEL, then FALLBACK_HF_MODELS.
    - For each model: try router chat completions first; if that fails with 400,
      fall back to router text-generation style call.
    - Persist the first working model.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set in the environment")

    from huggingface_hub import InferenceClient

    client = InferenceClient(token=HF_TOKEN)

    models_to_try: List[str] = []
    persisted = _read_persisted_model()
    if persisted:
        models_to_try.append(persisted)
    if DEFAULT_HF_MODEL and DEFAULT_HF_MODEL not in models_to_try:
        models_to_try.append(DEFAULT_HF_MODEL)
    for m in FALLBACK_HF_MODELS:
        if m and m not in models_to_try:
            models_to_try.append(m)

    # Chat payload (router chat completions)
    chat_payload = {
        "messages": [
            {"role": "system", "content": "You are a concise Instagram caption writer."},
            {"role": "user", "content": caption_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 180
    }

    # Text-generation fallback payload
    text_payload = {
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

        # 1) Try router chat completions via dynamic client or HTTP fallback
        try:
            print("  -> Attempting router chat completion")
            resp = _router_chat_call(client, model, chat_payload["messages"],
                                     chat_payload["temperature"], chat_payload["max_tokens"], HF_TOKEN)
            # Parse router chat response (object or dict)
            if hasattr(resp, "choices") and getattr(resp, "choices"):
                choice = resp.choices[0]
                msg = getattr(choice, "message", None) or {}
                text = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)) or getattr(choice, "text", None)
                if text:
                    text = text.strip()
                    _persist_working_model(model)
                    print(f"Model {model} succeeded via router chat and persisted.")
                    return text
            if isinstance(resp, dict):
                choices = resp.get("choices")
                if choices and len(choices) > 0:
                    msg = choices[0].get("message") or {}
                    text = msg.get("content") or choices[0].get("text")
                    if text:
                        text = text.strip()
                        _persist_working_model(model)
                        print(f"Model {model} succeeded via router chat and persisted.")
                        return text
        except Exception as e_chat:
            last_error = e_chat
            err_str = str(e_chat)
            print(f"  -> Router chat attempt failed for {model}: {err_str}")
            if "402" in err_str or "Payment Required" in err_str:
                raise RuntimeError(f"Model {model} returned 402 (access/credits). Last error: {err_str}")

        # 2) Fallback: try router text-generation style via dynamic client or HTTP fallback
        try:
            print("  -> Attempting router text-generation fallback")
            gen = _text_generation_router_call(client, model, text_payload["inputs"], text_payload["parameters"], HF_TOKEN)
            # Parse result (object or dict)
            if hasattr(gen, "generated_text"):
                text = gen.generated_text.strip()
                _persist_working_model(model)
                print(f"Model {model} succeeded via text-generation and persisted.")
                return text
            if isinstance(gen, dict):
                if "generated_text" in gen:
                    text = gen["generated_text"].strip()
                    _persist_working_model(model)
                    return text
                if "choices" in gen and gen["choices"]:
                    c = gen["choices"][0]
                    text = c.get("text") or (c.get("message") or {}).get("content")
                    if text:
                        text = text.strip()
                        _persist_working_model(model)
                        return text
            if isinstance(gen, str) and gen.strip():
                text = gen.strip()
                _persist_working_model(model)
                print(f"Model {model} returned text via text-generation and persisted.")
                return text
            if isinstance(gen, Exception):
                last_error = gen
                print(f"  -> text-generation call raised exception for {model}: {gen}")
                if "402" in str(gen) or "Payment Required" in str(gen):
                    raise RuntimeError(f"Model {model} returned 402 (access/credits). Last error: {gen}")
        except Exception as e:
            last_error = e
            print(f"  -> Fallback text-generation error for {model}: {e}")

    raise RuntimeError(f"All HuggingFace models failed or are unavailable. Last error: {last_error}")


# -------------------------
# Image generation
# -------------------------
def generate_image(prompt: str, preferred: Optional[str] = None) -> str:
    """
    Generate an image using InferenceClient.text_to_image via the Router.
    Tries preferred model, SD_MODEL env, then IMAGE_MODEL_CANDIDATES.
    Returns path to saved image (OUTPUT_IMAGE) or raises.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set in the environment")

    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)

    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    if SD_MODEL and SD_MODEL not in candidates:
        candidates.append(SD_MODEL)
    # append configured candidates while preserving order and dedup
    for m in IMAGE_MODEL_CANDIDATES:
        if m not in candidates:
            candidates.append(m)

    last_error = None
    for model in candidates:
        if not model:
            continue
        print(f"Attempting image model: '{model}'")
        try:
            # Use the high-level client.text_to_image if available (dynamic access)
            try:
                text_to_image = getattr(client, "text_to_image")
                img = text_to_image(prompt, model=model)
                # The client may return PIL.Image or bytes; handle both
                if hasattr(img, "save"):
                    img.save(OUTPUT_IMAGE)
                else:
                    # assume bytes
                    with open(OUTPUT_IMAGE, "wb") as f:
                        if isinstance(img, bytes):
                            f.write(img)
                        else:
                            # try to convert to bytes
                            f.write(bytes(img))
                print(f"Image generated with model {model} and saved to {OUTPUT_IMAGE}")
                _persist_working_model(model)
                return OUTPUT_IMAGE
            except Exception as e_client_img:
                # Fallback: call router text-to-image endpoint directly
                last_error = e_client_img
                err_str = str(e_client_img)
                print(f"  -> client.text_to_image failed for {model}: {err_str}")
                if "402" in err_str or "Payment Required" in err_str:
                    # credits/access issue; stop trying further models
                    raise RuntimeError(f"Model {model} returned 402 (access/credits). Last error: {err_str}")

                # Router text-to-image endpoint (best-effort)
                url = "https://router.huggingface.co/v1/text-to-image"
                body = {"model": model, "prompt": prompt}
                try:
                    resp = requests.post(url, headers={"Authorization": f"Bearer {HF_TOKEN}"}, json=body, timeout=300)
                    resp.raise_for_status()
                    # Many router image endpoints return bytes or base64; try to handle common cases
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        j = resp.json()
                        # try common keys
                        if isinstance(j, dict) and "images" in j and j["images"]:
                            # images may be base64 strings
                            import base64
                            b64 = j["images"][0]
                            data = base64.b64decode(b64)
                            with open(OUTPUT_IMAGE, "wb") as f:
                                f.write(data)
                            _persist_working_model(model)
                            return OUTPUT_IMAGE
                        # else fall through to error
                    else:
                        # assume raw image bytes
                        with open(OUTPUT_IMAGE, "wb") as f:
                            f.write(resp.content)
                        _persist_working_model(model)
                        return OUTPUT_IMAGE
                except Exception as e_http:
                    last_error = e_http
                    print(f"  -> Router HTTP image attempt failed for {model}: {e_http}")
                    # If 402 encountered here, stop trying further models
                    if "402" in str(e_http) or "Payment Required" in str(e_http):
                        raise RuntimeError(f"Model {model} returned 402 (access/credits). Last error: {e_http}")
                    # otherwise continue to next model
        except Exception as e:
            last_error = e
            print(f"Model {model} failed: {e}")

    raise RuntimeError(f"All image models failed. Last error: {last_error}")


# -------------------------
# Main flow (example)
# -------------------------
def main():
    # Example posts list; replace with your real post source
    posts = [
        {
            "caption_prompt": "A short reflective caption about bioluminescent defense in deep sea creatures.",
            "image_prompt": "A glowing deep-sea creature using bioluminescence to ward off predators, cinematic, photorealistic"
        },
        {
            "caption_prompt": "Failure as architect — a short motivational caption about learning from setbacks.",
            "image_prompt": "An abstract blueprint made of shattered glass pieces forming a phoenix, dramatic lighting"
        }
    ]

    # pick a post (rotate, randomize, or pick by schedule)
    post = posts[0]

    print("Selected post:", post["caption_prompt"])

    # Generate caption
    try:
        caption = generate_caption(post["caption_prompt"])
        print("Generated caption:\n", caption)
        with open(CAPTION_FILE, "w", encoding="utf-8") as f:
            f.write(caption)
    except Exception as e:
        print("Caption generation failed:", e)
        raise

    # Generate image
    try:
        image_path = generate_image(post["image_prompt"], preferred=HF_MODEL)
        print("Image saved to:", image_path)
    except Exception as e:
        print("Image generation failed:", e)
        raise

    print("Done.")


if __name__ == "__main__":
    main()