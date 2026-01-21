import os
import json
import random
import requests
from pathlib import Path

POSTS_FILE = Path("posts.json")
STATE_FILE = Path("state.json")  # to track which posts have been used

GROQ_API_KEY = os.getenv("GROQ_API_KEY")      # or other LLM provider
HF_TOKEN = os.getenv("HF_TOKEN")              # for image generation
IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN")
IG_USER_ID = os.getenv("IG_USER_ID")


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
        # reset if all used
        state["used_ids"] = []
        save_state(state)
        remaining = posts
    post = random.choice(remaining)
    state["used_ids"].append(post["id"])
    save_state(state)
    return post


def generate_caption(caption_prompt: str) -> str:
    # Example using a chat completion style API (Groq / OpenAI-compatible)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama3-8b",
        "messages": [
            {"role": "system", "content": "You are M.W.E. Wigman, writing reflective, philosophical Instagram captions based on your books 'The Nine Stitches' and 'A Burden of One's Choice'. Your tone is calm, precise, and layered, blending nature, systems thinking, and human psychology."},
            {"role": "user", "content": caption_prompt}
        ],
        "temperature": 0.7
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def generate_image(image_prompt: str, output_path: str = "image.jpg") -> str:
    # Example using HuggingFace Inference API for Stable Diffusion
    url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": image_prompt}
    r = requests.post(url, headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(r.content)
    return output_path


def upload_image_to_host(image_path: str) -> str:
    """
    Simplest approach: commit image to repo and use raw GitHub URL,
    or use an external image host/API. For now, this is a placeholder.
    """
    # TODO: implement actual upload logic or pre-hosting strategy.
    raise NotImplementedError("Implement image hosting and return a public URL.")


def post_to_instagram(image_url: str, caption: str):
    # Step 1: create media object
    create_url = f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media"
    create_data = {
        "image_url": image_url,
        "caption": caption,
        "access_token": IG_ACCESS_TOKEN
    }
    create_resp = requests.post(create_url, data=create_data, timeout=60)
    create_resp.raise_for_status()
    creation_id = create_resp.json()["id"]

    # Step 2: publish media object
    publish_url = f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media_publish"
    publish_data = {
        "creation_id": creation_id,
        "access_token": IG_ACCESS_TOKEN
    }
    publish_resp = requests.post(publish_url, data=publish_data, timeout=60)
    publish_resp.raise_for_status()
    return publish_resp.json()


def main():
    posts = load_posts()
    state = load_state()
    post = pick_next_post(posts, state)

    print(f"Selected post: {post['id']} - {post['title']}")

    caption = generate_caption(post["caption_prompt"])
    print("Generated caption:\n", caption)

    image_path = generate_image(post["image_prompt"])
    print(f"Generated image at: {image_path}")

    # You must implement this to return a public URL accessible by Instagram
    image_url = upload_image_to_host(image_path)

    resp = post_to_instagram(image_url, caption)
    print("Instagram response:", resp)


if __name__ == "__main__":
    main()