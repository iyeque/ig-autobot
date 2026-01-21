import os
import json
import random
import requests
from pathlib import Path

POSTS_FILE = Path("posts.json")
STATE_FILE = Path("state.json")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

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

def generate_caption(caption_prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama3-8b",
        "messages": [
            {"role": "system", "content": "You are M.W.E. Wigman, writing reflective, philosophical Instagram captions."},
            {"role": "user", "content": caption_prompt}
        ],
        "temperature": 0.7
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def generate_image(image_prompt: str, output_path: str = "output.jpg") -> str:
    url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": image_prompt}
    r = requests.post(url, headers=headers, json=payload, timeout=300)
    r.raise_for_status()
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