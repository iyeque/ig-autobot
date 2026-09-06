#!/usr/bin/env python3
"""
Instagram quote poster: deterministic single-image quote posts.
- Picks next unposted quote from posts.json (pillar == "quote")
- Generates dark minimalist image with quote overlay (no AI Horde)
- Posts to Instagram
- Tracks posted quote IDs in quotes_state.json
"""
import os
import sys
import json
import time
import re
import requests
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bot import add_static_text_overlay

QUOTES_STATE = Path("quotes_state.json")
POSTS_FILE = Path("posts.json")
IMAGE_OUT = Path("images/quote_post.jpg")


def load_quotes_state():
    if QUOTES_STATE.exists():
        return json.loads(QUOTES_STATE.read_text(encoding="utf-8"))
    return {"posted_ids": [], "last_posted_id": None, "schedule": []}


def save_quotes_state(state):
    QUOTES_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_next_quote():
    posts = json.loads(POSTS_FILE.read_text(encoding="utf-8"))
    quotes = [p for p in posts if p.get("pillar") == "quote"]
    state = load_quotes_state()
    posted = set(state.get("posted_ids", []))
    for q in quotes:
        if q["id"] not in posted:
            return q, state
    # All quotes posted — reset
    state["posted_ids"] = []
    save_quotes_state(state)
    return quotes[0] if quotes else None, state


def generate_quote_image(quote_text: str, title: str) -> str:
    """Generate dark minimalist quote image with text overlay."""
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import textwrap

    w, h = 1080, 1350
    bg = (8, 8, 8)
    center = (28, 30, 36)
    inset = (60, 64, 72)
    text_color = (240, 240, 240)
    footer_color = (160, 165, 175)

    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    # Central panel
    margin_x, margin_y = 70, 170
    panel = (margin_x, margin_y, w - margin_x, h - margin_y)
    draw.rectangle(panel, fill=center)
    draw.rectangle((panel[0]+18, panel[1]+18, panel[2]-18, panel[3]-18), outline=inset, width=2)

    # Soft orb
    orb_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(orb_layer)
    orb_size = min(w, h) // 3
    od.ellipse(
        ((w - orb_size) / 2 - orb_size * 0.15, (h - orb_size) / 2 - orb_size * 0.05,
         (w + orb_size) / 2 + orb_size * 0.15, (h + orb_size) / 2 + orb_size * 0.15),
        fill=(50, 58, 80, 120),
    )
    orb_layer = orb_layer.filter(ImageFilter.GaussianBlur(radius=70))
    img = Image.alpha_composite(img.convert("RGBA"), orb_layer).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Fonts
    def _load_font(size, bold=False):
        paths = [
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for p in paths:
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    # Wrap and draw quote
    max_text_w = panel[2] - panel[0] - 18 * 2 - 40
    header_size = 62
    font = _load_font(header_size)
    raw = f'"{quote_text}"'
    wrap_width = max(16, int(max_text_w / (header_size * 0.48)))
    wrapped = textwrap.wrap(raw, width=wrap_width)
    line_hs = [
        draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
        for line in wrapped
    ]
    line_h = max(line_hs) if line_hs else 40
    line_spacing = 26
    th = line_h * len(wrapped) + line_spacing * max(0, len(wrapped) - 1)
    footer_space = 110
    available = (panel[3] - footer_space) - (panel[1] + 40)
    block_y = panel[1] + 40 + (available - th) / 2
    block_x = panel[0] + 40

    for i, line in enumerate(wrapped):
        lw = draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
        draw.text((block_x + (max_text_w - lw) / 2, block_y + i * (line_h + line_spacing)), line, font=font, fill=text_color)

    # Footer
    footer = "M.W.E. WIGMAN | THE NINE STITCHES"
    font_footer = _load_font(28)
    fbbox = draw.textbbox((0, 0), footer, font=font_footer)
    fw, fh = fbbox[2] - fbbox[0], fbbox[3] - fbbox[1]
    fx = panel[0] + (panel[2] - panel[0] - fw) / 2
    fy = panel[3] - 18 - 20 - fh
    draw.text((fx, fy), footer, font=font_footer, fill=footer_color)

    IMAGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(IMAGE_OUT), format="JPEG", quality=95, optimize=True)
    return str(IMAGE_OUT)


def generate_quote_caption(quote_text: str, title: str) -> str:
    short = quote_text.strip().rstrip(".")
    if len(short) > 120:
        short = short[:117] + "..."
    return f'"{short}"\n\n— The Nine Stitches'


def post_to_instagram(image_path: str, caption: str):
    user_id = os.environ.get("IG_USER_ID")
    token = os.environ.get("IG_ACCESS_TOKEN")
    if not user_id or not token:
        print("❌ Missing IG_USER_ID or IG_ACCESS_TOKEN")
        sys.exit(1)

    with open(image_path, "rb") as f:
        res = requests.post(
            f"https://graph.facebook.com/v18.0/{user_id}/media",
            data={"caption": caption, "access_token": token},
            files={"source": (os.path.basename(image_path), f, "image/jpeg")},
        ).json()

    creation_id = res.get("id")
    if not creation_id:
        print(f"❌ Upload failed: {res}")
        sys.exit(1)

    for _ in range(10):
        status = requests.get(
            f"https://graph.facebook.com/v18.0/{creation_id}",
            params={"fields": "status_code", "access_token": token},
        ).json()
        if status.get("status_code") == "FINISHED":
            break
        time.sleep(5)
    else:
        print("❌ Media processing timeout")
        sys.exit(1)

    pub = requests.post(
        f"https://graph.facebook.com/v18.0/{user_id}/media_publish",
        data={"creation_id": creation_id, "access_token": token},
    ).json()
    if "id" in pub:
        print(f"✓ Instagram quote post published: {pub['id']}")
        return True
    print(f"❌ Publish failed: {pub}")
    return False


def main():
    quote, state = get_next_quote()
    if not quote:
        print("⏭️ No quotes available.")
        sys.exit(0)

    print(f"Posting quote ID {quote['id']}: {quote['title']}")
    # Extract actual quote text from caption_prompt if possible
    text = quote.get("caption_prompt", "")
    quote_text = text
    # Match standalone quoted text (not possessives like "book's")
    m = re.search(r"(?<![a-zA-Z])'([^']{10,200})'", text)
    if m:
        quote_text = m.group(1)
    else:
        quote_text = quote.get("title", text[:120])

    image_path = generate_quote_image(quote_text, quote.get("title", ""))
    caption = generate_quote_caption(quote_text, quote.get("title", ""))

    if post_to_instagram(image_path, caption):
        state["posted_ids"].append(quote["id"])
        state["last_posted_id"] = quote["id"]
        state.setdefault("schedule", []).append({
            "id": quote["id"],
            "title": quote["title"],
            "posted_at": datetime.utcnow().isoformat() + "Z",
        })
        save_quotes_state(state)
        print(f"✓ Recorded quote {quote['id']} as posted.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
