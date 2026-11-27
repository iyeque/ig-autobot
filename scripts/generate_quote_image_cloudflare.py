#!/usr/bin/env python3
"""Cloudflare-backed quote imagery for the main brand (The Nine Stitches).

Primary background source for Instagram quote posts: one AI background per
quote via Cloudflare Workers AI (flux-1-schnell), with the *exact same* quote
text and book footer as publish_instagram_quotes.generate_quote_image() drawn
on top at strong contrast.  The quote text comes from the book, never from an
AI model.

This module returns None whenever Cloudflare is unconfigured or any request
fails, so the existing PIL renderer (generate_quote_image) runs unchanged as
the fallback.  Secrets: CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN.
"""
import os
import io
import re
import hashlib
import base64
import textwrap
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter


CLOUDFLARE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
# Dark editorial backgrounds in the same family as the main-brand carousels.
# The quote text is never part of the prompt -- it is overlaid locally so the
# book wording stays byte-for-byte identical to the PIL renderer.
BG_PROMPTS = [
    "moody editorial nature photograph, frost on bare branches at blue hour, deep slate-teal and indigo palette, soft cinematic bokeh, dreamlike stillness, minimal, generous dark negative space in the center, no text, no words, no letters",
    "quiet winter landscape at dusk, gentle snowfall, desaturated blue-grey tones, soft diffused light, contemplative minimalism, cinematic, dark open center, no text, no words, no letters",
    "abstract soft light and shadow on dark textured paper, muted cool slate tones, subtle film grain, calm atmospheric negative space in the center, no text, no words, no letters",
]

# Palette -- cream quote on a deeply darkened photograph for strong contrast.
CREAM = (240, 240, 240)
FOOTER = (178, 182, 192)
ACCENT = (214, 160, 110)
VIGNETTE = (6, 8, 13)

DEFAULT_OUT = Path("images/quote_post.jpg")
FOOTER_TEXT = "M.W.E. WIGMAN | THE NINE STITCHES"


def _load_font(size: int, serif: bool = True):
    """Serif-first font stack matching the main-brand editorial carousels."""
    if serif:
        cands = ["C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/georgiab.ttf",
                 "C:/Windows/Fonts/times.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "DejaVuSerif.ttf"]
    else:
        cands = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]
    for p in cands:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _cf_background(prompt: str, account_id: str, token: str) -> Image.Image:
    url = (f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
           f"/ai/run/{CLOUDFLARE_MODEL}")
    r = requests.post(url, headers={"Authorization": f"Bearer {token}"},
                      json={"prompt": prompt}, timeout=120)
    r.raise_for_status()
    if "image" in r.headers.get("content-type", ""):
        raw = r.content
    else:
        raw = base64.b64decode(r.json().get("result", {}).get("image", ""))
    if not raw:
        raise RuntimeError("Cloudflare returned no image payload")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _clean_quote_text(quote_text: str) -> str:
    """Reproduce the exact text cleaning of publish_instagram_quotes.generate_quote_image.

    Kept in one place so the AI-backed image and the PIL fallback always show
    identical wording.  Strips chapter/author/book attributions (the footer
    already covers them) and wraps the bare quote in quotation marks.
    """
    raw_text = (quote_text or "").strip()
    raw_text = re.sub(
        r"\s*[—–-]\s*from\s+(Chapter\s+\d+|Introduction|Prologue|Preface|Epilogue|Appendix).*$",
        "", raw_text, flags=re.IGNORECASE | re.DOTALL,
    )
    raw_text = re.sub(r"\s+by\s+M\.W\.E\.\s*Wigman\.?\s*$", "", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"\s+in\s+The\s+Nine\s+Stitches\.?", "", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"\s{2,}", " ", raw_text).strip().rstrip(".—–-")
    return f'"{raw_text}"' if raw_text else '"'


def _pick_prompt(quote_text: str) -> str:
    """Deterministic background per quote so a given quote always looks the same."""
    digest = hashlib.sha1(_clean_quote_text(quote_text).encode("utf-8")).hexdigest()
    return BG_PROMPTS[int(digest, 16) % len(BG_PROMPTS)]


def _render_quote(bg: Image.Image, quote_text: str, prompt_index: int = 0) -> Image.Image:
    """Overlay the book text + footer on an AI background at strong contrast."""
    W, H = 1080, 1350
    img = bg.resize((W, H), Image.Resampling.LANCZOS)

    # Heavy dark gradient: keeps the photographic texture but guarantees a
    # high-contrast field for cream serif text (unlike a raw bright photo).
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for y in range(H):
        alpha = 120 + int(60 * (y / H))
        sd.line([(0, y), (W, y)], fill=(*VIGNETTE, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), shade).convert("RGB")

    # Soft central panel echoing the PIL fallback's inset card, blurred so it
    # reads as depth rather than a hard rectangle over the photo.
    panel = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    pd.rectangle((70, 170, W - 70, H - 170), fill=(*VIGNETTE, 95))
    panel = panel.filter(ImageFilter.GaussianBlur(40))
    img = Image.alpha_composite(img.convert("RGBA"), panel).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Brand accent rule above the quote.
    draw.rectangle((W // 2 - 42, 300, W // 2 + 42, 304), fill=ACCENT)

    raw = _clean_quote_text(quote_text)
    header_size = 62
    font = _load_font(header_size)
    max_text_w = 860
    wrap_width = max(16, int(max_text_w / (header_size * 0.48)))
    wrapped = textwrap.wrap(raw, width=wrap_width) or [raw]
    line_hs = [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
               for line in wrapped]
    line_h = max(line_hs) if line_hs else 40
    line_spacing = 26
    th = line_h * len(wrapped) + line_spacing * max(0, len(wrapped) - 1)
    top = 240
    bottom = H - 210
    available = max(0, bottom - top)
    block_y = top + max(0, (available - th) / 2)
    for i, line in enumerate(wrapped):
        lw = draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
        draw.text(((W - lw) / 2, block_y + i * (line_h + line_spacing)),
                  line, font=font, fill=CREAM)

    # Book footer -- identical wording/placement to the PIL renderer, with an
    # accent dot for a touch of brand polish.
    ff = _load_font(28, serif=False)
    fb = draw.textbbox((0, 0), FOOTER_TEXT, font=ff)
    fw, fh = fb[2] - fb[0], fb[3] - fb[1]
    fx = (W - fw) / 2 + 10
    fy = H - 170 - fh
    draw.text((fx, fy), FOOTER_TEXT, font=ff, fill=FOOTER)
    draw.ellipse((fx - 24, fy + 10, fx - 12, fy + 22), fill=ACCENT)

    # Fine grain for texture.
    import random
    rnd = random.Random((prompt_index + 1) * 7919)
    grain = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grain)
    for _ in range(8000):
        x, y = rnd.randrange(W), rnd.randrange(H)
        gd.point((x, y), fill=(255, 255, 255, rnd.randint(3, 9)))
    img = Image.alpha_composite(img.convert("RGBA"), grain).convert("RGB")
    return img


def generate_quote_image_cloudflare(quote_text: str, title: str,
                                    out_path: Optional[str] = None) -> Optional[str]:
    """Render the quote post with a Cloudflare AI background.

    Returns the saved image path on success, or None (unconfigured / failure)
    so publish_instagram_quotes.generate_quote_image() can run unchanged.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not account_id or not token:
        return None
    try:
        prompt = _pick_prompt(quote_text)
        bg = _cf_background(prompt, account_id, token)
        img = _render_quote(bg, quote_text, BG_PROMPTS.index(prompt))
        dest = Path(out_path) if out_path else DEFAULT_OUT
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(dest), format="JPEG", quality=95, optimize=True)
        print(f"  ✓ Cloudflare quote background: {dest}")
        return str(dest)
    except Exception as e:
        print(f"  ⚠ Cloudflare quote image failed ({e}) — using PIL quote fallback.")
        return None
