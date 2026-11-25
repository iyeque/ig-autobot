#!/usr/bin/env python3
"""Digital Guardian / Wilma Cloudflare-backed carousel slides.

Wilma's brand = light, airy cream paper + centered serif text + warm footer,
NOT main's dark editorial card. This module generates one soft daylight AI
background per slide via Cloudflare Workers AI (flux-1-schnell), then applies
Wilma-specific overlay: near-black serif hero text, cream scrim, sage-teal
accent (Digital Guardian), slide number, and her footer on the CTA slide only.

Returns None when unconfigured/failed so generate_carousel_from_bundle.py uses
bot.generate_wilma_carousel() (the existing PIL renderer) unchanged.
"""
import os
import io
import base64
import textwrap
from typing import List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter


CLOUDFLARE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
# Light, warm, calm imagery that suits a parenting/safety brand and carries
# dark serif text without a heavy dark overlay.
BG_PROMPTS = [
    "soft sunlit nature photograph, gentle morning light through leaves, warm cream and pale sage palette, dreamy bokeh, calm minimal, airy composition, open center space, no text, no words, no letters",
    "warm sunlit kitchen table with a single plant, soft natural window light, cream and light tan palette, minimal calm lifestyle, airy, open center, no text, no words, no letters",
    "tender candid moment of a parent and child walking outdoors in soft daylight, warm muted tones, gentle bokeh, calm minimalist, airy, open center, no text, no words, no letters",
    "soft abstract watercolor wash in cream, pale sage and warm sand, delicate light, minimal calm texture, airy, open center, no text, no words, no letters",
    "sunlit window with sheer curtain and soft greenery, warm cream light, airy minimalist, gentle calm, open center, no text, no words, no letters",
]

CREAM = (245, 242, 235)     # matches Wilma's BG
TEXT = (18, 18, 18)         # matches Wilma's TEXT_COLOR
FOOT = (90, 85, 78)         # matches Wilma's FOOTER_COLOR
ACCENT = (74, 110, 106)     # calm sage-teal — Digital Guardian accent


def _font(size: int, bold: bool = False):
    cands = (["C:/Windows/Fonts/timesbd.ttf", "C:/Windows/Fonts/georgiab.ttf",
              "C:/Windows/Fonts/LiberationSerif-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"] if bold else
             ["C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/georgia.ttf",
              "C:/Windows/Fonts/LiberationSerif-Regular.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"])
    for p in cands:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _cf_background(prompt: str, account_id: str, token: str) -> Image.Image:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{CLOUDFLARE_MODEL}"
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


def _render_slide(bg: Image.Image, text: str, idx: int, footer: str, is_cta: bool) -> Image.Image:
    W, H = 1080, 1350
    img = bg.resize((W, H), Image.Resampling.LANCZOS)

    # Light cream veil so the photo stays airy but dark serif text stays legible.
    veil = Image.new("RGBA", (W, H), (*CREAM, 178))
    img = Image.alpha_composite(img.convert("RGBA"), veil).convert("RGB")
    # Soft cream glow behind the text block for extra separation.
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((110, 420, W - 110, 1000), fill=(*CREAM, 90))
    glow = glow.filter(ImageFilter.GaussianBlur(80))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    d = ImageDraw.Draw(img)

    # Brand accents — sage rule + wordmark + slide number.
    ax, ay = 90, 230
    d.rectangle((ax, ay, ax + 64, ay + 4), fill=ACCENT)
    d.text((ax, ay + 18), "DIGITAL GUARDIAN", font=_font(20), fill=FOOT)
    d.text((W - 150, ay - 10), f"{idx:02d}", font=_font(26), fill=FOOT)

    clean = "\n".join(l.strip() for l in text.split("\n") if l.strip())
    for p in ("Slide #2: ", "Slide #3: ", "Slide #4: "):
        clean = clean.replace(p, "")
    longest = max(clean.split("\n"), key=len)
    width = max(14, int(760 / (58 * 0.5)))
    lines = []
    for part in clean.split("\n"):
        lines += textwrap.wrap(part, width=width) or [""]
    if len(lines) > 4:
        lines = lines[:4]
        lines[-1] = lines[-1].rstrip() + "…"
    f = _font(60 if is_cta else 56, bold=is_cta)
    leading = 78
    y0 = 470
    for li, line in enumerate(lines):
        bb = d.textbbox((0, 0), line, font=f)
        lw = bb[2] - bb[0]
        d.text(((W - lw) // 2, y0 + li * leading), line, font=f, fill=TEXT)

    # Wilma shows the footer only on the CTA slide (matches her PIL renderer).
    if is_cta and footer:
        ff = _font(28)
        bb = d.textbbox((0, 0), footer, font=ff)
        fw = bb[2] - bb[0]
        fx = (W - fw) // 2 + 10
        fy = H - 200
        d.text((fx, fy), footer, font=ff, fill=FOOT)
        d.ellipse((fx - 24, fy + 10, fx - 12, fy + 22), fill=ACCENT)

    # Fine paper grain (dark specks on cream), sparse and low-alpha.
    import random
    rnd = random.Random(idx * 104729)
    grain = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grain)
    for _ in range(9000):
        x, y = rnd.randrange(W), rnd.randrange(H)
        gd.point((x, y), fill=(40, 35, 30, rnd.randint(3, 9)))
    img = Image.alpha_composite(img.convert("RGBA"), grain).convert("RGB")
    return img


def generate_carousel_cloudflare_wilma(topic: str, footer_text: str, timestamp: str,
                                       slide_texts: List[str]) -> Optional[List[str]]:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not account_id or not token or len(slide_texts) != 5:
        return None
    try:
        os.makedirs("images", exist_ok=True)
        paths = []
        for i, text in enumerate(slide_texts):
            bg = _cf_background(BG_PROMPTS[i], account_id, token)
            slide = _render_slide(bg, text, i + 1, footer_text, is_cta=(i == 4))
            p = f"images/carousel_{timestamp}_slide_{i+1}.jpg"
            slide.save(p, format="JPEG", quality=95, optimize=True)
            paths.append(p)
            print(f"  ✓ Cloudflare Wilma carousel slide {i+1}: {p}")
        return paths
    except Exception as e:
        print(f"  ⚠ Cloudflare Wilma carousel failed ({e}) — using PIL Wilma fallback.")
        return None
