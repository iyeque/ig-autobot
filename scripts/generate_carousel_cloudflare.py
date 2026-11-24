#!/usr/bin/env python3
"""Cloudflare-workers-backed carousel slides with the existing PIL renderer as fallback.

Main brand only. Generates one AI background per slide via Cloudflare Workers AI
(flux-1-schnell), applies an editorial overlay (serif hero text, warm cream palette,
brand accent, soft vignette, fine grain), and returns the same 1080x1350 slide list
the publisher expects. If Cloudflare is unconfigured or any request fails, control
returns None and the caller uses bot.generate_carousel() unchanged.
"""
import os
import io
import base64
import textwrap
from typing import List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter


CLOUDFLARE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
# Consistent theme per slide; the narrative text stays separate and is overlaid.
BG_PROMPTS = [
    "ethereal winter nature photography, snow falling on bare branches, soft slate-teal bokeh, dreamlike calm, minimal, cinematic, negative space in center, no text, no words, no letters",
    "misty winter forest path at dawn, faint footprints, desaturated blue-grey, gentle bokeh, contemplative, minimal, cinematic, open center space, no text, no words, no letters",
    "bare winter trees silhouetted against pale overcast sky, soft light, muted cool palette, dreamlike stillness, minimal, cinematic, open center, no text, no words, no letters",
    "delicate frost patterns on window glass at blue hour, deep slate tones, soft glow, contemplative, minimal, cinematic, open center, no text, no words, no letters",
    "quiet winter landscape at dusk, frost on grass, faded indigo and grey, dreamlike stillness, minimal, cinematic, open center, no text, no words, no letters",
]

# Palette
CREAM = (244, 238, 224)
MUTED = (198, 190, 172)
ACCENT = (214, 160, 110)   # warm terracotta/gold
VIGNETTE = (6, 9, 14)


def _font(size: int, serif: bool = False, bold: bool = False):
    if serif:
        cands = (["C:/Windows/Fonts/georgiab.ttf", "C:/Windows/Fonts/timesbd.ttf"] if bold else
                 ["C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/times.ttf"])
        cands += ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else
                  "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "DejaVuSerif.ttf"]
    else:
        cands = (["C:/Windows/Fonts/arialbd.ttf"] if bold else ["C:/Windows/Fonts/arial.ttf"])
        cands += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]
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

    # Dark radial-ish vignette: heavier at edges and lower half, lighter center so
    # the hero text pops without flattening the whole photograph.
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for y in range(H):
        base = 96 + int(70 * (y / H))          # darker toward bottom
        d.line([(0, y), (W, y)], fill=(*VIGNETTE, base))
    # center soft-light window
    cw, ch = 900, 620
    cx, cy = (W - cw) // 2, 300
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((cx - 60, cy - 40, cx + cw + 60, cy + ch + 40), fill=(0, 0, 0, 85))
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    ov = Image.alpha_composite(ov, glow)
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    d = ImageDraw.Draw(img)

    # Localized scrim behind the hero text so snowy branches can't steal contrast.
    scrim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(scrim)
    sd.rectangle((70, 440, W - 70, 1010), fill=(6, 9, 14, 92))
    scrim = scrim.filter(ImageFilter.GaussianBlur(45))
    img = Image.alpha_composite(img.convert("RGBA"), scrim).convert("RGB")
    d = ImageDraw.Draw(img)

    # Brand accent: short rule above the quote + slide number, no rigid frame.
    accent_x, accent_y = 90, 230
    d.rectangle((accent_x, accent_y, accent_x + 64, accent_y + 4), fill=ACCENT)
    d.text((W - 150, accent_y - 10), f"{idx:02d}", font=_font(26, serif=True), fill=MUTED)
    # Small brand wordmark top-left — makes the brand unmistakable, not just a credit.
    d.text((accent_x, accent_y + 18), "THE NINE STITCHES", font=_font(20, serif=True), fill=MUTED)

    # Hero quote — serif, warm cream, generous leading.
    clean = "\n".join(l.strip() for l in text.split("\n") if l.strip())
    clean = clean.replace("Slide #2: ", "").replace("Slide #3: ", "").replace("Slide #4: ", "")
    raw_line = max(clean.split("\n"), key=len)
    width = max(14, int(760 / (48 * 0.5)))
    lines = []
    for part in clean.split("\n"):
        lines += textwrap.wrap(part, width=width) or [""]
    if len(lines) > 4:
        lines = lines[:4]
        lines[-1] = lines[-1].rstrip() + "…"
    f = _font(58, serif=True, bold=is_cta)
    leading = 76
    th = leading * len(lines)
    y0 = 470
    for li, line in enumerate(lines):
        bb = d.textbbox((0, 0), line, font=f)
        lw = bb[2] - bb[0]
        d.text(((W - lw) // 2, y0 + li * leading), line, font=f, fill=CREAM)

    # Footer with a small accent dot — turns generic quote into branded asset.
    if not is_cta and footer:
        ff = _font(28, serif=False)
        bb = d.textbbox((0, 0), footer, font=ff)
        fw = bb[2] - bb[0]
        fx = (W - fw) // 2 + 12
        fy = H - 190
        d.text((fx, fy), footer, font=ff, fill=CREAM)
        d.ellipse((fx - 24, fy + 10, fx - 12, fy + 22), fill=ACCENT)

    # Fine grain for texture (cheap: sparse low-alpha noise).
    import random
    rnd = random.Random(idx * 7919)
    grain = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grain)
    for _ in range(9000):
        x, y = rnd.randrange(W), rnd.randrange(H)
        gd.point((x, y), fill=(255, 255, 255, rnd.randint(3, 10)))
    img = Image.alpha_composite(img.convert("RGBA"), grain).convert("RGB")
    return img


def generate_carousel_cloudflare(topic: str, footer_text: str, timestamp: str,
                                 slide_texts: List[str]) -> Optional[List[str]]:
    """Return 5 slide paths using Cloudflare, or None to fall back to PIL."""
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
            print(f"  ✓ Cloudflare carousel slide {i+1}: {p}")
        return paths
    except Exception as e:
        print(f"  ⚠ Cloudflare carousel failed ({e}) — using PIL carousel fallback.")
        return None
