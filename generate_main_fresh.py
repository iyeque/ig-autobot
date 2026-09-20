"""
Fresh image generation for main post 298.
Builds a wabi-sabi inspired dark image from scratch (no base image).
Then overlays text + WP WIGMAN logo.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os, math, random

BASE = r"C:\Users\Huawei\Downloads\ig-autobot"
OUT = os.path.join(BASE, "images", "post_20260919_070629.jpg")
CLEAN = os.path.join(BASE, "images", "post_20260919_070629_clean.jpg")
W = 1080
H = 1350


def find_font(size=28, bold=False):
    candidates = [
        (r"C:\Windows\Fonts\arialbd.ttf", True),
        (r"C:\Windows\Fonts\arial.ttf", False),
        (r"C:\Windows\Fonts\georgia.ttf", False),
        (r"C:\Windows\Fonts\times.ttf", False),
    ]
    for fp, is_bold in candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def draw_text_with_outline(draw, xy, text, font, fill=(255,255,255), outline=(0,0,0), outline_width=2):
    """Draw text with a dark outline for readability on any background."""
    x, y = xy
    # Draw outline
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    # Draw main text
    draw.text((x, y), text, font=font, fill=fill)


def make_wabi_sabi_background(w, h):
    """Create a dark textured background evoking weathered ceramic / kintsugi aesthetic."""
    img = Image.new("RGB", (w, h), (18, 18, 20))
    arr = np.array(img, dtype=np.float32)

    # Add subtle noise texture (like paper grain)
    noise = np.random.normal(0, 4, (h, w, 3)).astype(np.float32)
    arr += noise

    # Add subtle radial gradient (slightly lighter in center)
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    gradient = 1.0 - (dist / max_dist) * 0.15  # subtle darkening at edges
    gradient = np.clip(gradient, 0.85, 1.0)
    arr *= gradient[:, :, np.newaxis]

    # Add a few random thin gold-ish lines (kintsugi inspiration)
    draw = ImageDraw.Draw(img)
    gold_color = (180, 150, 80)
    for _ in range(3):
        # Random curved line
        start_x = random.randint(100, w - 100)
        start_y = random.randint(100, h - 100)
        points = [(start_x, start_y)]
        for _ in range(5):
            px, py = points[-1]
            nx = px + random.randint(-80, 80)
            ny = py + random.randint(-80, 80)
            points.append((max(0, min(w, nx)), max(0, min(h, ny))))
        # Draw as thin gold line
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=gold_color, width=1)

    arr = np.array(img, dtype=np.uint8)
    return Image.fromarray(arr)


# ── Generate background ──────────────────────────────────────────────

print("Creating wabi-sabi background...")
bg = make_wabi_sabi_background(W, H)
bg.save(CLEAN, quality=95)
print(f"Clean background saved: {CLEAN} ({os.path.getsize(CLEAN)} bytes)")

# ── Apply text overlay ───────────────────────────────────────────────

print("Applying text overlay...")

# Convert to RGBA for compositing
canvas = bg.convert("RGBA")
overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

font_title = find_font(38)
font_subtitle = find_font(28)

lines = [
    ("WABI-SABI IN THE AGE", font_title, 40),
    ("OF ALGORITHMS #20", font_subtitle, 16),
]

# Calculate text block dimensions
total_height = 0
line_info = []
max_width = 0
for text, font, y_offset in lines:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    line_info.append((text, font, width, height, y_offset))
    max_width = max(max_width, width)
    total_height += height + 8  # line spacing

total_height -= 8  # remove last spacing

# Center position
text_x = (W - max_width) // 2
text_y = (H - total_height) // 2

# Semi-transparent dark background bar behind text
pad_x = 40
pad_y = 20
bar_x0 = text_x - pad_x
bar_y0 = text_y - pad_y
bar_x1 = text_x + max_width + pad_x
bar_y1 = text_y + total_height + pad_y

draw.rounded_rectangle(
    [bar_x0, bar_y0, bar_x1, bar_y1],
    radius=12,
    fill=(0, 0, 0, 140),
)

# Draw text with outline
cursor_y = text_y
for text, font, width, height, _ in line_info:
    draw_text_with_outline(draw, (text_x, cursor_y), text, font,
                           fill=(255, 255, 255), outline=(0, 0, 0), outline_width=3)
    cursor_y += height + 8

# Composite
canvas = Image.alpha_composite(canvas, overlay)
canvas_rgb = canvas.convert("RGB")

# ── Apply WP WIGMAN logo ─────────────────────────────────────────────

print("Applying WP WIGMAN logo...")

logo_path = os.path.join(BASE, "wp_logo_white.png")
logo = Image.open(logo_path).convert("RGBA")

# Scale logo to reasonable size (about 80px wide for bottom-right corner)
target_width = 90
scale = target_width / logo.width
new_size = (int(logo.width * scale), int(logo.height * scale))
logo_resized = logo.resize(new_size, Image.LANCZOS)

# Position: bottom-right with margin
margin = 30
logo_x = W - logo_resized.width - margin
logo_y = H - logo_resized.height - margin

print(f"Logo: {new_size} at ({logo_x}, {logo_y})")

canvas_rgb.paste(logo_resized, (logo_x, logo_y), logo_resized)

# Save final
canvas_rgb.save(OUT, quality=92)
print(f"Final image saved: {OUT} ({os.path.getsize(OUT)} bytes)")

# ── Verify ───────────────────────────────────────────────────────────

arr_final = np.array(canvas_rgb)
print(f"\nVerification:")
print(f"  Image size: {W}x{H}")
print(f"  Overall mean: {arr_final.mean(axis=(0,1)).round(1)}")
print(f"  Overall std: {arr_final.std():.1f}")

# Check BR corner for logo
br_region = arr_final[H - logo_resized.height - 10:H - 10,
                      W - logo_resized.width - 10:W - 10]
br_bright = (br_region.mean(axis=2) > 150).sum()
print(f"  BR logo area bright pixels: {br_bright} (should be > 100)")

# Check text area
text_region = arr_final[text_y:text_y + total_height + 20,
                        text_x - 20:text_x + max_width + 20]
text_bright = (text_region.mean(axis=2) > 200).sum()
print(f"  Text area bright pixels: {text_bright} (should be > 50)")

print("\nDone!")
