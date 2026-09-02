import sys, json, os, shutil, textwrap
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '..')
from wilma_bot import apply_logo_watermark
from wilma_bot import _generate_text_ai_horde

STATE_FILE = Path('state.json')
state = json.load(open(STATE_FILE))
schedule = json.load(open('schedule.json'))
day_data = schedule[3]
topic = day_data['topic']

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_name = f"day4_repub_{timestamp}.jpg"
image_path = f"images/{image_name}"

base_png = Path('images/post_20260504_111405.png')
if not base_png.exists():
    raise SystemExit(f'Missing base image: {base_png}')

base_img = Image.open(base_png).convert('RGB')
base_img.save('temp_output.jpg', format='JPEG', quality=95, optimize=True)
print('Base loaded from user-specified PNG')

# Logo watermark
try:
    apply_logo_watermark('temp_output.jpg', str(Path('..') / 'assets' / 'digital_guardian_logo.png'))
    print('Logo watermark applied')
except Exception as e:
    print('Logo watermark failed:', e)

# Fixed overlay helper
def _load_font(size: int):
    paths = [
        "DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "Arial Bold.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

def clean_text_overlay(image_path: str, text_overlay: str) -> str:
    overlay = (text_overlay or "").strip().replace("\n", " ")
    if not overlay:
        return image_path
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    wrapped = "\n".join(textwrap.wrap(overlay.upper(), width=18)) if overlay else ""
    font_size = 75
    font = _load_font(font_size)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=20, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    max_box_w, max_box_h = w - 80, h - 80
    while (th > max_box_h or tw > max_box_w) and font_size > 28:
        font_size -= 4
        font = _load_font(font_size)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=20, align="center")
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    clear_margin = 160
    cx1 = max(0, (w - max(tw + clear_margin * 2, max_box_w)) // 2)
    cy1 = max(0, (h - max(th + clear_margin * 2, max_box_h)) // 2)
    cx2 = min(w, w - cx1)
    cy2 = min(h, h - cy1)
    draw.rectangle((cx1, cy1, cx2, cy2), fill=(0, 0, 0))
    pad_x, pad_y = 60, 50
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2
    box_x = int((w - box_w) // 2)
    box_y = int((h - box_h) // 2 - (h * 0.05))
    box_x = max(20, min(box_x, w - box_w - 20))
    box_y = max(20, min(box_y, h - box_h - 20))
    overlay_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay_layer)
    odraw.rectangle((box_x, box_y, box_x + box_w, box_y + box_h), fill=(0, 0, 0, 180))
    img = Image.alpha_composite(img.convert("RGBA"), overlay_layer).convert("RGB")
    draw = ImageDraw.Draw(img)
    tx = (w - tw) // 2
    ty = box_y + pad_y
    draw.multiline_text((tx, ty), wrapped, font=font, fill=(255, 255, 255), spacing=20, align="center")
    img.save(image_path, format="JPEG", quality=95, optimize=True)
    return image_path

clean_text_overlay('temp_output.jpg', topic)
shutil.copy('temp_output.jpg', image_path)
print('Saved clean image:', image_path)

master_system = """You are the lead strategist for Digital Guardian, writing as Wilma. Mission: Empower families and professionals to thrive in a balanced digital life through relatable, research-backed guidance.

Voice rules:
- Empathetic, authoritative, research-backed, and relatable. Never preachy.
- Write like a founder who has lived the tension between "scary tech" and healthy family life.
- Keep language plain and warm. Personal and vulnerable in Builder content; practical and direct in educational content.
- Every claim should feel earned by story or result, not guru lecture.
- End with a single, low-friction engagement hook. No jargon, no marketing fluff, no AI-isms.
- CRITICAL: Wilma has ONE daughter, age 2. When content involves children, frame examples ONLY around her 2-year-old daughter, OR use generic collective terms like "kids," "children," or "families." NEVER invent stories about other specific children with different ages. NEVER say "my 4-year-old," "my 5-year-old," or any age other than 2.
- If the topic implies a different age, adapt it to her 2-year-old daughter or use a generic framing.
- Stay in the digital wellness lane 80% of the time. Cross-venture content is allowed only in Builder posts as founder-life context, never as standalone promo.
- For Bluesky: keep it tighter and conversational, end with the fixed CTA line only.
- For LinkedIn: keep it longer and platform-native, but still avoid mid-thought cutoffs.
Write a complete, polished post about the topic below. Finish every sentence. Do not trail off mid-thought.
"""

linkedin_caption = _generate_text_ai_horde(
    f"Topic: {topic}\nAudience: {day_data['audience']}\nPlatform: LinkedIn",
    system_prompt=master_system + "\nWrite for LinkedIn.",
    max_tokens=768,
)
bluesky_caption = _generate_text_ai_horde(
    f"Topic: {topic}\nAudience: {day_data['audience']}\nPlatform: Bluesky",
    system_prompt=master_system + "\nWrite for Bluesky.",
    max_tokens=768,
)

print('LinkedIn:', linkedin_caption[:120])
print('Bluesky:', bluesky_caption[:120])

state['content_queue'] = [b for b in state.get('content_queue', []) if not (isinstance(b, dict) and b.get('post_id') == 'day_4')]
bundle = {
    "post_id": "day_4",
    "timestamp": timestamp,
    "post": day_data,
    "image": image_path,
    "master_reflection": linkedin_caption,
    "bundle_captions": {"linkedin": linkedin_caption, "bluesky": bluesky_caption},
    "captions": {"linkedin": linkedin_caption, "bluesky": bluesky_caption},
    "carousel": [],
    "platforms_posted": [],
    "platforms_prepared": [],
}
state['content_queue'].append(bundle)
with open('caption.txt', 'w', encoding='utf-8') as f:
    f.write(linkedin_caption)
with open(STATE_FILE, 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=4, ensure_ascii=False)

print('state.json updated. queue:', [b.get('post_id') for b in state.get('content_queue', [])])
