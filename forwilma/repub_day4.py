import sys, json, os, shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '..')
from wilma_bot import _generate_wilma_visual_prompt, generate_image, _write_output_jpg, apply_logo_watermark, add_static_text_overlay, _generate_text_ai_horde

STATE_FILE = Path('state.json')
state = json.load(open(STATE_FILE))
schedule = json.load(open('schedule.json'))
day_data = schedule[3]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_name = f"day4_repub_{timestamp}.jpg"
image_path = f"images/{image_name}"

visual_metaphor = _generate_wilma_visual_prompt(day_data["topic"])
# Use env vars if available, otherwise empty strings
base = os.environ.get('WILMA_BRAND_BASE', '')
suffix = os.environ.get('WILMA_BRAND_SUFFIX', '')
image_prompt = ', '.join(filter(None, [base, visual_metaphor, suffix]))
print('Prompt:', image_prompt)

raw_image = generate_image(image_prompt)
processed = _write_output_jpg(raw_image, "temp_output.jpg")
apply_logo_watermark("temp_output.jpg", str(Path('..') / 'assets' / 'digital_guardian_logo.png'))
add_static_text_overlay("temp_output.jpg", day_data["topic"])
shutil.copy("temp_output.jpg", image_path)
print('Saved:', image_path)

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
    f"Topic: {day_data['topic']}\nAudience: {day_data['audience']}\nPlatform: LinkedIn",
    system_prompt=master_system + "\nWrite for LinkedIn.",
    max_tokens=768,
)
bluesky_caption = _generate_text_ai_horde(
    f"Topic: {day_data['topic']}\nAudience: {day_data['audience']}\nPlatform: Bluesky",
    system_prompt=master_system + "\nWrite for Bluesky.",
    max_tokens=768,
)

print('LinkedIn caption:', linkedin_caption[:100])
print('Bluesky caption:', bluesky_caption[:100])

bundle = {
    "post_id": "day_4",
    "timestamp": timestamp,
    "post": day_data,
    "image": image_path,
    "master_reflection": linkedin_caption,
    "bundle_captions": {
        "linkedin": linkedin_caption,
        "bluesky": bluesky_caption,
    },
    "carousel": [],
    "platforms_posted": [],
    "platforms_prepared": [],
}

state.setdefault("content_queue", []).append(bundle)
with open('caption.txt', 'w', encoding='utf-8') as f:
    f.write(linkedin_caption)

tmp = STATE_FILE.with_suffix('.tmp')
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=4, ensure_ascii=False)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, STATE_FILE)
print('state.json updated. queue:', [b.get('post_id') for b in state.get('content_queue', [])])
