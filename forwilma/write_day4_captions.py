import sys, json, os
from pathlib import Path

sys.path.insert(0, '..')
from wilma_bot import _generate_text_ai_horde

STATE_FILE = Path('state.json')
state = json.load(open(STATE_FILE))

pending = state.get('pending_bundle') or {}
bundle = pending.get('bundle_captions') or {}
post = pending.get('post') or {}
topic = post.get('topic') or ''
audience = post.get('audience') or 'All'

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
    f"Topic: {topic}\nAudience: {audience}\nPlatform: LinkedIn",
    system_prompt=master_system + "\nWrite for LinkedIn.",
    max_tokens=768,
)
bluesky_caption = _generate_text_ai_horde(
    f"Topic: {topic}\nAudience: {audience}\nPlatform: Bluesky",
    system_prompt=master_system + "\nWrite for Bluesky.",
    max_tokens=768,
)

print('LinkedIn:', linkedin_caption[:100])
print('Bluesky:', bluesky_caption[:100])

bundle['linkedin'] = linkedin_caption or bundle.get('linkedin') or ''
bundle['bluesky'] = bluesky_caption or bundle.get('bluesky') or ''
pending['bundle_captions'] = bundle
state['pending_bundle'] = pending

with open('caption.txt', 'w', encoding='utf-8') as f:
    f.write(bundle['linkedin'])

with open(STATE_FILE, 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=4, ensure_ascii=False)
print('Captions written.')
