#!/usr/bin/env python3
"""Finish Wilma day_11: generate master reflection + LinkedIn/Bluesky captions
using bot.py's own AI Horde functions, then patch the queued bundle."""
import json
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'forwilma')

from bot import _generate_text_ai_horde, _ai_verify_caption  # noqa: E402
from shared_utils import clean_caption_formatting  # noqa: E402
from wilma_bot import _enforce_wilma_persona, _strip_bluesky_cta, DIGITAL_GUARDIAN_MISSION  # noqa: E402

STATE = 'forwilma/state.json'

s = json.load(open(STATE, encoding='utf-8'))
queue = s.get('content_queue', [])
day11 = next((b for b in queue if b.get('post_id') == 'day_11'), None)
if not day11:
    print('day_11 not in queue'); sys.exit(1)

post = {
    'topic': 'What happened when a classroom banned phones for a semester',
    'audience': 'All',
    'pillar': 'Proof Wednesday',
}

master_system = f"""You are the lead strategist for Digital Guardian, writing as Wilma. Mission: {DIGITAL_GUARDIAN_MISSION}
Voice rules:
- Empathetic, authoritative, research-backed, and relatable. Never preachy.
- Write like a founder who has lived the tension between "scary tech" and healthy family life.
- Use real references when relevant: American Academy of Pediatrics, University of Michigan, etc.
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

print('Generating master reflection...')
reflection = ''
for attempt in range(2):
    reflection = _generate_text_ai_horde(
        f"Topic: {post['topic']}\nAudience: {post['audience']}",
        system_prompt=master_system,
        max_tokens=500,  # AI Horde KudosUpfront: >512 tokens requires kudos we don't have
    )
    print(f'  attempt {attempt+1}: {len(reflection)} chars')
    if reflection and reflection.rstrip().endswith(('.', '!', '?', '…', ':', ';')):
        break

if not reflection:
    print('REFLECTION FAILED — aborting, no state change')
    sys.exit(1)

print('Reflection preview:', reflection[:200])
day11['master_reflection'] = reflection

captions = {}
for p, max_c in [('linkedin', 1800), ('bluesky', 250)]:
    print(f'Tailoring {p}...')
    tailored = _ai_verify_caption(reflection, p, max_c)
    final = clean_caption_formatting(tailored or '')
    final = _enforce_wilma_persona(final)
    if p == 'linkedin':
        final += '\n\n#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips'
    else:
        final = _strip_bluesky_cta(final) + '\n\nWant to read more?... check out my LinkedIn'
    limit = 2000 if p == 'linkedin' else 300
    if len(final) > limit:
        final = final[:limit - 3] + '...'
    captions[p] = final
    print(f'  {p}: {len(final)} chars')

day11['captions'] = captions
json.dump(s, open(STATE, 'w', encoding='utf-8'), indent=2)
print('day_11 patched: reflection + captions written')
