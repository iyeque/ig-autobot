#!/usr/bin/env python3
import json, shutil, os, sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from bot import add_static_text_overlay, generate_carousel

FORWILMA_DIR = Path(__file__).parent
STATE_FILE = FORWILMA_DIR / 'state.json'
SCHEDULE_FILE = FORWILMA_DIR / 'schedule.json'

with open(STATE_FILE, 'r', encoding='utf-8') as f:
    state = json.load(f)
with open(SCHEDULE_FILE, 'r', encoding='utf-8') as f:
    sched = json.load(f)

day6 = next(item for item in sched if item.get('day') == 6)
topic = day6['topic']
cta = day6.get('cta', 'Poll')
pillar = day6.get('pillar', '')
audience = day6.get('audience', '')
graphics = day6.get('graphics', '')
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# Clean old day_6 assets
for p in FORWILMA_DIR.glob('images/day6_*'):
    p.unlink()
for p in FORWILMA_DIR.glob('images/carousel_*day6*'):
    p.unlink()

# Hero image
image_path = FORWILMA_DIR / f'images/day6_{ts}.jpg'
candidates = sorted(FORWILMA_DIR.glob('images/day*_*.jpg'))
base = candidates[0] if candidates else None
if not base:
    raise RuntimeError('No existing day image to reuse for day_6 hero')
shutil.copy(base, image_path)
add_static_text_overlay(str(image_path), topic)

# Day-6-specific carousel text
day6_slides = [
    'What screen-time rule actually works in your house?',
    'Top-down limits vs. co-created agreements.',
    'The paradox: stricter rules often backfire.',
    'What actually helped our family was simpler than I expected.',
    'Drop your one rule in the comments.',
]
carousel_paths = generate_carousel(
    pillar or day6.get('type', 'General'),
    topic,
    ts,
    footer_text='DIGITAL GUARDIAN | WILMA',
    slides=day6_slides,
)
if not carousel_paths:
    raise RuntimeError('generate_carousel returned no slides')

linkedin = topic + "\n\n" + (
    "We polled families on what they try versus what actually sticks. The honest answers surprised us.\n\n"
    "Top-down limits\n"
    "- Time limits per day: tried by most, honored by few.\n"
    "- No screens before bedtime: sounds sensible, yet chaos erupts around 7 p.m.\n"
    "- Screen-free zones: popular in theory, tougher when a toddler craves cartoons at breakfast.\n\n"
    "What actually worked\n"
    "- Rules co-created with kids consistently earned the highest reliability score.\n"
    "- Co-watching and pausing to discuss content mattered more than timers.\n"
    "- A ten-minute transition ritual between activities beat rigid hours.\n\n"
    "In my own home, I watch a two-year-old unload an iPad faster than I can pour coffee. We haven’t imposed rigid limits; we’re building awareness first. She already knows the tablet goes back in its case when the songs end.\n\n"
    "So here’s my question for you: What single rule do you actually enforce? Did it emerge from a conversation with your child, or was it imposed from above?\n\n"
    "Share in the comments and I’ll share the raw data next week.\n\n"
    "#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips"
)
bluesky = topic + "\n\n" + (
    "Top-down limits vs. co-created agreements.\n\n"
    "- Time limits: tried by most, honored by few.\n"
    "- No screens before bedtime: chaos at 7 p.m.\n"
    "- Co-created rules: highest reliability score.\n\n"
    "One rule that actually worked for us: the tablet goes back in its case when the songs end.\n\n"
    "#DigitalGuardian #DigitalParenting"
)

state['active_bundle'] = {
    'post_id': 'day_6',
    'timestamp': ts,
    'image': f'images/day6_{ts}.jpg',
    'carousel': [str(Path(p)) for p in carousel_paths],
    'captions': {
        'linkedin': linkedin.strip(),
        'bluesky': bluesky.strip(),
    },
    'platforms_posted': [],
    'pillar': pillar,
    'audience': audience,
    'topic': topic,
    'cta': cta,
    'graphics': graphics,
}
queue = state.get('content_queue', [])
if not queue or queue[0] != 'day_7':
    queue = ['day_7'] + [q for q in queue if q != 'day_7']
state['content_queue'] = queue

with open(STATE_FILE, 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2, ensure_ascii=False)

print('Built day_6 clean state')
print('Hero:', image_path)
print('Carousel slides:', len(carousel_paths))
print('Queue:', [q.get('post_id') if isinstance(q, dict) else q for q in queue])
