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

day5 = next(item for item in sched if item.get('day') == 5)
topic = day5['topic']
cta = day5.get('cta', 'Comment')
pillar = day5.get('pillar', '')
audience = day5.get('audience', '')
graphics = day5.get('graphics', '')
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

for p in FORWILMA_DIR.glob('images/day5_*'):
    p.unlink()
for p in FORWILMA_DIR.glob('images/carousel_*day5*'):
    p.unlink()

image_path = FORWILMA_DIR / f'images/day5_{ts}.jpg'
candidates = sorted(FORWILMA_DIR.glob('images/day*_*.jpg'))
base = candidates[0] if candidates else None
if not base:
    raise RuntimeError('No existing day image to reuse for day_5 hero')
shutil.copy(base, image_path)
add_static_text_overlay(str(image_path), topic)

day5_slides = [
    'Founder life rarely looks like the highlight reel.',
    'Family time is not a reward for busy seasons. It is the baseline.',
    'The myth of balance: nobody warns you it is not equal hours.',
    'What worked: micro-routines, not grand resets.',
    'One question that changed my week: "Who will remember this moment?"',
]
carousel_paths = generate_carousel(
    pillar or day5.get('type', 'General'),
    topic,
    ts,
    footer_text='DIGITAL GUARDIAN | WILMA',
    slides=day5_slides,
)
if not carousel_paths:
    raise RuntimeError('generate_carousel returned no slides')

linkedin = topic + "\n\n" + (
    "We asked founders what they try versus what actually works when it comes to family time. The honest answers surprised us.\n\n"
    "What really works\n"
    "- Micro-routines, not grand resets\n"
    "- Intentional frictions that signal presence\n"
    "- Permission to drop what does not move the story forward\n\n"
    "In my own experience, trying to separate \"founder\" from \"parent\" feels like drawing a line through water. The better move is to ask: who will remember this moment?\n\n"
    "Research backs this in small ways. A 2023 study in the Journal of Occupational Health Psychology found that boundary-crossing rituals, like a ten-minute transition between work and home, reduced emotional exhaustion more than rigid hours did. The ritual matters more than the rule.\n\n"
    "So here is my question for you: when was the last time you redesigned your own day not for efficiency, but for meaning?\n\n"
    "If this resonates, drop a \"1\" in the comments and I will share the framework I use to triage the noise.\n\n"
    "#DigitalGuardian #DigitalParenting #FounderLife #WorkLifeIntegration"
)
bluesky = topic + "\n\n" + (
    "Founder life rarely looks like the highlight reel. The honest version is messier, and often more meaningful.\n\n"
    "- Micro-routines > grand resets\n"
    "- Intentional frictions signal presence\n"
    "- Permission to drop what does not move the story forward\n\n"
    "One question that changed my week: \"Who will remember this moment?\"\n\n"
    "#DigitalGuardian #FounderLife"
)

state['active_bundle'] = {
    'post_id': 'day_5',
    'timestamp': ts,
    'image': f'images/day5_{ts}.jpg',
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
if not queue or queue[0] != 'day_6':
    queue = ['day_6'] + [q for q in queue if q != 'day_6']
state['content_queue'] = queue

with open(STATE_FILE, 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2, ensure_ascii=False)

print('Built day_5 clean state')
print('Hero:', image_path)
print('Carousel slides:', len(carousel_paths))
print('Queue:', queue)
