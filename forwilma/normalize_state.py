#!/usr/bin/env python3
import json
from pathlib import Path

STATE_FILE = Path('state.json')
SCHEDULE_FILE = Path('schedule.json')

with open(STATE_FILE, 'r', encoding='utf-8') as f:
    state = json.load(f)
with open(SCHEDULE_FILE, 'r', encoding='utf-8') as f:
    sched = {item['day']: item for item in json.load(f)}

def ensure_dict(bundle, day):
    if isinstance(bundle, dict):
        return bundle
    if isinstance(bundle, str):
        day_num = None
        s = str(bundle)
        if s.startswith('day_'):
            try:
                day_num = int(s.split('_', 1)[1])
            except ValueError:
                day_num = None
        day_info = sched.get(day_num, {})
        topic = day_info.get('topic', '')
        cta = day_info.get('cta', 'Comment')
        pillar = day_info.get('pillar', day_info.get('type', 'General'))
        audience = day_info.get('audience', 'All')
        graphics = day_info.get('graphics', '')
        candidates = sorted(Path('images').glob('day{}_*.jpg'.format(day_num)))
        image = 'images/{}'.format(candidates[0].name) if candidates else ''
        carousel = ['{}'.format(p) for p in sorted(Path('images').glob('carousel_*day{}*.jpg'.format(day_num)))]
        return {
            'post_id': s,
            'timestamp': '',
            'image': image,
            'carousel': carousel,
            'captions': {},
            'platforms_posted': [],
            'pillar': pillar,
            'audience': audience,
            'topic': topic,
            'cta': cta,
            'graphics': graphics,
        }
    return {'post_id': str(bundle)}

ab = state.get('active_bundle')
if isinstance(ab, (int, str)):
    s = str(ab)
    day_num = None
    if s.startswith('day_'):
        try:
            day_num = int(s.split('_', 1)[1])
        except ValueError:
            pass
    state['active_bundle'] = ensure_dict(ab, day_num)
elif isinstance(ab, dict):
    carousel = ab.get('carousel') or []
    ab['carousel'] = [str(p).replace('\\', '/') for p in carousel]

queue = state.get('content_queue', [])
new_queue = []
for item in queue:
    if isinstance(item, (int, str)):
        s = str(item)
        day_num = None
        if s.startswith('day_'):
            try:
                day_num = int(s.split('_', 1)[1])
            except ValueError:
                pass
        new_queue.append(ensure_dict(item, day_num))
    else:
        new_queue.append(item)
state['content_queue'] = new_queue

with open(STATE_FILE, 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2, ensure_ascii=False)

print('Normalized state:')
print('active post_id:', state['active_bundle'].get('post_id'))
print('queue types:', [type(q).__name__ for q in state['content_queue']])
print('queue post_ids:', [q.get('post_id') if isinstance(q, dict) else q for q in state['content_queue']])
