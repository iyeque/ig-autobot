#!/usr/bin/env python3
"""Seed Wilma pending_bundle with day_11 so wilma_bot can resume it."""
import json

sched = json.load(open('forwilma/schedule.json', encoding='utf-8'))
s = json.load(open('forwilma/state.json', encoding='utf-8'))

day11 = next((d for d in sched if d.get('day') == 11), None)
print('schedule day_11:', {k: str(v)[:60] for k, v in day11.items()} if day11 else 'NOT FOUND')

s['pending_bundle'] = {
    'post_id': 'day_11',
    'timestamp': '20260921_001628',
    'post': day11,
    'image': 'images/day11_20260921_001628.jpg',
    'master_reflection': None,
    'bundle_captions': {},
}
s['content_queue'] = [b for b in s.get('content_queue', []) if b.get('post_id') != 'day_11']
json.dump(s, open('forwilma/state.json', 'w', encoding='utf-8'), indent=2)
print('pending seeded, queue:', [b.get('post_id') for b in s['content_queue']])
