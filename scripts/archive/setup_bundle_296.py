#!/usr/bin/env python3
"""Setup bundle 296 for end-to-end posting test."""
import json
from pathlib import Path

state_path = Path('state.json')
state = json.load(state_path.open('r', encoding='utf-8'))

# Setup active bundle for 296
active = {
    'post_id': '296',
    'pillar': 'systems_psychology',
    'topic': 'The Feedback Loop We Live In #18',
    'title': 'The Feedback Loop We Live In #18',
    'image': 'images/post_20260922_072453.jpg',
    'image_prompt': 'minimalist circuit diagram merged with tree roots',
    'timestamp': '20260922_072453',
    'master_reflection': None,
    'captions': {},
    'carousel': [],
    'reel': 'reels/reel_20260922_072453.mp4',
}

# Set active
state['active_bundle'] = active

# Make sure 296 is NOT in used_ids so it can be posted
for platform in state.get('used_ids', {}):
    if 296 in state['used_ids'][platform]:
        state['used_ids'][platform].remove(296)
    if '296' in state['used_ids'][platform]:
        state['used_ids'][platform].remove('296')

# Make sure 296 is NOT in ppb so we can re-post
for platform in state.get('platform_posted_bundles', {}):
    if 296 in state['platform_posted_bundles'][platform]:
        state['platform_posted_bundles'][platform].remove(296)
    if '296' in state['platform_posted_bundles'][platform]:
        state['platform_posted_bundles'][platform].remove('296')

# Clear pending/state artifacts
state['pending_bundle'] = None
state['content_queue'] = []

state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
print('State setup for bundle 296:')
print(json.dumps(state.get('active_bundle'), indent=2)[:600])
