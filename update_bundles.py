#!/usr/bin/env python3
import os
import json

def update_state_bundles(state_path):
    if not os.path.exists(state_path):
        print(f"{state_path} not found")
        return

    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    content_queue = state.get('content_queue', [])
    is_wilma = 'forwilma' in state_path

    for bundle in content_queue:
        captions = bundle.get('captions', {})

        # Fix Threads: remove hashtags
        if 'threads' in captions:
            threads_cap = captions['threads']
            # Split into lines and remove lines that are only hashtags
            lines = threads_cap.split('\n')
            cleaned_lines = []
            for line in lines:
                line_stripped = line.strip()
                if not (line_stripped.startswith('#') and ' ' not in line_stripped or len(line_stripped) > 0 and all(c == '#' or c.isalpha() or c.isspace() or c in ['-', '_'] for c in line_stripped)):
                    cleaned_lines.append(line)
            captions['threads'] = '\n'.join(cleaned_lines)

        # Fix Bluesky: add the "Want to read more..." line
        if 'bluesky' in captions:
            bluesky_cap = captions['bluesky']
            if 'Want to read more' not in bluesky_cap:
                # Find a good place to add the CTA (end of caption)
                if bluesky_cap.strip() != '':
                    bluesky_cap = bluesky_cap.rstrip() + '\n\nWant to read more?... check out my LinkedIn'
                captions['bluesky'] = bluesky_cap

        bundle['captions'] = captions

    state['content_queue'] = content_queue
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=4)
    print(f"Updated {state_path}")

if __name__ == '__main__':
    update_state_bundles('state.json')
    update_state_bundles('forwilma/state.json')
