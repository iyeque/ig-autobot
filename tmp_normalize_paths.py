import json
from pathlib import Path

p = Path('forwilma/state.json')
state = json.loads(p.read_text(encoding='utf-8'))
changed = 0
for bundle in [state.get('active_bundle')] + state.get('content_queue', []):
    if not isinstance(bundle, dict):
        continue
    img = bundle.get('image')
    if isinstance(img, str) and '\\' in img:
        bundle['image'] = img.replace('\\', '/')
        changed += 1
if changed:
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(p)
    print(f'Normalized {changed} path(s)')
else:
    print('No backslash paths found')
