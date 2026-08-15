import json
with open('state.json', 'r', encoding='utf-8') as f:
    state = json.load(f)
ab = state.get('active_bundle')
if isinstance(ab, dict):
    ab['carousel'] = [p.replace('\\', '/').replace('\\\\', '/') for p in ab.get('carousel', [])]
    if isinstance(ab.get('image'), str):
        ab['image'] = ab['image'].replace('\\', '/').replace('\\\\', '/')
    for item in state.get('content_queue', []):
        if isinstance(item, dict):
            item['carousel'] = [p.replace('\\', '/').replace('\\\\', '/') for p in item.get('carousel', [])]
            if isinstance(item.get('image'), str):
                item['image'] = item['image'].replace('\\', '/').replace('\\\\', '/')
with open('state.json', 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2, ensure_ascii=False)
print('normalized paths')
