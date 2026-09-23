from PIL import Image
import numpy as np, glob, os, json

clean = []
for p in sorted(glob.glob('forwilma/images/post_*.png')):
    try:
        img = Image.open(p).convert('RGB')
        img_small = img.resize((100, 100))
        arr = np.array(img_small)
        text_rows = sum(1 for y in range(arr.shape[0]) if np.std(arr[y]) > 25)
        pct = (text_rows / arr.shape[0]) * 100
        # Accept if < 15% of rows have text (will be covered by brand box)
        if pct < 15:
            clean.append({
                'path': p,
                'size': os.path.getsize(p),
                'dims': img.size,
                'text_pct': round(pct, 1),
            })
    except:
        pass

# Sort by text_pct (least text first)
clean.sort(key=lambda x: x['text_pct'])

print(f'Acceptable fallback bases (< 15%% text rows): {len(clean)}')
for c in clean[:25]:
    print(f'  {c["path"]} {c["dims"]} {c["size"]} text={c["text_pct"]}%')

json.dump({'clean_bases': clean}, open('forwilma/clean_bases.json', 'w', encoding='utf-8'), indent=2)
print(f'\nSaved forwilma/clean_bases.json')
