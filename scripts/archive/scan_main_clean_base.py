from PIL import Image
import numpy as np, glob, os, json

# Scan images/ for clean bases (same method as Wilma scan)
clean = []
for p in sorted(glob.glob('images/post_*.png')) + sorted(glob.glob('images/post_*.jpg')):
    try:
        img = Image.open(p).convert('RGB')
        img_small = img.resize((100, 100))
        arr = np.array(img_small)
        text_rows = sum(1 for y in range(arr.shape[0]) if np.std(arr[y]) > 25)
        pct = (text_rows / arr.shape[0]) * 100
        if pct < 15:
            clean.append({'path': p, 'size': os.path.getsize(p), 'dims': img.size, 'text_pct': round(pct, 1)})
    except:
        pass

# Also check carousel slides
for p in sorted(glob.glob('images/carousel_*_slide_*.jpg')):
    try:
        img = Image.open(p).convert('RGB')
        img_small = img.resize((100, 100))
        arr = np.array(img_small)
        text_rows = sum(1 for y in range(arr.shape[0]) if np.std(arr[y]) > 25)
        pct = (text_rows / arr.shape[0]) * 100
        if pct < 15:
            clean.append({'path': p, 'size': os.path.getsize(p), 'dims': img.size, 'text_pct': round(pct, 1)})
    except:
        pass

clean.sort(key=lambda x: x['text_pct'])
print(f'Clean bases found: {len(clean)}')
for c in clean[:20]:
    print(f'  {c["path"]} {c["dims"]} {c["size"]} text={c["text_pct"]}%')

# Save the best one as the base for bundle 296
if clean:
    best = clean[0]
    import shutil
    shutil.copy2(best['path'], 'images/post_20260922_072453.jpg')
    print(f'\nCopied {best["path"]} -> images/post_20260922_072453.jpg')
