#!/usr/bin/env python3
import json, sys, os

# prints space-separated paths from carousel.json or exits nonzero if invalid

if not os.path.isfile('carousel.json'):
    sys.exit(0)

with open('carousel.json','r', encoding='utf-8') as f:
    try:
        arr = json.load(f)
    except Exception as e:
        print('')
        sys.exit(1)

if not isinstance(arr, list):
    sys.exit(1)

print(' '.join(arr))
