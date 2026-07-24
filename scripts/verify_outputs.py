#!/usr/bin/env python3
import json, sys, os

# This script is meant to run in GitHub Actions to verify that the bot
# produced the required files (output.jpg or carousel.json and caption.txt).

has_carousel = os.path.isfile('carousel.json')
if has_carousel:
    with open('carousel.json','r', encoding='utf-8') as f:
        try:
            arr = json.load(f)
        except Exception as e:
            print('❌ Failed to parse carousel.json:', e)
            sys.exit(1)
    if not isinstance(arr, list) or len(arr) < 2:
        print('❌ carousel.json invalid or too few images')
        sys.exit(1)
    print('[OK] Carousel manifest:', len(arr), 'images')
else:
    if not os.path.isfile('output.jpg'):
        print('❌ Image not generated')
        sys.exit(1)

# Reel support: if reel flag exists, reel.mp4 must exist
if os.path.isfile('post_reel.flag'):
    if not os.path.isfile('reel.mp4'):
        print('❌ Reel flag present but reel.mp4 missing')
        sys.exit(1)
    print('[OK] Reel output present')

# Story support: if story flag exists, story.jpg must exist
if os.path.isfile('post_story.flag'):
    if not os.path.isfile('story.jpg'):
        print('❌ Story flag present but story.jpg missing')
        sys.exit(1)
    print('[OK] Story output present')

if not os.path.isfile('caption.txt'):
    print('❌ Caption not generated')
    sys.exit(1)

with open('caption.txt', 'r', encoding='utf-8') as f:
    caption = f.read().strip()
if not caption:
    print('❌ Caption file is empty')
    sys.exit(1)

print('[OK] Outputs verified')
