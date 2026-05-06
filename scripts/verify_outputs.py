#!/usr/bin/env python3
import json, sys, os

# This script is meant to run in GitHub Actions to verify that the bot
# produced the required files (output.jpg or carousel.json and caption.txt).

if os.path.isfile('carousel.json'):
    with open('carousel.json','r', encoding='utf-8') as f:
        try:
            arr = json.load(f)
        except Exception as e:
            print('❌ Failed to parse carousel.json:', e)
            sys.exit(1)
    if not isinstance(arr, list) or len(arr) < 2:
        print('❌ carousel.json invalid or too few images')
        sys.exit(1)
    print('✓ Carousel manifest OK:', len(arr), 'images')
else:
    if not os.path.isfile('output.jpg'):
        print('❌ Image not generated')
        sys.exit(1)

# Reel support: if reel flag exists, reel.mp4 must exist
if os.path.isfile('post_reel.flag'):
    if not os.path.isfile('reel.mp4'):
        print('❌ Reel flag present but reel.mp4 missing')
        sys.exit(1)
    print('✓ Reel output OK')

# Story support: if story flag exists, story.jpg must exist
if os.path.isfile('post_story.flag'):
    if not os.path.isfile('story.jpg'):
        print('❌ Story flag present but story.jpg missing')
        sys.exit(1)
    print('✓ Story output OK')

if not os.path.isfile('caption.txt'):
    print('❌ Caption not generated')
    sys.exit(1)

print('✓ Outputs verified')
