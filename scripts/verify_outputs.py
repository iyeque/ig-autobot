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

if not os.path.isfile('caption.txt'):
    print('❌ Caption not generated')
    sys.exit(1)

print('✓ Outputs verified')
