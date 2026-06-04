#!/usr/bin/env python3
import os
import json
import sys
import argparse
import shutil

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    args = parser.parse_args()
    platform = args.platform.lower()

    # 1. Locate the Bundle
    # We check root first, then forwilma/ subdirectory
    search_paths = [
        {"bundle": "captions_bundle.json", "out_dir": "."},
        {"bundle": "wilma_bundle.json", "out_dir": "."},
        {"bundle": "forwilma/wilma_bundle.json", "out_dir": "forwilma"}
    ]
    
    found_bundle = None
    target_dir = "."

    for p in search_paths:
        if os.path.exists(p["bundle"]):
            found_bundle = p["bundle"]
            target_dir = p["out_dir"]
            break

    if not found_bundle:
        print(f"❌ Error: No caption bundle found. Did the generation job run?")
        sys.exit(1)

    print(f"📖 Using bundle: {found_bundle}")

    with open(found_bundle, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    if platform not in bundle:
        print(f"❌ Error: Caption for platform '{platform}' not found in bundle.")
        sys.exit(1)

    # 2. Write Caption
    caption_out = os.path.join(target_dir, "caption.txt")
    with open(caption_out, "w", encoding="utf-8") as f:
        f.write(bundle[platform])
    print(f"✓ Prepared {caption_out} for {platform.upper()}")

    # 3. Ensure media is in the right place
    # If we are in 'forwilma' dir mode, we need the latest image from forwilma/images
    if target_dir == "forwilma":
        img_dir = os.path.join("forwilma", "images")
        if os.path.exists(img_dir):
            images = sorted([f for f in os.listdir(img_dir) if f.startswith("day")], reverse=True)
            if images:
                src = os.path.join(img_dir, images[0])
                dst = os.path.join("forwilma", "output.jpg")
                shutil.copy(src, dst)
                print(f"✓ Copied latest Wilma image to {dst}")

if __name__ == "__main__":
    prepare()
