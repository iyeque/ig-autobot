#!/usr/bin/env python3
import os
import json
import sys
import argparse
import shutil

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--bundle", help="Explicit bundle file to use")
    args = parser.parse_args()
    platform = args.platform.lower()

    # 1. Locate the Bundle
    if args.bundle:
        if os.path.exists(args.bundle):
            found_bundle = args.bundle
            # If the bundle is in forwilma/ or named wilma_bundle, out_dir should be forwilma
            if "wilma" in args.bundle:
                target_dir = "forwilma"
            else:
                target_dir = "."
        else:
            print(f"❌ Error: Specified bundle '{args.bundle}' not found.")
            sys.exit(1)
    else:
        # Auto-discovery logic (Legacy/Default)
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

    print(f"📖 Using bundle: {found_bundle} (Target Dir: {target_dir})")

    with open(found_bundle, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    if platform not in bundle:
        print(f"❌ Error: Caption for platform '{platform}' not found in bundle.")
        sys.exit(1)

    # 2. Write Caption
    # Ensure target directory exists
    if target_dir != "." and not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        
    caption_out = os.path.join(target_dir, "caption.txt")
    with open(caption_out, "w", encoding="utf-8") as f:
        f.write(bundle[platform])
    print(f"✓ Prepared {caption_out} for {platform.upper()}")

    # 3. Ensure media is in the right place
    # If we are in 'forwilma' dir mode, we need the latest image from forwilma/images
    if target_dir == "forwilma":
        img_dir = os.path.join("forwilma", "images")
        if os.path.exists(img_dir):
            images = sorted([f for f in os.listdir(img_dir) if f.startswith("day") or f.startswith("post")], reverse=True)
            if images:
                src = os.path.join(img_dir, images[0])
                dst = os.path.join("forwilma", "output.jpg")
                shutil.copy(src, dst)
                print(f"✓ Copied latest Wilma image {src} to {dst}")
        else:
            # Fallback: if output.jpg exists in root but we want forwilma, copy it?
            # No, Wilma bot should have generated its own output.jpg in forwilma/
            pass
    else:
        # For the trilogy, we expect output.jpg and reel.mp4 in the root
        pass

if __name__ == "__main__":
    prepare()
