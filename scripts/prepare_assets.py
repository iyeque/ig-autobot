#!/usr/bin/env python3
import os
import json
import sys
import argparse

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    args = parser.parse_args()
    platform = args.platform.lower()

    # Support both main and Wilma bundles
    bundle_path = "wilma_bundle.json" if os.path.exists("wilma_bundle.json") else "captions_bundle.json"
    
    if not os.path.exists(bundle_path):
        print(f"❌ Error: {bundle_path} not found. Did the generation job run?")
        sys.exit(1)

    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    if platform not in bundle:
        print(f"❌ Error: Caption for platform '{platform}' not found in bundle.")
        sys.exit(1)

    # Write the platform-specific caption to the legacy location
    with open("caption.txt", "w", encoding="utf-8") as f:
        f.write(bundle[platform])

    print(f"✓ Prepared caption.txt for {platform.upper()}")

if __name__ == "__main__":
    prepare()
