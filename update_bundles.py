#!/usr/bin/env python3
import os
import json
import re


def _strip_hashtags_from_threads(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Remove inline hashtag blobs at end of line
        line = re.sub(r"\s+#\w+(\s+#\w+)*\s*$", "", line).rstrip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def _ensure_bluesky_linkedin_cta(text: str) -> str:
    cta = "Want to read more?... check out my LinkedIn"
    if cta.lower() in text.lower():
        return text
    return text.rstrip() + "\n\n" + cta


def _fix_bundle_captions(bundle: dict) -> dict:
    captions = bundle.get("captions")
    if not isinstance(captions, dict):
        return bundle

    if "threads" in captions:
        captions["threads"] = _strip_hashtags_from_threads(str(captions["threads"]))

    if "bluesky" in captions:
        captions["bluesky"] = _ensure_bluesky_linkedin_cta(str(captions["bluesky"]))

    bundle["captions"] = captions
    return bundle


def update_state_bundles(state_path):
    if not os.path.exists(state_path):
        print(f"{state_path} not found")
        return

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    queue = state.get("content_queue", [])
    for i, bundle in enumerate(queue):
        if isinstance(bundle, dict):
            queue[i] = _fix_bundle_captions(bundle)

    state["content_queue"] = queue

    active = state.get("active_bundle")
    if isinstance(active, dict):
        state["active_bundle"] = _fix_bundle_captions(active)

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)
    print(f"Updated {state_path}")


if __name__ == "__main__":
    update_state_bundles("state.json")
    update_state_bundles("forwilma/state.json")
