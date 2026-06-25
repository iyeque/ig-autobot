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


def _strip_bluesky_cta(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    cleaned = []
    cta_markers = [
        "Want to read more?... check out my LinkedIn",
        "check out my LinkedIn",
        "Read the rest on LinkedIn",
        "Read more on LinkedIn",
        "Continue reading on LinkedIn",
        "Full post on LinkedIn",
        "Follow for more",
        "👉 Follow",
        "Save this",
        "Share this",
        "Comment below",
    ]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        is_cta = False
        for marker in cta_markers:
            if marker.lower() in stripped.lower():
                is_cta = True
                break
        if not is_cta:
            cleaned.append(line)
    text = "\n".join(cleaned).strip()
    while text.endswith("\n\n\n"):
        text = text[:-1]
    return text


def _ensure_bluesky_linkedin_cta(text: str) -> str:
    text = _strip_bluesky_cta(text)
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
