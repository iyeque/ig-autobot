#!/usr/bin/env python3
"""
Generate deterministic carousel slides from the active bundle's topic.
Writes carousel.json so the publisher picks it up on carousel days.
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bot import generate_carousel, generate_wilma_carousel, _build_carousel_narrative

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_path", default="state.json")
    parser.add_argument("--footer", default="M.W.E. WIGMAN | THE NINE STITCHES")
    parser.add_argument("--wilma", action="store_true", help="Use Wilma carousel style")
    args = parser.parse_args()

    state_path = Path(args.state_path)
    if not state_path.exists():
        print(f"❌ State not found: {state_path}")
        sys.exit(1)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    active = state.get("active_bundle")
    if not active:
        queue = state.get("content_queue", [])
        if not queue:
            # Fallback: most recently posted bundle. The carousel must not
            # depend on gen/publish cycle timing — with daily generation the
            # queue is usually empty by carousel day (Mon/Wed/Fri), which
            # silently no-oped the whole flow. Every narrative is generated
            # fresh, so the same topic still yields a new carousel.
            pbc = state.get("posted_bundle_content") or {}
            if pbc:
                def _num(key):
                    digits = "".join(ch for ch in str(key) if ch.isdigit())
                    return int(digits) if digits else 0
                latest_key = max(pbc.keys(), key=_num)
                latest = pbc[latest_key] or {}
                active = {
                    "post_id": latest.get("post_id") or latest_key,
                    "topic": latest.get("topic") or "",
                    "pillar": latest.get("pillar") or "General",
                    "master_reflection": latest.get("master_reflection") or "",
                    "captions": latest.get("captions") or {},
                }
                print(f"↩ Queue empty — building carousel from most recent post {latest_key}: {str(active['topic'])[:50]}")
            else:
                print("❌ No active bundle, queue, or posted content.")
                sys.exit(0)
        else:
            active = queue[0]
            state["active_bundle"] = active
            state["content_queue"] = queue[1:]
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    post_id = active.get("post_id", "unknown")

    # Guard: skip if carousel.json already exists for this bundle
    state_dir = state_path.parent if state_path.parent != Path(".") else Path(".")
    carousel_json = state_dir / "carousel.json"
    if carousel_json.exists():
        print(f"⏭️ carousel.json already exists for {post_id}. Skipping generation.")
        sys.exit(0)

    # Derive topic from the bundle's caption text if topic/caption_prompt
    # fields are missing — this ensures slides match the actual post content.
    topic = active.get("topic") or active.get("caption_prompt")
    caption_text = active.get("captions", {}).get("linkedin") or active.get("caption", "")
    if not topic and caption_text:
        # Use the first meaningful line of the caption as the topic
        first_lines = []
        for line in caption_text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                first_lines.append(line)
            if first_lines:
                break
        topic = " ".join(first_lines[:2]) if first_lines else "Content"
    if not topic:
        topic = "Content"
    pillar = active.get("pillar") or active.get("type") or "General"
    topic_clean = topic.strip().rstrip(".")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Generating carousel for {post_id}: {topic_clean[:60]}")

    # Build the narrative ONCE. generate_carousel / generate_wilma_carousel
    # each call _build_carousel_narrative internally, and calling it again
    # here produced two separate LLM outputs — the rendered slides and the
    # carousel.json captions disagreed (slide 1 said one thing, the caption
    # said another). One narrative, passed in, drives both.
    style = "wilma" if args.wilma else "dark"
    narrative = _build_carousel_narrative(pillar, topic_clean, style=style)
    slide_texts = list(narrative.get("slides") or [])

    if args.wilma:
        slides = generate_wilma_carousel(
            pillar, topic_clean, timestamp,
            footer_text=args.footer,
            slides=slide_texts,
        )
    else:
        slides = generate_carousel(
            pillar, topic_clean, timestamp,
            footer_text=args.footer,
            slides=slide_texts,
        )

    if not slides:
        print("⚠ Carousel generation returned no slides.")
        sys.exit(0)

    state_dir = state_path.parent if state_path.parent != Path(".") else Path(".")

    # Build structured carousel data: paths + per-slide captions + post caption
    rel_paths = [str(Path(p).relative_to(state_dir) if Path(p).is_absolute() else p) for p in slides]
    per_slide = [
        {"path": p, "caption": slide_texts[i] if i < len(slide_texts) else ""}
        for i, p in enumerate(rel_paths)
    ]
    carousel_data = {
        "post_id": post_id,
        "style": style,
        "post_caption": narrative.get("post_caption", ""),
        "slides": per_slide,
    }
    carousel_json = state_dir / "carousel.json"
    carousel_json.write_text(json.dumps(carousel_data, indent=2), encoding="utf-8")
    print(f"✓ Wrote {len(slides)} slides to {carousel_json}")
    print("Slides:", [s["path"] for s in per_slide])

if __name__ == "__main__":
    main()
