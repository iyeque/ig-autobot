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
            print("❌ No active bundle or queue.")
            sys.exit(0)
        active = queue[0]
        state["active_bundle"] = active
        state["content_queue"] = queue[1:]
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    post_id = active.get("post_id", "unknown")
    topic = active.get("topic") or active.get("caption_prompt") or "Content"
    pillar = active.get("pillar") or active.get("type") or "General"
    topic_clean = topic.strip().rstrip(".")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Generating carousel for {post_id}: {topic_clean[:60]}")

    if args.wilma:
        slides = generate_wilma_carousel(
            pillar, topic_clean, timestamp,
            footer_text=args.footer,
        )
    else:
        slides = generate_carousel(
            pillar, topic_clean, timestamp,
            footer_text=args.footer,
        )

    if not slides:
        print("⚠ Carousel generation returned no slides.")
        sys.exit(0)

    state_dir = state_path.parent if state_path.parent != Path(".") else Path(".")

    # Build structured carousel data: paths + per-slide captions + post caption
    style = "wilma" if args.wilma else "dark"
    narrative = _build_carousel_narrative(pillar, topic_clean, style=style)

    rel_paths = [str(Path(p).relative_to(state_dir) if Path(p).is_absolute() else p) for p in slides]
    per_slide = [
        {"path": p, "caption": (narrative.get("slides") or [""] * 5)[i]}
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
