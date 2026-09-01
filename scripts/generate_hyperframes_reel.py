#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE = REPO_ROOT / "state.json"
COMPOSITIONS_DIR = REPO_ROOT / "hyperframes" / "compositions"
TEMPLATE_PATH = COMPOSITIONS_DIR / "template.html"

PILLAR_MOTION = {
    "micro_philosophy": "typewriter",
    "nature_metaphor": "kenburns",
    "systems_psychology": "parallax",
    "author_voice": "fade_in",
    "quote": "slide_up",
    "personalgrowth": "fade_in",
}


def ensure_jinja2() -> bool:
    try:
        import jinja2  # noqa: F401
        return True
    except Exception:
        return False


def split_beats(text: str, max_words: int = 8):
    words = text.split()
    beats = []
    for i in range(0, len(words), max_words):
        beats.append(" ".join(words[i : i + max_words]))
    return beats or [text]


def sanitize_identifier(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value)
    return value.strip("-").lower() or "composition"


def read_state(state_path: Path):
    if not state_path.exists():
        raise FileNotFoundError(f"state.json not found: {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_motion(pillar: str):
    key = (pillar or "").lower()
    return PILLAR_MOTION.get(key, "kenburns")


def resolve_audio_path(raw: str | None) -> str | None:
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return str(path.resolve())
    return None


def generate_composition(
    post_id: str,
    image_rel_path: str,
    caption_text: str,
    pillar: str,
    topic: str,
    duration_s: int = 10,
    audio_path: str | None = None,
    show_caption_rail: bool = False,
    caption_excerpt: str = "",
) -> Path:
    if not ensure_jinja2():
        raise RuntimeError("Jinja2 is required. Install it with: pip install jinja2")

    from jinja2 import Template

    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))

    motion = choose_motion(pillar)
    safe_id = sanitize_identifier(str(post_id))
    composition_name = f"bundle-{safe_id}"
    output_path = COMPOSITIONS_DIR / f"{composition_name}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paragraphs = [p.strip() for p in caption_text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [topic or "Insight"]

    beats = []
    for p in paragraphs[:4]:
        beats.extend(split_beats(p, max_words=8))
    beats = beats[:6]
    if not beats:
        beats = [topic or "Insight"]

    total_beats = len(beats)
    beat_slot = max(1.2, (duration_s - 2.5) / max(total_beats, 1))

    motion_css_map = {
        "kenburns": "transform-origin: 50% 50%; will-change: transform;",
        "slide_up": "transform-origin: 50% 100%; will-change: transform, opacity;",
        "fade_in": "will-change: opacity;",
        "typewriter": "transform-origin: 50% 50%; will-change: transform;",
        "parallax": "transform-origin: 50% 50%; will-change: transform;",
    }
    motion_gsap_map = {
        "kenburns": f'tl.to("#bg", {{ scale: 1.08, duration: {duration_s}, ease: "none" }}, 0)',
        "slide_up": f'tl.from("#bg", {{ y: 40, opacity: 0.6, duration: {duration_s}, ease: "none" }}, 0)',
        "fade_in": f'tl.to("#bg", {{ opacity: 1, duration: {duration_s}, ease: "none" }}, 0)',
        "typewriter": f'tl.to("#bg", {{ scale: 1.02, duration: {duration_s}, ease: "none" }}, 0)',
        "parallax": f'tl.to("#bg", {{ scale: 1.05, duration: {duration_s}, ease: "none" }}, 0)',
    }

    motion_css = motion_css_map.get(motion, motion_css_map["kenburns"])
    motion_gsap = motion_gsap_map.get(motion, motion_gsap_map["kenburns"])

    beat_models = []
    timeline_parts = []
    cursor = 0.8
    top_start = 8
    top_end = 72
    top_range = top_end - top_start
    spacing = top_range / max(len(beats) - 1, 1)
    for idx, text in enumerate(beats):
        top_pct = top_start + idx * spacing if len(beats) > 1 else (top_start + top_end) / 2
        font_size = 44 if idx == 0 else 30
        font_weight = "700" if idx == 0 else "400"
        beat_models.append(
            {
                "top": round(top_pct, 1),
                "font_size": font_size,
                "font_weight": font_weight,
                "text": text,
            }
        )
        timeline_parts.append(
            f'tl.to("#beat{idx+1}", {{ opacity: 1, y: 0, duration: 0.9, ease: "power2.out" }}, {cursor})'
        )
        cursor += beat_slot

    highlight_duration = min(1.2, beat_slot * 0.9)
    highlight_start = 0.2
    outro_start = max(cursor, duration_s - 1.8)
    logo_start = outro_start

    rendered = template.render(
        composition_name=composition_name,
        image_rel_path=image_rel_path,
        duration_s=duration_s,
        motion=motion,
        motion_css=motion_css,
        motion_gsap=motion_gsap,
        highlight_duration=f"{highlight_duration:.2f}",
        highlight_start=f"{highlight_start:.1f}",
        logo_start=f"{logo_start:.1f}",
        beats=beat_models,
        timeline_parts=timeline_parts,
        show_caption_rail=bool(show_caption_rail),
        caption_excerpt=(caption_excerpt or "").strip(),
        audio_path=(audio_path or "").strip(),
    )

    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a HyperFrames composition from an ig-autobot bundle")
    parser.add_argument("--post_id", required=True, help="Bundle post_id to use")
    parser.add_argument("--image", required=False, help="Relative image path, e.g. images/post_3003_unique.jpg")
    parser.add_argument("--caption", required=False, help="Caption text to animate")
    parser.add_argument("--pillar", required=False, help="Pillar name, e.g. micro_philosophy")
    parser.add_argument("--topic", required=False, help="Topic / hook line")
    parser.add_argument("--duration_s", type=int, default=10, help="Target duration in seconds")
    parser.add_argument("--state_path", default=str(DEFAULT_STATE), help="Path to state.json")
    parser.add_argument("--audio", required=False, help="Optional audio path for background music")
    parser.add_argument("--caption_rail", action="store_true", help="Add a permanent caption rail at the bottom")
    return parser.parse_args()


def main():
    args = parse_args()
    state = read_state(Path(args.state_path))

    active = state.get("active_bundle") or {}

    # Allow int active_bundle values as well as dicts
    if isinstance(active, int):
        active = None

    if active is None or not isinstance(active, dict):
        if str(active) == str(args.post_id):
            active = {"post_id": active}
        else:
            def _queue_candidate(queue, post_id):
                for b in queue:
                    if isinstance(b, dict):
                        if str(b.get("post_id")) == str(post_id):
                            return b
                    elif str(b) == str(post_id):
                        return {"post_id": b}
                return None

            candidate = _queue_candidate(state.get("content_queue", []), args.post_id)
            if candidate:
                active = candidate
    elif str(active.get("post_id")) != str(args.post_id):
        def _queue_candidate(queue, post_id):
            for b in queue:
                if isinstance(b, dict):
                    if str(b.get("post_id")) == str(post_id):
                        return b
                elif str(b) == str(post_id):
                    return {"post_id": b}
            return None

        candidate = _queue_candidate(state.get("content_queue", []), args.post_id)
        if candidate:
            active = candidate
    if not active:
        raise SystemExit(f"post_id {args.post_id} not found in state.json")

    pillar = args.pillar or active.get("pillar") or ""
    topic = args.topic or active.get("topic") or ""
    image_rel = args.image or active.get("image") or active.get("story") or ""
    image_path = REPO_ROOT / image_rel if image_rel else None
    if not image_path or not image_path.exists():
        raise SystemExit(f"Image not found for bundle {args.post_id}: {image_rel}")

    captions = active.get("captions") or {}
    caption_text = args.caption or captions.get("instagram") or captions.get("linkedin") or ""
    if not caption_text:
        raise SystemExit(f"No caption available for bundle {args.post_id}")

    paragraphs = [p.strip() for p in caption_text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [topic]

    target_image_rel = f"../assets/bundle-{args.post_id}.jpg"
    target_image_path = COMPOSITIONS_DIR / f"bundle-{args.post_id}.jpg"
    if not target_image_path.exists():
        target_image_path.write_bytes(image_path.read_bytes())

    audio_path = resolve_audio_path(args.audio)
    caption_excerpt = " ".join(caption_text.split())
    if len(caption_excerpt) > 180:
        caption_excerpt = caption_excerpt[:177].rstrip() + "..."

    out = generate_composition(
        post_id=args.post_id,
        image_rel_path=target_image_rel,
        caption_text=caption_text,
        pillar=pillar,
        topic=topic,
        duration_s=args.duration_s,
        audio_path=audio_path,
        show_caption_rail=args.caption_rail,
        caption_excerpt=caption_excerpt,
    )
    print(f"✅ Wrote composition: {out}")
    print(f"   Image asset : {target_image_path}")
    print(f"   Motion preset: {choose_motion(pillar)}")
    if audio_path:
        print(f"   Audio input : {audio_path}")


if __name__ == "__main__":
    main()
