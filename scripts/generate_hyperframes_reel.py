#!/usr/bin/env python3
"""Generate an engagement-optimized HyperFrames reel from an ig-autobot bundle.

Three templates map to content pillars:
- hook-machine:      micro_philosophy, quote            (fast cuts, hook frame, loopable)
- bold-bright:      personalgrowth, author_voice        (bright colors, punchy, end CTA)
- educational-save: nature_metaphor, systems_psychology (save-bait, badges, highlights)
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE = REPO_ROOT / "state.json"
COMPOSITIONS_DIR = REPO_ROOT / "hyperframes" / "compositions"
TEMPLATES_DIR = COMPOSITIONS_DIR / "templates"

# Map pillar → template name
PILLAR_TEMPLATE = {
    "micro_philosophy": "hook-machine",
    "quote": "hook-machine",
    "author_voice": "bold-bright",
    "personalgrowth": "bold-bright",
    "nature_metaphor": "educational-save",
    "systems_psychology": "educational-save",
}

TEMPLATE_FILES = {
    "hook-machine": TEMPLATES_DIR / "hook-machine.html",
    "bold-bright": TEMPLATES_DIR / "bold-bright.html",
    "educational-save": TEMPLATES_DIR / "educational-save.html",
}

DEFAULT_BRAND_NAME = "THE NINE STITCHES"
DEFAULT_BRAND_HANDLE = "@theninestitches"


def ensure_jinja2() -> bool:
    try:
        import jinja2  # noqa: F401
        return True
    except Exception:
        return False


def sanitize_identifier(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value)
    return value.strip("-").lower() or "composition"


def read_state(state_path: Path):
    if not state_path.exists():
        raise FileNotFoundError(f"state.json not found: {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_template(pillar: str, override: str | None = None) -> str:
    if override:
        return override
    key = (pillar or "").lower()
    return PILLAR_TEMPLATE.get(key, "hook-machine")


def resolve_audio_path(raw: str | None) -> str | None:
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return str(path.resolve())
    return None


def split_phrases(text: str, max_words: int = 6) -> list:
    words = text.split()
    phrases = []
    for i in range(0, len(words), max_words):
        phrase = " ".join(words[i : i + max_words])
        if phrase:
            phrases.append(phrase)
    return phrases or [text]


def _prepare_hook_machine(post_id, image_rel_path, caption_text, topic, duration_s,
                          brand_name, brand_handle, audio_path):
    """Fast cuts, hook frame first, pattern interrupts every 2 beats, loopable."""
    phrases = split_phrases(caption_text, max_words=5)[:6]
    if not phrases:
        phrases = [topic or "Insight"]

    beats = []
    timeline_parts = []
    flashes = []
    cursor = 1.5  # hook frame occupies 0–1.5s
    total = len(phrases)
    slot = max(1.5, (duration_s - 3.5) / max(total, 1))

    for idx, text in enumerate(phrases):
        top = 25 + idx * 10
        beats.append({
            "top": min(top, 75),
            "font_size": 56 if idx == 0 else 40,
            "font_weight": "900" if idx < 2 else "700",
            "text": text.upper(),
        })
        timeline_parts.append(
            f'tl.to("#beat{idx+1}", {{ opacity: 1, y: 0, scale: 1, '
            f'duration: 0.25, ease: "back.out(1.7)" }}, {round(cursor, 2)})'
        )
        if idx > 0 and idx % 2 == 0:
            flashes.append({"time": round(cursor - 0.1, 2)})
        cursor += slot

    return {
        "composition_name": f"bundle-{sanitize_identifier(str(post_id))}",
        "image_rel_path": image_rel_path,
        "duration_s": duration_s,
        "hook_text": (topic or phrases[0] or "HOOK").upper(),
        "beats": beats,
        "flashes": flashes,
        "timeline_parts": timeline_parts,
        "end_start": round(duration_s - 2.0, 2),
        "brand_name": brand_name,
        "audio_path": audio_path or "",
    }


def _prepare_bold_bright(post_id, image_rel_path, caption_text, topic, duration_s,
                         brand_name, brand_handle, audio_path):
    """Bright, punchy, rapid slide cuts, end CTA."""
    phrases = split_phrases(caption_text, max_words=6)[:5]
    if not phrases:
        phrases = [topic or "Bold insight"]

    slides = []
    for i, text in enumerate(phrases):
        slide = {"num": str(i + 1), "headline": text.upper()}
        if i == len(phrases) - 1:
            slide["sub"] = f"Follow {brand_handle} for more"
        slides.append(slide)

    slide_timeline = []
    end_start = duration_s - 2.0
    slot = (end_start - 1.0) / max(len(slides), 1)
    for i in range(len(slides)):
        start = 0.8 + i * slot
        end = start + slot - 0.1
        slide_timeline.append({"idx": i + 1, "start": round(start, 2), "end": round(end, 2)})

    return {
        "composition_name": f"bundle-{sanitize_identifier(str(post_id))}",
        "image_rel_path": image_rel_path,
        "duration_s": duration_s,
        "slides": slides,
        "slide_timeline": slide_timeline,
        "end_headline": "FOLLOW FOR MORE",
        "end_sub": "Save this for later",
        "brand_handle": brand_handle,
        "brand_name": brand_name,
        "audio_path": audio_path or "",
    }


def _prepare_educational_save(post_id, image_rel_path, caption_text, topic, duration_s,
                              brand_name, brand_handle, audio_path):
    """Educational save-bait: badges, highlighted keywords, progress bar, save CTA."""
    phrases = split_phrases(caption_text, max_words=8)[:4]
    if not phrases:
        phrases = [topic or "Did you know?"]

    badges_pool = ["DIGITAL WELLNESS", "SCREEN TIME", "MENTAL HEALTH", "PRODUCTIVITY"]
    frames = []
    for i, text in enumerate(phrases):
        frame = {"headline": text}
        if i == 0:
            frame["badges"] = badges_pool[:3]
        if i == 1:
            frame["sub"] = "Save this for your next deep work session"
        elif i == 2:
            frame["stat"] = "4+"
            frame["stat_label"] = "hours per day on screens"
        frames.append(frame)

    frame_timeline = []
    end_start = duration_s - 2.0
    slot = (end_start - 1.0) / max(len(frames), 1)
    for i in range(len(frames)):
        start = 0.8 + i * slot
        end = start + slot - 0.1
        frame_timeline.append({"idx": i + 1, "start": round(start, 2), "end": round(end, 2)})

    return {
        "composition_name": f"bundle-{sanitize_identifier(str(post_id))}",
        "image_rel_path": image_rel_path,
        "duration_s": duration_s,
        "frames": frames,
        "frame_timeline": frame_timeline,
        "brand_name": brand_name,
        "brand_handle": brand_handle,
        "audio_path": audio_path or "",
    }


def generate_composition(
    post_id: str,
    image_rel_path: str,
    caption_text: str,
    pillar: str,
    topic: str,
    duration_s: int = 9,
    audio_path: str | None = None,
    template_override: str | None = None,
    brand_name: str = DEFAULT_BRAND_NAME,
    brand_handle: str = DEFAULT_BRAND_HANDLE,
) -> Path:
    if not ensure_jinja2():
        raise RuntimeError("Jinja2 is required. Install it with: pip install jinja2")

    from jinja2 import Template

    template_name = choose_template(pillar, template_override)
    template_path = TEMPLATE_FILES.get(template_name, TEMPLATES_FILES["hook-machine"])
    template = Template(template_path.read_text(encoding="utf-8"))

    safe_id = sanitize_identifier(str(post_id))
    composition_name = f"bundle-{safe_id}"
    output_path = COMPOSITIONS_DIR / f"{composition_name}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if template_name == "bold-bright":
        data = _prepare_bold_bright(post_id, image_rel_path, caption_text, topic,
                                   duration_s, brand_name, brand_handle, audio_path)
    elif template_name == "educational-save":
        data = _prepare_educational_save(post_id, image_rel_path, caption_text, topic,
                                         duration_s, brand_name, brand_handle, audio_path)
    else:
        data = _prepare_hook_machine(post_id, image_rel_path, caption_text, topic,
                                    duration_s, brand_name, brand_handle, audio_path)

    rendered = template.render(**data)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an engagement-optimized HyperFrames reel")
    parser.add_argument("--post_id", required=True, help="Bundle post_id to use")
    parser.add_argument("--image", required=False, help="Relative image path")
    parser.add_argument("--caption", required=False, help="Caption text to animate")
    parser.add_argument("--pillar", required=False, help="Pillar name")
    parser.add_argument("--topic", required=False, help="Topic / hook line")
    parser.add_argument("--duration_s", type=int, default=9, help="Target duration (6-15s)")
    parser.add_argument("--state_path", default=str(DEFAULT_STATE), help="Path to state.json")
    parser.add_argument("--audio", required=False, help="Optional audio path")
    parser.add_argument("--template", required=False, help="Override template")
    parser.add_argument("--brand_name", default=DEFAULT_BRAND_NAME, help="Brand name for end card")
    parser.add_argument("--brand_handle", default=DEFAULT_BRAND_HANDLE, help="Brand handle/social")
    return parser.parse_args()


def main():
    args = parse_args()
    state = read_state(Path(args.state_path))

    active = state.get("active_bundle") or {}

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

    target_image_rel = f"bundle-{args.post_id}.jpg"
    target_image_path = COMPOSITIONS_DIR / f"bundle-{args.post_id}.jpg"
    if not target_image_path.exists():
        target_image_path.write_bytes(image_path.read_bytes())

    audio_path = resolve_audio_path(args.audio)
    template_used = choose_template(pillar, args.template)

    out = generate_composition(
        post_id=args.post_id,
        image_rel_path=target_image_rel,
        caption_text=caption_text,
        pillar=pillar,
        topic=topic,
        duration_s=args.duration_s,
        audio_path=audio_path,
        template_override=args.template,
        brand_name=args.brand_name,
        brand_handle=args.brand_handle,
    )

    print(f"✅ Wrote composition: {out}")
    print(f"   Image asset : {target_image_path}")
    print(f"   Template    : {template_used}")
    if audio_path:
        print(f"   Audio input : {audio_path}")


if __name__ == "__main__":
    main()
