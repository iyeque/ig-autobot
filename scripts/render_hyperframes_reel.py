#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE = REPO_ROOT / "state.json"
RENDER_WRAPPER_DIR = REPO_ROOT / "hyperframes" / "compositions"
REELS_DIR = REPO_ROOT / "reels"


def run(cmd: list[str] | str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)


def choose_motion(pillar: str) -> str:
    mapping = {
        "micro_philosophy": "typewriter",
        "nature_metaphor": "kenburns",
        "systems_psychology": "parallax",
        "author_voice": "fade_in",
        "quote": "slide_up",
        "personalgrowth": "fade_in",
    }
    return mapping.get((pillar or "").lower(), "kenburns")


def generate_composition(post_id: str, audio: str | None, caption_rail: bool, duration_s: int) -> Path:
    from generate_hyperframes_reel import generate_composition as _generate_composition, read_state, ensure_jinja2

    if not ensure_jinja2():
        raise RuntimeError("Jinja2 is required. Install it with: pip install jinja2")

    state = read_state(DEFAULT_STATE)
    active = state.get("active_bundle") or {}
    if str(active.get("post_id")) != str(post_id):
        candidate = next((b for b in state.get("content_queue", []) if str(b.get("post_id")) == str(post_id)), None)
        if candidate:
            active = candidate
    pillar = active.get("pillar") or ""
    topic = active.get("topic") or ""
    image_rel = active.get("image") or active.get("story") or ""
    image_path = REPO_ROOT / image_rel if image_rel else None
    if not image_path or not image_path.exists():
        raise SystemExit(f"Image not found for bundle {post_id}: {image_rel}")
    captions = active.get("captions") or {}
    caption_text = captions.get("instagram") or captions.get("linkedin") or ""

    out = _generate_composition(
        post_id=post_id,
        image_rel_path=f"../assets/bundle-{post_id}.jpg",
        caption_text=caption_text,
        pillar=pillar,
        topic=topic,
        duration_s=duration_s,
        audio_path=audio,
        show_caption_rail=caption_rail,
        caption_excerpt=" ".join(caption_text.split())[:180],
    )
    return out


def render_with_hyperframes(composition_dir: Path, output_path: Path) -> bool:
    cmd = f'cd "{composition_dir}" && npx hyperframes render --output "{output_path}"'
    proc = run(cmd)
    return proc.returncode == 0 and output_path.exists()


def mix_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    cmd = (
        f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
        f'-c:v copy -c:a aac -b:a 192k -shortest "{output_path}"'
    )
    proc = run(cmd)
    return proc.returncode == 0 and output_path.exists()


def parse_args():
    parser = argparse.ArgumentParser(description="Render an ig-autobot HyperFrames reel end-to-end")
    parser.add_argument("--post_id", required=True, help="Bundle post_id to render")
    parser.add_argument("--duration_s", type=int, default=10, help="Target duration in seconds")
    parser.add_argument("--audio", required=False, help="Optional audio file path to mix into the reel")
    parser.add_argument("--caption_rail", action="store_true", help="Add a permanent caption rail in the composition")
    parser.add_argument("--out", required=False, help="Output MP4 path, default: reels/reel_<post_id>_hyperframes.mp4")
    parser.add_argument("--state_path", default=str(DEFAULT_STATE), help="Path to state.json")
    return parser.parse_args()


def main():
    args = parse_args()
    post_id = str(args.post_id)
    default_out = REELS_DIR / f"reel_{post_id}_hyperframes.mp4"
    output_path = Path(args.out) if args.out else default_out
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🎬 Rendering HyperFrames reel for bundle {post_id}")
    print(f"   Output : {output_path}")

    try:
        composition_path = generate_composition(
            post_id=post_id,
            audio=args.audio,
            caption_rail=args.caption_rail,
            duration_s=args.duration_s,
        )
    except Exception as e:
        raise SystemExit(f"Composition generation failed: {e}")

    composition_dir = composition_path.parent
    temp_video = composition_dir / f"bundle-{post_id}.mp4"

    print(f"   Composition: {composition_path}")
    print(f"   Rendering via HyperFrames CLI...")
    if not render_with_hyperframes(composition_dir, temp_video):
        raise SystemExit("HyperFrames render failed. Check the composition in HyperFrames preview.")

    final_path = output_path
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"⚠ Audio not found: {audio_path}. Using video-only output.")
            final_path = output_path
            if temp_video != output_path:
                temp_video.replace(output_path)
        else:
            mixed = composition_dir / f"bundle-{post_id}_audio.mp4"
            if mix_audio(temp_video, audio_path, mixed):
                mixed.replace(output_path)
                final_path = output_path
            else:
                print("⚠ Audio mix failed. Using video-only output.")
                if temp_video != output_path:
                    temp_video.replace(output_path)

    print(f"✅ Reel ready: {final_path}")
    print(f"   Size    : {final_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
