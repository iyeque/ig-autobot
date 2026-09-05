#!/usr/bin/env python3
import os
import json
import sys
import argparse
import shutil
import subprocess
import requests
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared_utils import load_state, save_state, is_platform_posted, required_platforms, clean_caption_formatting, is_bundle_consumed_for_platform


def upload_to_catbox(local_path: str) -> str | None:
    """Upload an image to catbox.moe and return the public URL."""
    try:
        with open(local_path, "rb") as f:
            r = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": (os.path.basename(local_path), f, "image/jpeg")},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"},
                timeout=60,
            )
        r.raise_for_status()
        url = r.text.strip()
        if url.startswith("http"):
            print(f"Uploaded {os.path.basename(local_path)} -> {url}")
            return url
        print(f"Unexpected catbox response for {local_path}: {url}")
    except Exception as exc:
        print(f"Warning: catbox upload failed for {local_path}: {exc}")
    return None


def _bundle_consumed_for_platform(bundle: dict, platform: str, state: dict, state_path: str) -> bool:
    if not isinstance(bundle, dict):
        return False

    if is_bundle_consumed_for_platform(bundle, platform, state_path, state):
        return True

    post_id = bundle.get("post_id")
    bundle_posted = platform in (bundle.get("platforms_posted") or [])
    if bundle_posted:
        return True

    platform_history = state.get("platform_posted_bundles", {})
    if post_id and post_id in platform_history.get(platform, []):
        return True

    required = required_platforms(state_path)
    posted = bundle.get("platforms_posted", [])
    if not required:
        return False

    return all(
        p in posted or (post_id and post_id in platform_history.get(p, []))
        for p in required
    )


def _select_next_bundle_for_platform(state: dict, platform: str, state_path: str):
    active = state.get("active_bundle")
    if isinstance(active, dict) and not _bundle_consumed_for_platform(active, platform, state, state_path):
        return active, list(state.get("content_queue", []))

    queue = list(state.get("content_queue", []))
    chosen_index = None
    for index, candidate in enumerate(queue):
        if not _bundle_consumed_for_platform(candidate, platform, state, state_path):
            chosen_index = index
            break

    if chosen_index is not None:
        chosen = queue.pop(chosen_index)
        return chosen, queue

    return None, queue


def _platform_policy(platform: str) -> dict:
    platform = platform.lower()
    if platform in {"instagram", "threads", "bluesky", "linkedin", "youtube"}:
        return {
            "use_static_image": True,
            "use_reel": platform in {"instagram", "youtube"},
            "use_carousel": platform == "instagram",
            "caption_style": "short" if platform in {"threads", "bluesky", "youtube"} else "long",
            "cta_mode": "linkedin" if platform == "bluesky" else "none",
        }
    return {
        "use_static_image": True,
        "use_reel": False,
        "use_carousel": False,
        "caption_style": "long",
        "cta_mode": "none",
    }


def _apply_platform_tailoring(caption: str, platform: str) -> str:
    platform = platform.lower()
    text = (caption or "").strip()
    if not text:
        return text

    if platform == "threads":
        if len(text) > 420:
            text = text[:417].rstrip() + "..."
        return text

    if platform == "bluesky":
        suffix = "\n\nWant to read more?... check out my LinkedIn"
        if suffix not in text:
            text = text.rstrip() + suffix
        if len(text) > 280:
            text = text[:277].rstrip() + "..."
        return text

    if platform == "youtube":
        if len(text) > 400:
            text = text[:397].rstrip() + "..."
        return text

    if platform == "linkedin":
        if len(text) > 1800:
            text = text[:1797].rstrip() + "..."
        return text

    return text


def _instagram_format_for_weekday(weekday: int) -> str:
    weekday = int(weekday) % 7
    cadence = ["carousel", "reel", "carousel", "reel", "carousel", "reel", "static"]
    return cadence[weekday]


def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--bundle", help="Legacy support (ignored in queue mode)")
    parser.add_argument("--state_path", default="state.json", help="Path to state.json")
    args = parser.parse_args()
    platform = args.platform.lower()
    state_path = args.state_path
    state_dir = os.path.dirname(state_path) or "."

    if not os.path.exists(state_path):
        print(f"Error: {state_path} not found.")
        sys.exit(1)

    # Normalize any Windows-style backslashes in paths to forward slashes.
    # This prevents stale GitHub Actions states from failing on Linux runners.
    _fixed_paths = False
    def _normalize_paths(obj):
        global _fixed_paths
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, str) and chr(92) in v:
                    obj[k] = v.replace(chr(92), "/")
                    _fixed_paths = True
                else:
                    _normalize_paths(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str) and chr(92) in v:
                    obj[i] = v.replace(chr(92), "/")
                    _fixed_paths = True
                else:
                    _normalize_paths(v)
    _normalize_paths(state := load_state(state_path))
    if _fixed_paths:
        save_state(state, state_path)

    # Ensure we are operating on the freshest remote state to avoid stale skips.
    if os.environ.get("GITHUB_ACTIONS") == "true":
        try:
            result = subprocess.run(
                ["git", "pull", "--rebase", "origin", "master"],
                check=False,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"Git pull rebase failed with code {result.returncode}, trying merge fallback...")
                print(result.stderr)
                # Fallback: try a non-rebase pull
                result2 = subprocess.run(
                    ["git", "pull", "origin", "master"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                print(result2.stdout)
                print(result2.stderr)
                if result2.returncode != 0:
                    print(f"Git pull fallback also failed with code {result2.returncode}. Proceeding with local state.")
        except Exception as pull_exc:
            print(f"Git pull exception: {pull_exc}")

    state = load_state(state_path)

    # --- Normalize active_bundle: allow string ID or dict ---
    active = state.get("active_bundle")
    if isinstance(active, str):
        queue = state.get("content_queue", [])
        found = next((item for item in queue if item.get("post_id") == active), None)
        if found is None:
            print(f"Active bundle id {active} not found in queue. Clearing.")
            state["active_bundle"] = None
            active = None
            save_state(state, state_path)
        else:
            active = found
            state["active_bundle"] = active
            save_state(state, state_path)

    # --- Self-healing: clear stale active bundle when all required platforms are already posted ---
    if active and isinstance(active, dict):
        required = required_platforms(state_path)
        posted = active.get("platforms_posted", [])
        if all(p in posted for p in required):
            print(f"Active bundle {active.get('post_id')} already fully posted. Clearing.")
            state["active_bundle"] = None
            active = None
            save_state(state, state_path)

    # --- Queue Management ---
    if not active:
        queue = list(state.get("content_queue", []))
        if not queue:
            print(f"Content queue in {state_path} is empty. Nothing to prepare.")
            sys.exit(0)

        active, queue = _select_next_bundle_for_platform(state, platform, state_path)
        if not active:
            print(f"Content queue in {state_path} has only already-posted bundles for {platform.upper()}. Clearing.")
            state["content_queue"] = []
            save_state(state, state_path)
            sys.exit(0)

        state["active_bundle"] = active
        state["content_queue"] = queue
        print(f"Pulled bundle {active.get('post_id')} from queue for {platform.upper()}. Remaining: {len(queue)}")

    if isinstance(active, int):
        # active_bundle stored as bare int — look it up in queue or state
        found = None
        for b in state.get("content_queue", []):
            if isinstance(b, dict):
                if b.get("post_id") == active:
                    found = b
                    break
            elif b == active:
                found = {"post_id": b}
                break
        if found:
            active = found
            state["active_bundle"] = found
            save_state(state, state_path)
        else:
            # Int not in queue: pull next from queue instead
            queue = list(state.get("content_queue", []))
            if queue:
                active = queue.pop(0) if isinstance(queue[0], dict) else {"post_id": queue.pop(0)}
                state["active_bundle"] = active
                state["content_queue"] = queue
                save_state(state, state_path)
                print(f"Pulled bundle {active.get('post_id')} from queue for {platform.upper()}. Remaining: {len(queue)}")
            else:
                print(f"Error: active_bundle {active} not found and queue is empty.")
                sys.exit(1)

    if not isinstance(active, dict):
        print("Error: active_bundle is not a valid dict.")
        sys.exit(1)

    # Never re-prepare a platform that already posted this bundle.
    # Removed redundant skip guard; publishers handle skip logic themselves.

    print(f"[{state_path}] Preparing assets from bundle: {active.get('post_id')}")

    bundle_format = (active.get('format') or 'image').lower()
    # Determine required media based on bundle format/content
    if bundle_format == 'reel':
        media_required = {"reel"}
        media_optional = {"image", "story"}
    elif bundle_format == 'carousel':
        media_required = {"image"}
        media_optional = {"reel", "story"}
    else:
        media_required = {"image"}
        media_optional = {"reel", "story"}
    # Clear stale artifacts before copying new ones to prevent old posts from persisting
    stale = ["output.jpg", "caption.txt", "reel.mp4", "story.jpg", "carousel.json"]
    for fname in stale:
        p = os.path.join(state_dir, fname)
        if os.path.exists(p):
            os.remove(p)
            print(f"Removed stale {fname}")
    carousel_dir = os.path.join(state_dir, "carousel")
    if os.path.isdir(carousel_dir):
        shutil.rmtree(carousel_dir, ignore_errors=True)
        print("Removed stale carousel/")

    media_map = {
        "image": "output.jpg",
        "reel": "reel.mp4",
        "story": "story.jpg",
    }

    for key, local_name in media_map.items():
        if key in media_optional:
            src = active.get(key)
            if not src:
                continue
            target_path = os.path.join(state_dir, local_name)
            candidates = [
                src,
                os.path.join(state_dir, src),
                os.path.join(state_dir, os.path.basename(src)),
            ]
            copied = False
            for cand in candidates:
                if not copied and os.path.exists(cand):
                    shutil.copy(cand, target_path)
                    print(f"Copied {cand} -> {target_path}")
                    copied = True
                    break
            if not copied:
                print(f"⚠ Optional media '{key}' ({src}) not found; skipping.")
            continue

        src = active.get(key)
        if not src:
            # Caption-only bundle (e.g., Wilma when image generation failed).
            # Skip media copy; only caption will be posted.
            print(f"⚠ Required media '{key}' missing for bundle {active.get('post_id') or state.get('active_bundle', {}).get('post_id')} — caption-only mode.")
            continue

        target_path = os.path.join(state_dir, local_name)
        copied = False
        norm_src = src.replace('\\\\', '/').replace('/', os.sep)
        candidates = [
            src,
            os.path.join(state_dir, src),
            os.path.join(state_dir, norm_src),
            os.path.join(state_dir, os.path.basename(src)),
            os.path.join(state_dir, os.path.basename(norm_src)),
        ]
        for cand in candidates:
            if not copied and os.path.exists(cand):
                shutil.copy(cand, target_path)
                print(f"Copied {cand} -> {target_path}")
                copied = True
                break
        if not copied:
            print(f"❌ Critical: Required media '{key}' ({src}) not found for bundle {active.get('post_id') or state.get('active_bundle', {}).get('post_id')}.")
            print(f"   Tried candidates: {candidates}")
            sys.exit(1)

    policy = _platform_policy(platform)

    # --- Instagram weekly cadence ---
    if platform.lower() == "instagram":
        env_override = os.environ.get("INSTAGRAM_CAROUSEL_WORKFLOW", "").strip().lower()
        if env_override == "1":
            today_format = "carousel"
        else:
            today_format = _instagram_format_for_weekday(datetime.utcnow().weekday())
        print(f"Instagram weekly cadence for today: {today_format}")
        if today_format == "reel":
            policy["use_carousel"] = False
            policy["use_reel"] = True
            policy["use_static_image"] = False
        elif today_format == "carousel":
            policy["use_carousel"] = True
            policy["use_reel"] = False
            policy["use_static_image"] = False
        else:
            policy["use_carousel"] = False
            policy["use_reel"] = False
            policy["use_static_image"] = True

    # --- Prepare Carousel (if present) ---
    carousel_paths = active.get("carousel") or []

    # Carousel-day guard: LinkedIn carousels on Mon/Wed/Fri/Sun
    if carousel_paths and platform.lower() == 'linkedin' and datetime.utcnow().weekday() not in {0, 2, 4, 6}:
        print(f"⏭️ Carousel skipped: bundle {active.get('post_id')} has carousel slides, but today is not a carousel day. Falling back to single image.")
        carousel_paths = []

    if not carousel_paths:
        carousel_json = os.path.join(state_dir, "carousel.json")
        if os.path.exists(carousel_json):
            os.remove(carousel_json)
            print(f"Removed stale {carousel_json} for non-carousel bundle {active.get('post_id')}")
        carousel_dir = os.path.join(state_dir, "carousel")
        if os.path.isdir(carousel_dir):
            shutil.rmtree(carousel_dir, ignore_errors=True)
            print(f"Removed stale {carousel_dir}/ for non-carousel bundle {active.get('post_id')}")

    # Fallback: if the bundle has no carousel paths but we are on a carousel
    # day and there are existing deterministic carousel slides in images/, use
    # them so the post is not downgraded to a static image/reel.
    if not carousel_paths and policy.get("use_carousel") and platform.lower() == "instagram":
        existing = sorted(Path(state_dir).glob("images/carousel_*_slide_*.jpg"))
        if existing:
            existing = existing[-5:]
            carousel_paths = [str(p) for p in existing]
            print(f"Using {len(carousel_paths)} existing carousel slides as fallback for bundle {active.get('post_id')}")

    if carousel_paths and policy.get("use_carousel"):
        carousel_dir = os.path.join(state_dir, "carousel")
        os.makedirs(carousel_dir, exist_ok=True)
        prepared_paths = []
        for idx, src in enumerate(carousel_paths, start=1):
            target_path = os.path.join(carousel_dir, f"slide_{idx}.jpg")
            if os.path.exists(src):
                shutil.copy(src, target_path)
                prepared_paths.append(target_path)
            else:
                alt_src = os.path.join(state_dir, os.path.basename(src))
                if os.path.exists(alt_src):
                    shutil.copy(alt_src, target_path)
                    prepared_paths.append(target_path)
                else:
                    print(f"Warning: Carousel slide {idx} ({src}) not found.")
        if prepared_paths:
            carousel_json = os.path.join(state_dir, "carousel.json")
            url_paths = ["carousel/" + os.path.basename(p) for p in prepared_paths]
            with open(carousel_json, "w", encoding="utf-8") as f:
                json.dump(url_paths, f, indent=2)
            print(f"Prepared carousel.json with {len(url_paths)} slides")

            if platform == "instagram":
                hosted_path = os.path.join(state_dir, "carousel_hosted_urls.json")
                if not os.path.exists(hosted_path):
                    hosted_urls = []
                    for p in prepared_paths:
                        url = upload_to_catbox(p)
                        if url:
                            hosted_urls.append(url)
                        else:
                            hosted_urls.append("")
                            print(f"Warning: carousel host upload failed for {p}")
                    with open(hosted_path, "w", encoding="utf-8") as f:
                        json.dump(hosted_urls, f, indent=2)
                    print(f"Wrote {len(hosted_urls)} hosted carousel URLs to {hosted_path}")

    # --- Prepare Caption ---
    captions = active.get("captions", {})
    if platform not in captions:
        print(f"Error: Caption for platform '{platform}' not found in active bundle.")
        sys.exit(1)

    raw_caption = clean_caption_formatting(captions.get(platform) or "")
    raw_caption = _apply_platform_tailoring(raw_caption, platform)
    if not raw_caption.strip():
        print(f"⚠ Caption for {platform.upper()} is empty — skipping ready flag to prevent blank post.")
        sys.exit(1)
    if "[Caption generation failed" in raw_caption or "Traceback" in raw_caption:
        print(f"⚠ Caption for {platform.upper()} looks like an error payload — aborting prepare.")
        sys.exit(1)

    caption_path = os.path.join(state_dir, "caption.txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(raw_caption)
    print(f"Prepared caption.txt for {platform.upper()} at {caption_path}")

    # --- Create Ready Flag ---
    is_wilma = "forwilma" in state_path or "wilma" in platform
    flag_prefix = "wilma_" if is_wilma else ""
    flag_name = f"{flag_prefix}{platform}_ready.flag"
    flag_path = os.path.join(state_dir, flag_name)

    with open(flag_path, "w", encoding="utf-8") as f:
        f.write(str(active.get("timestamp", "")))
    print(f"Created {flag_path}")

    if platform.lower() == "instagram":
        format_marker_path = os.path.join(state_dir, "instagram_format.txt")
        if os.path.exists(format_marker_path):
            try:
                with open(format_marker_path, "r", encoding="utf-8") as f:
                    current = (f.read() or "").strip().lower()
                if current in {"carousel", "reel", "static"}:
                    print(f"Preserved existing Instagram format marker: {current}")
            except Exception:
                pass
        else:
            with open(format_marker_path, "w", encoding="utf-8") as f:
                f.write(_instagram_format_for_weekday(datetime.utcnow().weekday()))
            print(f"Wrote Instagram format marker to {format_marker_path}")

    # Track prep attempts (informational only; posting guard uses platforms_posted).
    if "platforms_prepared" not in state["active_bundle"]:
        state["active_bundle"]["platforms_prepared"] = []
    if platform not in state["active_bundle"]["platforms_prepared"]:
        state["active_bundle"]["platforms_prepared"].append(platform)

    save_state(state, state_path)
    print(f"Assets ready for {platform.upper()}.")


if __name__ == "__main__":
    prepare()
