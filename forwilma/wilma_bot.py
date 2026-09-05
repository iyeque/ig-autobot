import os
import sys

# type: ignore[reportAttributeAccessIssue] — Pyright doesn't track io.TextIOWrapper.reconfigure
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[reportAttributeAccessIssue]
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[reportAttributeAccessIssue]

import json
import time
import shutil
import argparse
import requests
from datetime import datetime
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent
FORWILMA_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

# Load .env if not already loaded (for local testing)
from dotenv import load_dotenv
load_dotenv(dotenv_path=BASE_DIR / '.env')

# Import core logic
from shared_utils import clean_caption_formatting
try:
    from bot import (
        generate_caption,
        generate_image,
        _write_output_jpg,
        add_static_text_overlay,
        generate_reel,
        generate_wilma_carousel,
        apply_logo_watermark,
        _ai_verify_caption,
        _generate_text_ai_horde,
        _generate_image_ai_horde,
    )
except Exception as _e:
    import traceback
    print("⚠️ Falling back to Wilma local stubs for missing bot.py helpers")
    traceback.print_exc()

    def _fallback_generate_image_ai_horde(prompt: str) -> str:
        # type: ignore[reportReturnType] — local stub; returns empty string, not None
        return ""

    def _fallback_generate_text_ai_horde(prompt: str, system_prompt: str = "", max_tokens: int = 512) -> str:
        return ""

    def _fallback_add_logo(_path: str, *_args, **_kwargs) -> str:
        return _path

    def _fallback_add_static_overlay(_path: str, _text: str) -> str:
        return _path

    def _fallback_generate_reel(_img: str, _hook: str, _out: str):
        return _out, ""

    def _fallback_generate_caption(*_args, **_kwargs):
        return ""

    def _fallback_generate_carousel(*_args, **_kwargs):
        return []

    def _fallback_sanitize_image_prompt(prompt: str) -> str:
        return (prompt or "").strip()

    def _fallback_extract_hook_text(_text: str) -> str:
        text = (_text or "").strip()
        if not text:
            return ""
        return text.splitlines()[0][:100]

    def _fallback_editor_fallback(caption: str, platform: str, max_chars: int) -> str:
        text = caption.strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        return text.strip()

    def _fallback_get_available_horde_text_models() -> list[str]:
        return []

    def _fallback_write_output_jpg(src: str, dst: str) -> str:
        """Copy src image to dst (best-effort, no processing)."""
        try:
            shutil.copy(src, dst)
        except Exception:
            pass
        return dst

    def _fallback_ai_verify_caption(caption: str, platform: str, max_c: int) -> str:
        """No-op AI verification in fallback mode — return caption as-is."""
        return caption.strip() if caption else ""

    generate_image = _fallback_generate_image_ai_horde
    add_static_text_overlay = _fallback_add_static_overlay
    apply_logo_watermark = _fallback_add_logo
    generate_reel = _fallback_generate_reel
    generate_caption = _fallback_generate_caption
    generate_wilma_carousel = _fallback_generate_carousel
    _generate_text_ai_horde = _fallback_generate_text_ai_horde
    _generate_image_ai_horde = _fallback_generate_image_ai_horde
    sanitize_image_prompt = _fallback_sanitize_image_prompt
    extract_hook_text = _fallback_extract_hook_text
    _editor_fallback = _fallback_editor_fallback
    _get_available_horde_text_models = _fallback_get_available_horde_text_models
    _write_output_jpg = _fallback_write_output_jpg
    _ai_verify_caption = _fallback_ai_verify_caption


# ---------------------------------------------------------------------
# Wilma config and constants
# ---------------------------------------------------------------------
# Cerebras has been removed from the codebase.

# Digital Guardian / Wilma Specific Config
SCHEDULE_FILE = FORWILMA_DIR / "schedule.json"
STATE_FILE = FORWILMA_DIR / "state.json"
WILMA_IMAGES_DIR = FORWILMA_DIR / "images"
LOGO_PATH = FORWILMA_DIR / "DG Logo.png"

# Mission Context for AI
DIGITAL_GUARDIAN_MISSION = (
    "Digital Guardian is a digital safety platform that simplifies parenting. "
    "Mission: Bridge the gap between children's online exploration and well-being. "
    "Values: Healthy digital habits, fostering family bonds, open conversations, and proactive safety. "
    "Voice: Grounded in real family life, evidence-based, warm, and direct."
)

# WILMA BRAND SETTINGS (Safe, Trustworthy, Nature-Abstract)
WILMA_BRAND_BASE = (
    "ethereal nature photography, soft bokeh, pastel color palette, "
    "dreamlike atmosphere, gentle gradients, abstract organic forms, "
    "no people, no figures, no faces, no hands, no text"
)
WILMA_BRAND_SUFFIX = (
    "fine art print, painterly texture, studio ghibli inspired, "
    "watercolor overlay, serene mood, zentangle patterns, mandala motifs, "
    "tilt-shift blur, macro lens, morning mist, golden hour backlight"
)


# ---------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------
def _read_schedule():
    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
                if "content_queue" not in state:
                    state["content_queue"] = []
                return state
        except Exception:
            pass
    return {"current_day_index": 0, "history": [], "content_queue": []}

def _write_state(state):
    tmp_path = STATE_FILE.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, STATE_FILE)

# --- Pending-bundle helpers for mid-run failure recovery ---
def _save_pending(state, pending_data):
    state["pending_bundle"] = pending_data
    _write_state(state)

def _load_and_clear_pending(state):
    pending = state.pop("pending_bundle", None)
    if pending:
        _write_state(state)
    return pending


# ---------------------------------------------------------------------
# Resume / recovery
# ---------------------------------------------------------------------

def _try_resume_pending_wilma(state, platforms):
    """Resume a partially-generated Wilma bundle. Returns True if resumed."""
    pending = _load_and_clear_pending(state)
    if not pending:
        return False

    print(f"\n🔄 Resuming pending Wilma bundle: {pending.get('post_id')}")

    image_ok = pending.get("image") and os.path.exists(pending["image"])
    reflection_ok = bool(pending.get("master_reflection"))
    post = pending.get("post")

    if not image_ok or not reflection_ok:
        if not post:
            print("  ❌ Cannot resume: post data missing")
            return False
        try:
            print("  ⚠ Regenerating Wilma media assets...")
            visual_metaphor = _generate_wilma_visual_prompt(post["topic"])
            image_prompt = f"{WILMA_BRAND_BASE}, {visual_metaphor}, {WILMA_BRAND_SUFFIX}"

            if pending.get("image") and os.path.exists(pending["image"]):
                raw_image = pending["image"]
                print(f"  ♻️ Reusing existing Wilma image for resume: {raw_image}")
            else:
                print("⚠ No image available for resume; proceeding caption-only.")
                pending["image"] = None
                _save_pending(state, pending)
                return False

            processed = _write_output_jpg(raw_image, "temp_output.jpg")
            apply_logo_watermark("temp_output.jpg", str(LOGO_PATH))
            add_static_text_overlay("temp_output.jpg", post["topic"])
            shutil.copy("temp_output.jpg", pending["image"])

            master_system = f"""You are the lead strategist for Digital Guardian, writing as Wilma. Mission: {DIGITAL_GUARDIAN_MISSION}
            Voice rules:
            - Empathetic, authoritative, research-backed, and relatable. Never preachy.
            - Write like a founder who has lived the tension between "scary tech" and healthy family life.
            - Use real references when relevant: American Academy of Pediatrics, University of Michigan, etc.
            - Keep language plain and warm. Personal and vulnerable in Builder content; practical and direct in educational content.
            - Every claim should feel earned by story or result, not guru lecture.
            - End with a single, low-friction engagement hook. No jargon, no marketing fluff, no AI-isms.
            - CRITICAL: Wilma has ONE daughter, age 2. When content involves children, frame examples ONLY around her 2-year-old daughter, OR use generic collective terms like "kids," "children," or "families." NEVER invent stories about other specific children with different ages. NEVER say "my 4-year-old," "my 5-year-old," or any age other than 2.
            - If the topic implies a different age, adapt it to her 2-year-old daughter or use a generic framing.
            - Stay in the digital wellness lane 80% of the time. Cross-venture content is allowed only in Builder posts as founder-life context, never as standalone promo.
            - For Bluesky: keep it tighter and conversational, end with the fixed CTA line only.
            - For LinkedIn: keep it longer and platform-native, but still avoid mid-thought cutoffs.
            Write a complete, polished post about the topic below. Finish every sentence. Do not trail off mid-thought.
            """
            reflection_attempts = 2
            master_reflection = ""
            for _ in range(reflection_attempts):
                master_reflection = _generate_text_ai_horde(
                    f"Topic: {post['topic']}\nAudience: {post['audience']}",
                    system_prompt=master_system,
                    max_tokens=768,
                )
                if master_reflection and master_reflection.rstrip().endswith((".", "!", "?", "…", ":", ";")):
                    break
            pending["master_reflection"] = master_reflection
            print("  ✓ Wilma media regenerated")
        except Exception as e:
            print(f"  ❌ Resume failed: {e}")
            _save_pending(state, pending)
            return False
    else:
        print("  ✓ Reusing existing Wilma media assets")

    captions = pending.get("bundle_captions", {})
    for p in platforms:
        if captions.get(p):
            continue
        try:
            limits = {"bluesky": 300, "linkedin": 1800}
            hard_total_limits = {"bluesky": 300, "linkedin": 2000}
            max_c = limits.get(p.lower(), 1800)
            tailored_cap = _ai_verify_caption(pending.get("master_reflection") or "", p, max_c)
            tailored_cap = tailored_cap if tailored_cap is not None else ""
            final_cap = clean_caption_formatting(tailored_cap) or ""
            final_cap = _enforce_wilma_persona(final_cap)
            if p == "linkedin":
                final_cap += "\n\n#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips"
            elif p == "bluesky":
                final_cap = _strip_bluesky_cta(final_cap) + "\n\nWant to read more?... check out my LinkedIn"
            limit = hard_total_limits.get(p.lower(), 2000)
            if len(final_cap) > limit:
                final_cap = final_cap[: limit - 3] + "..."
            captions[p] = final_cap
            print(f"  ✓ Wilma caption for {p}: {len(final_cap)} chars")
        except Exception as e:
            print(f"  ⚠ Wilma caption failed for {p}: {e}")
            captions[p] = ""

    new_bundle = {
        "post_id": pending["post_id"],
        "timestamp": pending["timestamp"],
        "image": pending["image"],
        "captions": captions,
        "platforms_posted": [],
    }
    state["content_queue"].append(new_bundle)
    if post:
        state["last_topic"] = post.get("topic", "")
    _write_state(state)
    state.pop("pending_bundle", None)
    _write_state(state)
    print(f"  ✅ Wilma pending bundle resumed. Queue: {len(state['content_queue'])} items.\n")
    return True


# ---------------------------------------------------------------------
# Visual / prompt helpers
# ---------------------------------------------------------------------
def _generate_wilma_visual_prompt(topic):
    """
    Uses a deterministic fallback to turn a literal topic into a safe, abstract visual metaphor.
    """
    return topic


def _strip_bluesky_cta(text: str) -> str:
    """Remove common CTAs from Bluesky captions so only the hardcoded LinkedIn CTA remains."""
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


def _enforce_wilma_persona(caption: str) -> str:
    """
    Hard guard for Wilma voice rules.
    1. Wilma is a parent with ONE daughter, age 2.
    2. If the caption invents other specific children or wrong ages, coerce to
       either 'my 2-year-old daughter' or generic 'kids/children'.
    """
    if not caption:
        return caption
    import re

    caption = re.sub(
        r"\bmy\s+\d+-year-old(?:[\s-]+(?:old\s+)?(daughter|son|child|kid))?\b",
        "my 2-year-old daughter",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"\bmy\s+(child|kid|toddler|baby)\b",
        "my 2-year-old daughter",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"\b(?:a|an)\s+\d+-year-old(?:[\s-]+(?:old\s+)?(daughter|son|child|kid))?\b",
        "a 2-year-old",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"\bour\s+\d+-year-old(?:[\s-]+(?:old\s+)?(daughter|son|child|kid))?\b",
        "my 2-year-old daughter",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"\b(?:a|an)\s+\d+-year-old\b",
        "a 2-year-old",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"\b(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty)-year-old(?:[\s-]+(?:old\s+)?(daughter|son|child|kid))?\b",
        "2-year-old",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"\bmy\s+son\b",
        "my daughter",
        caption,
        flags=re.IGNORECASE,
    )
    return caption


def _wilma_carousel_slides(topic: str, pillar: str) -> list[str]:
    """
    Build 5 unique carousel slide texts for a Wilma bundle.
    Each slide is derived from the topic and pillar so every day gets
    different content instead of the old hardcoded repeats.
    """
    t = topic.strip().rstrip(".").lower()
    p = pillar.replace("_", " ").title()

    slide1 = f"What if {topic.strip()}?"

    if any(k in t for k in ("parental", "control", "rule", "screen", "phone")):
        slide2 = f"The {p} truth: controls alone don't change behavior."
    elif any(k in t for k in ("school", "morning", "lock", "phone-free")):
        slide2 = "Schools that tried this saw the same unexpected result."
    elif any(k in t for k in ("family", "balance", "juggling")):
        slide2 = "Balance isn't found — it's built, one small choice at a time."
    elif any(k in t for k in ("scam", "click", "link", "teen")):
        slide2 = "Three seconds is all it takes. Here's how to stop it."
    elif any(k in t for k in ("notification", "distraction", "audit")):
        slide2 = f"Your phone has {p.lower()} settings that cut noise fast."
    else:
        slide2 = f"{p} is not what most people think it is."

    if any(k in t for k in ("parental", "control", "rule", "screen", "phone")):
        slide3 = "The kids who thrived weren't the ones with the strictest rules."
    elif any(k in t for k in ("school", "morning", "lock", "phone-free")):
        slide3 = "Teachers reported something they hadn't expected: quieter hallways, louder classrooms."
    elif any(k in t for k in ("family", "balance", "juggling")):
        slide3 = "The myth is that balance means doing it all. The reality is choosing what matters."
    elif any(k in t for k in ("scam", "click", "link", "teen")):
        slide3 = "Most scams don't look like scams. They look like something your kid already trusts."
    elif any(k in t for k in ("notification", "distraction", "audit")):
        slide3 = "The average phone interrupts us 60 times a day. Most are optional."
    elif any(k in t for k in ("boundary", "limit", "change", "mind")):
        slide3 = "What I got wrong for years: controls are a shortcut, not a solution."
    else:
        slide3 = "What worked for our family was simpler than expected."

    if any(k in t for k in ("rule", "routine", "schedule")):
        slide4 = "One consistent rule beats a dozen broken promises."
    elif any(k in t for k in ("conversation", "talk", "question")):
        slide4 = "Start with one honest question. Not a lecture."
    elif any(k in t for k in ("boundary", "limit", "control")):
        slide4 = "Boundaries aren't barriers — they're guardrails."
    elif any(k in t for k in ("scam", "click", "link", "safety")):
        slide4 = "Pause. Check. Then decide. Three steps, every time."
    elif any(k in t for k in ("notification", "setting", "audit")):
        slide4 = "Turn off the pings that don't matter. Keep the ones that do."
    elif any(k in t for k in ("family", "balance", "juggling")):
        slide4 = "Put the phone down first. The rest follows."
    else:
        slide4 = "Small consistency beats big restrictions every time."

    if any(k in t for k in ("rule", "setting", "routine")):
        slide5 = "What's one screen-time rule that actually works in your house?"
    elif any(k in t for k in ("scam", "safety", "teen")):
        slide5 = "What's the sneakiest scam your teen almost fell for?"
    elif any(k in t for k in ("family", "balance", "juggling")):
        slide5 = "What's your biggest digital struggle right now?"
    elif any(k in t for k in ("school", "morning", "phone-free")):
        slide5 = "Would you try a phone-free morning at your kid's school?"
    elif any(k in t for k in ("notification", "distraction")):
        slide5 = "Which app do you wish had a mute button for good?"
    else:
        slide5 = "What's one change you'd make to your family's screen routine?"

    return [slide1, slide2, slide3, slide4, slide5]


# ---------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Digital Guardian (Wilma) Bot")
    parser.add_argument("--platform", type=str, default="linkedin", choices=["linkedin", "bluesky"],
                      help="Target platform for single-post mode")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "generate_all"],
                      help="Mode: single or generate_all")
    args = parser.parse_args()

    if args.mode == "generate_all":
        platforms = ["linkedin", "bluesky"]
        print(f"🚀 UNIFIED WILMA MODE: Creating assets for {platforms}")
    else:
        platforms = [args.platform]

    state = _read_state()
    os.chdir(str(FORWILMA_DIR))
    schedule = _read_schedule()

    # --- CONTENT QUEUE LOGIC ---
    target_buffer = 1
    current_buffer = len(state.get("content_queue", []))

    if args.mode == "generate_all":
        if current_buffer >= target_buffer:
            print(f"✅ Wilma buffer is full ({current_buffer}/{target_buffer}). Nothing to generate.")
            return
        to_generate = target_buffer - current_buffer
        print(f"🔄 Wilma Buffer status: {current_buffer}/{target_buffer}. Generating {to_generate} new bundles...")

        # Resume any pending Wilma bundle from a previous partial run first
        if _try_resume_pending_wilma(state, platforms):
            current_buffer = len(state.get("content_queue", []))
            to_generate = max(0, target_buffer - current_buffer)
            state["current_day_index"] += 1
            _write_state(state)
            print(f"Wilma buffer after resume: {current_buffer}/{target_buffer}. {to_generate} more to generate.")
    else:
        to_generate = 1

    for i in range(to_generate):
        if state["current_day_index"] >= len(schedule):
            print("🎉 Schedule complete! Restarting...")
            state["current_day_index"] = 0

        post_data = schedule[state["current_day_index"]]
        day_num = post_data["day"]

        posted_history = state.get("platform_posted_bundles", {})
        already_posted = (
            f"day_{day_num}" in posted_history.get("linkedin", []) and
            f"day_{day_num}" in posted_history.get("bluesky", [])
        )
        if already_posted:
            print(f"  ⏭️ Day {day_num} already posted to both platforms; advancing schedule.")
            state["current_day_index"] += 1
            _write_state(state)
            continue

        existing_ids = {b.get("post_id") for b in state.get("content_queue", [])}
        if f"day_{day_num}" in existing_ids:
            print(f"  ⏭️ Day {day_num} already queued; skipping to next day.")
            state["current_day_index"] += 1
            _write_state(state)
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"day{day_num}_{timestamp}.jpg"
        image_path = f"images/{image_name}"

        pending = {
            "post_id": f"day_{day_num}",
            "timestamp": timestamp,
            "post": post_data,
            "image": image_path,
            "master_reflection": None,
            "bundle_captions": {},
        }

        # --- 1. MEDIA GENERATION (step-by-step with progress save) ---
        image_available = False
        raw_image_path = None

        try:
            visual_metaphor = _generate_wilma_visual_prompt(post_data["topic"])
            print(f"  Visual Metaphor: {visual_metaphor}")

            image_prompt = f"{WILMA_BRAND_BASE}, {visual_metaphor}, {WILMA_BRAND_SUFFIX}"

            image_generated = False
            for image_attempt in range(3):
                try:
                    raw_image = generate_image(image_prompt)
                    print(f"  ✓ Wilma hero image generated on attempt {image_attempt + 1}: {raw_image}")
                    raw_image_path = raw_image
                    image_generated = True
                    break
                except Exception as e:
                    print(f"  ⚠ Image generation attempt {image_attempt + 1}/3 failed: {e}")
                    if image_attempt < 2:
                        print("  Waiting 5 minutes before next attempt...")
                        time.sleep(5 * 60)

            if image_generated and raw_image_path:
                processed = _write_output_jpg(raw_image_path, "temp_output.jpg")
                apply_logo_watermark("temp_output.jpg", str(LOGO_PATH))
                add_static_text_overlay("temp_output.jpg", post_data["topic"])
                shutil.copy("temp_output.jpg", image_path)
                print(f"✓ Image saved: {image_path}")
                pending["image"] = image_path
            else:
                pending["image"] = None
                print("⚠ Image unavailable after 3 attempts; proceeding caption-only.")

            pending["carousel"] = []
            if post_data.get("carousel"):
                print("  🎞 Generating local Wilma carousel slides...")
                for carousel_attempt in range(3):
                    try:
                        topic_clean = (post_data.get("topic") or "").strip().rstrip(".")
                        wilma_slides = _wilma_carousel_slides(
                            topic_clean,
                            post_data.get("pillar") or post_data.get("type") or "General",
                        )
                        carousel_paths = generate_wilma_carousel(
                            post_data.get("pillar") or post_data.get("type") or "General",
                            topic_clean,
                            timestamp,
                            footer_text="DIGITAL GUARDIAN | WILMA",
                            slides=wilma_slides,
                        )
                        if not carousel_paths:
                            raise RuntimeError("generate_carousel returned no slides")
                        pending["carousel"] = [str(Path(p)) for p in carousel_paths]
                        print(f"  ✓ Carousel slides prepared: {len(carousel_paths)}")
                        break
                    except Exception as e:
                        print(f"  ⚠ Carousel generation attempt {carousel_attempt + 1}/3 failed: {e}")
                        if carousel_attempt < 2:
                            print("  Retrying carousel generation immediately...")
                            continue
                        pending["carousel"] = []
                        print("  ⚠ Carousel unavailable after 3 attempts; continuing without carousel.")
            _save_pending(state, pending)
        except Exception as e:
            print(f"⚠ Media generation failed ({e}); proceeding caption-only.")
            pending["image"] = None
            pending["carousel"] = []
            _save_pending(state, pending)

        # --- THE MASTER REFLECTION ---
        print("Generating Master Reflection for Wilma...")
        master_system = f"""You are the lead strategist for Digital Guardian, writing as Wilma. Mission: {DIGITAL_GUARDIAN_MISSION}

Voice rules:
- Empathetic, authoritative, research-backed, and relatable. Never preachy.
- Write like a founder who has lived the tension between "scary tech" and healthy family life.
- Use real references when relevant: American Academy of Pediatrics, University of Michigan, etc.
- Keep language plain and warm. Personal and vulnerable in Builder content; practical and direct in educational content.
- Every claim should feel earned by story or result, not guru lecture.
- End with a single, low-friction engagement hook. No jargon, no marketing fluff, no AI-isms.
- CRITICAL: Wilma has ONE daughter, age 2. When content involves children, frame examples ONLY around her 2-year-old daughter, OR use generic collective terms like "kids," "children," or "families." NEVER invent stories about other specific children with different ages. NEVER say "my 4-year-old," "my 5-year-old," or any age other than 2.
- If the topic implies a different age, adapt it to her 2-year-old daughter or use a generic framing.
- Stay in the digital wellness lane 80% of the time. Cross-venture content is allowed only in Builder posts as founder-life context, never as standalone promo.
- For Bluesky: keep it tighter and conversational, end with the fixed CTA line only.
- For LinkedIn: keep it longer and platform-native, but still avoid mid-thought cutoffs.
Write a complete, polished post about the topic below. Finish every sentence. Do not trail off mid-thought.
"""
        reflection_attempts = 2
        master_reflection = ""
        for _ in range(reflection_attempts):
            master_reflection = _generate_text_ai_horde(
                f"Topic: {post_data['topic']}\nAudience: {post_data['audience']}",
                system_prompt=master_system,
                max_tokens=768,
            )
            if master_reflection and master_reflection.rstrip().endswith((".", "!", "?", "…", ":", ";")):
                break
            if _ < reflection_attempts - 1:
                print("⚠ Master reflection ended mid-sentence, retrying...")
        pending["master_reflection"] = master_reflection
        _save_pending(state, pending)
        print(f"✓ Master reflection acquired ({len(master_reflection)} chars).")

        # --- 2. CAPTION GENERATION (deterministic local editing) ---
        bundle_captions = {}
        for p in platforms:
            print(f"  Tailoring for {p.upper()}...")
            try:
                limits = {"bluesky": 250, "threads": 450, "instagram": 1400,
                          "linkedin": 1800, "pinterest": 450, "youtube": 400, "facebook": 500}
                hard_total_limits = {"bluesky": 300, "threads": 500, "pinterest": 500,
                                     "instagram": 1600, "linkedin": 2000, "youtube": 600, "facebook": 600}
                max_c = limits.get(p.lower(), 1800)
                tailored_cap = _ai_verify_caption(master_reflection, p, max_c)
                if tailored_cap is None:
                    raise ValueError("AI editor returned None")
                final_cap = clean_caption_formatting(tailored_cap)
                final_cap = _enforce_wilma_persona(final_cap)

                if p == "linkedin":
                    final_cap += "\n\n#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips"
                elif p == "bluesky":
                    final_cap = _strip_bluesky_cta(final_cap) + "\n\nWant to read more?... check out my LinkedIn"

                limit = hard_total_limits.get(p.lower(), 2000)
                if len(final_cap) > limit:
                    cut = final_cap[: limit - 3].rsplit(".", 1)
                    if len(cut) == 2 and len(cut[0]) > limit - 300:
                        final_cap = cut[0].rstrip() + "..."
                    else:
                        final_cap = final_cap[: limit - 3] + "..."
                    print(f"  ⚠ {p.upper()} caption truncated to {len(final_cap)} chars (hard limit {limit})")

                bundle_captions[p] = final_cap
                pending["bundle_captions"][p] = final_cap
                _save_pending(state, pending)
                print(f"  ✓ Caption for {p}: {len(final_cap)} chars")

            except Exception as e:
                print(f"  ⚠ Skipping {p} for this bundle due to caption generation failure: {e}")
                bundle_captions[p] = ""
                pending["bundle_captions"][p] = ""
                _save_pending(state, pending)

        # --- 3. ADD TO QUEUE ---
        carousel_paths = pending.get("carousel") or []
        new_bundle = {
            "post_id": f"day_{day_num}",
            "timestamp": timestamp,
            "image": image_path,
            "carousel": carousel_paths,
            "captions": bundle_captions,
            "platforms_posted": [],
            "type": post_data.get("type", "TOFU"),
            "pillar": post_data.get("pillar", ""),
            "topic": post_data.get("topic", ""),
            "audience": post_data.get("audience", "All"),
            "platforms_prepared": list(platforms),
        }

        if args.mode == "generate_all":
            state["content_queue"].append(new_bundle)
            state["current_day_index"] += 1
            state["history"].append({
                "day": day_num,
                "timestamp": datetime.now().isoformat(),
                "image": image_path,
            })
            _write_state(state)

            state.pop("pending_bundle", None)
            _write_state(state)

            timestamp = datetime.now().isoformat()
            for platform in platforms:
                flag_path = Path(f"wilma_{platform}_ready.flag")
                try:
                    flag_path.write_text(timestamp, encoding="utf-8")
                except Exception:
                    pass
            print(f"✅ Wilma Bundle Day {day_num} added to queue and ready flags written.")
        else:
            shutil.copy("temp_output.jpg", "output.jpg")
            with open("wilma_bundle.json", "w", encoding="utf-8") as f:
                json.dump(bundle_captions, f, indent=2)
            for p in platforms:
                with open(f"wilma_{p}_ready.flag", "w") as f:
                    f.write(timestamp)
            print("✓ Wilma single mode assets ready.")
            return

    print(f"✓ Wilma generation cycle complete. Queue: {len(state['content_queue'])} items.")


if __name__ == "__main__":
    main()
