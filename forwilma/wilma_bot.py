import os
import sys
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
try:
    from bot import (
        generate_caption,
        generate_image,
        _write_output_jpg,
        add_static_text_overlay,
        generate_reel,
        apply_logo_watermark,
        _clean_caption_formatting,
        _ai_verify_caption,
        _generate_text_ai_horde,
        _generate_image_ai_horde,
    )
except Exception as _e:
    import traceback
    print("❌ Error: Could not import core logic from bot.py.")
    traceback.print_exc()
    sys.exit(1)

# Environment
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")

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
        except: pass
    return {"current_day_index": 0, "history": [], "content_queue": []}

def _write_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

# --- Pending-bundle helpers for mid-run failure recovery ---
def _save_pending(state, pending_data):
    state["pending_bundle"] = pending_data
    _write_state(state)

def _load_and_clear_pending(state):
    pending = state.pop("pending_bundle", None)
    if pending:
        _write_state(state)
    return pending

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
            visual_metaphor = _generate_wilma_visual_prompt(post['topic'])
            image_prompt = f"{WILMA_BRAND_BASE}, {visual_metaphor}, {WILMA_BRAND_SUFFIX}"
            raw_image = generate_image(image_prompt)
            processed = _write_output_jpg(raw_image, "temp_output.jpg")
            apply_logo_watermark("temp_output.jpg", str(LOGO_PATH))
            add_static_text_overlay("temp_output.jpg", post['topic'])
            shutil.copy("temp_output.jpg", pending["image"])

            master_system = f"""You are the lead strategist for Digital Guardian, writing as Wilma. Mission: {DIGITAL_GUARDIAN_MISSION}
            Voice rules:
            - Speak like a parent who's actually lived this — relatable, not academic.
            - Use real-life scenarios: dinner tables, bedtime routines, car rides, homework struggles.
            - Reference concrete stats or research findings when relevant.
            - End with a single, low-friction engagement hook (a question or a small invitation), not a lecture.
            - Keep it concise. No jargon, no marketing fluff, no AI-isms.
            - CRITICAL: Wilma has ONE daughter, age 2. When content involves children, frame examples ONLY around her 2-year-old daughter, OR use generic collective terms like "kids," "children," or "families." NEVER invent stories about other specific children with different ages. NEVER say "my 4-year-old," "my 5-year-old," or any age other than 2.
            - If the topic implies a different age, adapt it to her 2-year-old daughter or use a generic framing.
            Write a complete, polished post about the topic below. Finish every sentence. Do not trail off mid-thought.
            """
            reflection_attempts = 2
            master_reflection = ""
            for _ in range(reflection_attempts):
                master_reflection = _generate_text_ai_horde(
                    f"Topic: {post['topic']}\nAudience: {post['audience']}",
                    system_prompt=master_system,
                    max_tokens=768
                )
                if master_reflection and master_reflection.rstrip().endswith(('.', '!', '?', '…', ':', ';')):
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
            limits = {"bluesky": 250, "linkedin": 1800}
            hard_total_limits = {"bluesky": 300, "linkedin": 2000}
            max_c = limits.get(p.lower(), 1800)
            tailored_cap = _ai_verify_caption(pending.get("master_reflection") or "", p, max_c)
            tailored_cap = tailored_cap if tailored_cap is not None else ""
            final_cap = _clean_caption_formatting(tailored_cap) or ""
            final_cap = _enforce_wilma_persona(final_cap)
            if p == "linkedin":
                final_cap += "\n\n#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips"
            elif p == "bluesky":
                final_cap = _strip_bluesky_cta(final_cap) + "\n\nWant to read more?... check out my LinkedIn"
            limit = hard_total_limits.get(p.lower(), 2000)
            if len(final_cap) > limit:
                final_cap = final_cap[:limit-3] + "..."
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
        "platforms_posted": []
    }
    state["content_queue"].append(new_bundle)
    if post:
        state["last_topic"] = post.get("topic", "")
    _write_state(state)
    state.pop("pending_bundle", None)
    _write_state(state)
    print(f"  ✅ Wilma pending bundle resumed. Queue: {len(state['content_queue'])} items.\n")
    return True

def _generate_wilma_visual_prompt(topic):
    """
    Uses Cerebras to turn a literal topic into a safe, abstract visual metaphor.
    This avoids triggering NSFW/CSAM filters by removing words like 'children' or 'kids'.
    """
    if not CEREBRAS_API_KEY:
        return topic # Fallback

    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
    
    prompt = f"""Topic: {topic}
    Generate a high-end visual metaphor for this digital wellness topic.
    RULES:
    1. NO humans, NO children, NO people, NO faces, NO hands, NO body parts.
    2. Use ONLY nature, light, weather, plants, water, stones, or abstract geometry.
    3. Mood adjectives: misty, luminous, calm, layered, symmetrical, organic.
    4. Format: 1 short sentence of descriptive keywords only.
    """
    
    try:
        payload = {
            "model": "gpt-oss-120b",
            "messages": [{"role": "system", "content": "You are a visual design expert. Output only the prompt."},
                         {"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 60
        }
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        msg = (r.json() or {}).get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "").strip()
        return content or topic
    except Exception:
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
    # Age-based replacement: my N-year-old -> my 2-year-old daughter
    caption = re.sub(
        r"\bmy\s+\d+-year-old(?:\s+(daughter|son|child|kid))?\b",
        "my 2-year-old daughter",
        caption,
        flags=re.IGNORECASE,
    )
    # Generic invented children: my child -> my 2-year-old daughter
    caption = re.sub(
        r"\bmy\s+(child|kid|toddler|baby)\b",
        "my 2-year-old daughter",
        caption,
        flags=re.IGNORECASE,
    )
    # If only age is mentioned without possessive, e.g. "a 4-year-old"
    caption = re.sub(
        r"\ba\s+\d+-year-old(?:\s+(daughter|son|child|kid))?\b",
        "a 2-year-old",
        caption,
        flags=re.IGNORECASE,
    )
    # Son -> daughter (Wilma only has a daughter)
    caption = re.sub(
        r"\bmy\s+son\b",
        "my daughter",
        caption,
        flags=re.IGNORECASE,
    )
    return caption

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
    target_buffer = 5
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
            # Advance the schedule pointer so we don't recreate the resumed day
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
        print(f"\n📦 GENERATING WILMA BUNDLE {i+1}/{to_generate} (Day {day_num})...")

        # Skip if this day is already queued to avoid duplicates
        existing_ids = {b.get("post_id") for b in state.get("content_queue", [])}
        if f"day_{day_num}" in existing_ids:
            print(f"  ⏭️ Day {day_num} already queued; skipping to next day.")
            state["current_day_index"] += 1
            _write_state(state)
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"day{day_num}_{timestamp}.jpg"
        image_path = os.path.join("images", image_name)

        # Initialize pending bundle for this run
        pending = {
            "post_id": f"day_{day_num}",
            "timestamp": timestamp,
            "post": post_data,
            "image": image_path,
            "master_reflection": None,
            "bundle_captions": {},
        }

        def _find_existing_day_image(day_num):
            """Reuse any existing image for this day if generation fails."""
            try:
                for name in os.listdir("images"):
                    if name.startswith(f"day{day_num}_") and name.endswith((".jpg", ".jpeg", ".png")):
                        return os.path.join("images", name)
            except Exception:
                pass
            return None

        # --- 1. MEDIA GENERATION (step-by-step with progress save) ---
        try:
            visual_metaphor = _generate_wilma_visual_prompt(post_data['topic'])
            print(f"  Visual Metaphor: {visual_metaphor}")
            
            image_prompt = f"{WILMA_BRAND_BASE}, {visual_metaphor}, {WILMA_BRAND_SUFFIX}"
            raw_image = generate_image(image_prompt)
            
            # Temporary local path for processing
            processed = _write_output_jpg(raw_image, "temp_output.jpg")
            apply_logo_watermark("temp_output.jpg", str(LOGO_PATH))
            add_static_text_overlay("temp_output.jpg", post_data['topic'])
            
            # Save to final persistent path
            shutil.copy("temp_output.jpg", image_path)
            print(f"✓ Image saved: {image_path}")
            _save_pending(state, pending)

        except Exception as e:
            # If the target image already exists on disk, reuse it and continue
            if os.path.exists(image_path):
                print(f"  ⚠ Image generation failed ({e}); reusing existing image: {image_path}")
            else:
                fallback = _find_existing_day_image(day_num)
                if fallback:
                    image_path = fallback
                    pending["image"] = image_path
                    print(f"  ⚠ Image generation failed ({e}); reusing earlier image: {image_path}")
                else:
                    print(f"❌ Image generation failed: {e}. Progress saved, will resume next run.")
                    _save_pending(state, pending)
                    return

        # --- THE MASTER REFLECTION ---
        print("Generating Master Reflection for Wilma...")
        master_system = f"""You are the lead strategist for Digital Guardian, writing as Wilma. Mission: {DIGITAL_GUARDIAN_MISSION}

Voice rules:
- Speak like a parent who's actually lived this — relatable, not academic.
- Use real-life scenarios: dinner tables, bedtime routines, car rides, homework struggles.
- Reference concrete stats or research findings when relevant.
- End with a single, low-friction engagement hook (a question or a small invitation), not a lecture.
- Keep it concise. No jargon, no marketing fluff, no AI-isms.
- CRITICAL: Wilma has ONE daughter, age 2. When content involves children, frame examples ONLY around her 2-year-old daughter, OR use generic collective terms like "kids," "children," or "families." NEVER invent stories about other specific children with different ages. NEVER say "my 4-year-old," "my 5-year-old," or any age other than 2.
- If the topic implies a different age, adapt it to her 2-year-old daughter or use a generic framing.
Write a complete, polished post about the topic below. Finish every sentence. Do not trail off mid-thought.
"""
        # Retry up to 2x if reflection ends abruptly
        reflection_attempts = 2
        master_reflection = ""
        for _ in range(reflection_attempts):
            master_reflection = _generate_text_ai_horde(
                f"Topic: {post_data['topic']}\nAudience: {post_data['audience']}",
                system_prompt=master_system,
                max_tokens=768
            )
            if master_reflection and master_reflection.rstrip().endswith(('.', '!', '?', '…', ':', ';')):
                break
            if _ < reflection_attempts - 1:
                print("⚠ Master reflection ended mid-sentence, retrying...")
        pending["master_reflection"] = master_reflection
        _save_pending(state, pending)
        print(f"✓ Master reflection acquired ({len(master_reflection)} chars).")

        # --- 2. CAPTION GENERATION (AI CRITIC EDITS) ---
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
                final_cap = _clean_caption_formatting(tailored_cap)
                final_cap = _enforce_wilma_persona(final_cap)
                
                if p == "linkedin":
                     final_cap += "\n\n#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips"
                elif p == "bluesky":
                     final_cap = _strip_bluesky_cta(final_cap) + "\n\nWant to read more?... check out my LinkedIn"

                # Hard limit enforcement (keep CTA/hashtags, truncate body only)
                limit = hard_total_limits.get(p.lower(), 2000)
                if len(final_cap) > limit:
                    # Back up from end and truncate at last sentence boundary before limit
                    cut = final_cap[:limit-3].rsplit('.', 1)
                    if len(cut) == 2 and len(cut[0]) > limit - 300:
                        final_cap = cut[0].rstrip() + '...'
                    else:
                        final_cap = final_cap[:limit-3] + '...'
                    print(f"  ⚠ {p.upper()} caption truncated to {len(final_cap)} chars (hard limit {limit})")

                bundle_captions[p] = final_cap
                pending["bundle_captions"][p] = final_cap
                _save_pending(state, pending)
                print(f"  ✓ Caption for {p}: {len(final_cap)} chars")

            except Exception as e:
                print(f"  Tailoring failed for {p}: {e}")
                bundle_captions[p] = f"[Caption generation failed: {e}]"
                pending["bundle_captions"][p] = bundle_captions[p]
                _save_pending(state, pending)

        # --- 3. ADD TO QUEUE ---
        new_bundle = {
            "post_id": f"day_{day_num}",
            "timestamp": timestamp,
            "image": image_path,
            "captions": bundle_captions,
            "platforms_posted": []
        }

        if args.mode == "generate_all":
            state["content_queue"].append(new_bundle)
            state["current_day_index"] += 1
            state["history"].append({
                "day": day_num,
                "timestamp": datetime.now().isoformat(),
                "image": image_path
            })
            _write_state(state)
            
            # Clear pending on success
            state.pop("pending_bundle", None)
            _write_state(state)
            
            print(f"✅ Wilma Bundle Day {day_num} added to queue.")
        else:
            # Legacy single mode
            shutil.copy("temp_output.jpg", "output.jpg")
            with open("wilma_bundle.json", "w", encoding="utf-8") as f:
                json.dump(bundle_captions, f, indent=2)
            for p in platforms:
                with open(f"wilma_{p}_ready.flag", "w") as f: f.write(timestamp)
            print("✓ Wilma single mode assets ready.")
            return

    print(f"✓ Wilma generation cycle complete. Queue: {len(state['content_queue'])} items.")

if __name__ == "__main__":
    main()
