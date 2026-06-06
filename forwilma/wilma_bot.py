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
        add_logo_watermark,
        _clean_caption_formatting,
        _ai_verify_caption,
        _generate_text_ai_horde
    )
except ImportError:
    print("❌ Error: Could not import core logic from bot.py.")
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
    "Values: Healthy digital habits, fostering family bonds, open conversations, and proactive safety."
)

# WILMA BRAND SETTINGS (Safe, Trustworthy, Modern)
WILMA_BRAND_BASE = (
    "high-end professional photography, clean composition, soft natural lighting, "
    "warm and safe atmosphere, minimal clutter, elegant aesthetic"
)
WILMA_BRAND_SUFFIX = (
    "no humans, no faces, no text, ultra-sharp detail, 8k resolution, professional architectural or fine-art style"
)

def _read_schedule():
    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {"current_day_index": 0, "history": []}

def _write_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

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
    1. NO humans, NO children, NO people.
    2. Focus on: Architecture, Nature, Minimalist Objects, or Light.
    3. Use words like: 'Growth', 'Structure', 'Clear Horizon', 'Polished Glass', 'Morning Sun'.
    4. Format: 1 short sentence of descriptive keywords.
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
        return r.json()["choices"][0]["message"]["content"].strip()
    except:
        return topic

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
    _write_state(state)
    
    os.chdir(str(FORWILMA_DIR))
    schedule = _read_schedule()
    
    if state["current_day_index"] >= len(schedule):
        print("🎉 Schedule complete! Restarting...")
        state["current_day_index"] = 0

    post_data = schedule[state["current_day_index"]]
    day_num = post_data["day"]
    
    print(f"🚀 Processing Day {day_num} for Digital Guardian (Wilma)...")

    # --- 1. MEDIA GENERATION (ONCE) ---
    image_path = os.path.join("images", f"day{day_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    try:
        # Turn literal topic into a safe visual metaphor
        visual_metaphor = _generate_wilma_visual_prompt(post_data['topic'])
        print(f"  Visual Metaphor: {visual_metaphor}")
        
        image_prompt = f"{WILMA_BRAND_BASE}, {visual_metaphor}, {WILMA_BRAND_SUFFIX}"
        raw_image = generate_image(image_prompt)
        processed = _write_output_jpg(raw_image, "output.jpg")
        
        # Add Logo
        add_logo_watermark("output.jpg", str(LOGO_PATH))
        
        # Move to persistent storage
        shutil.copy("output.jpg", image_path)
        print(f"✓ Image saved and watermarked.")
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        sys.exit(1)

    # --- THE MASTER REFLECTION (Wilma Style) ---
    print("Generating Master Reflection for Wilma...")
    master_system = f"You are the lead strategist for Digital Guardian. Mission: {DIGITAL_GUARDIAN_MISSION}. Write a professional, empathetic, and insightful post about the topic below. No length limit."
    master_reflection = _generate_text_ai_horde(f"Topic: {post_data['topic']}, Audience: {post_data['audience']}", system_prompt=master_system)
    print(f"✓ Master Reflection acquired.")

    # --- 2. CAPTION GENERATION (AI CRITIC EDITS) ---
    bundle = {}

    for p in platforms:
        print(f"  Tailoring for {p.upper()}...")
        try:
            # We give a tighter limit for Bluesky (240) to leave room for the referral CTA
            max_c = 240 if p == "bluesky" else 2000
            
            # Use the AI Critic to re-purpose the master reflection
            tailored_cap = _ai_verify_caption(master_reflection, p, max_c)
            final_cap = _clean_caption_formatting(tailored_cap)
            
            # Platform Specific assembly
            if p == "linkedin":
                 final_cap += "\n\n#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips"
            elif p == "bluesky":
                 # Use the specific CTA requested by the user
                 final_cap += "\n\nWant to read more?... check out my LinkedIn"

            bundle[p] = final_cap
            
            if args.mode == "single":
                with open("caption.txt", "w", encoding="utf-8") as f:
                    f.write(final_cap)
        except Exception as e:
            print(f"❌ Tailoring failed for {p}: {e}")

    # Save bundle
    with open("wilma_bundle.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    # --- 3. CREATE READY FLAGS (Wilma Style) ---
    if args.mode == "generate_all":
        for p in platforms:
            flag_name = f"wilma_{p}_ready.flag"
            with open(flag_name, "w") as f:
                f.write(datetime.now().isoformat())
            print(f"🚩 Wilma Flag created: {flag_name}")

    # Success! Advance the schedule
    state["current_day_index"] += 1
    state["history"].append({
        "day": day_num,
        "platforms": platforms,
        "timestamp": datetime.now().isoformat(),
        "image": image_path
    })
    _write_state(state)
    print(f"✅ Day {day_num} complete. Assets bundled in wilma_bundle.json")

if __name__ == "__main__":
    main()
