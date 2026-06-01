import os
import sys
import json
import time
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent
FORWILMA_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

# Import core logic
try:
    from bot import (
        generate_caption, 
        generate_image, 
        _write_output_jpg, 
        add_static_text_overlay,
        generate_reel,
        add_logo_watermark
    )
except ImportError:
    print("❌ Error: Could not import core logic from bot.py.")
    sys.exit(1)

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

def main():
    parser = argparse.ArgumentParser(description="Digital Guardian (Wilma) Bot")
    parser.add_argument("--platform", type=str, default="linkedin", choices=["linkedin", "bluesky"],
                      help="Target platform")
    args = parser.parse_args()
    platform = args.platform

    state = _read_state()
    _write_state(state) # Initialize immediately
    
    os.chdir(str(FORWILMA_DIR))
    schedule = _read_schedule()
    
    if state["current_day_index"] >= len(schedule):
        print("🎉 Schedule complete! Restarting...")
        state["current_day_index"] = 0

    post_data = schedule[state["current_day_index"]]
    day_num = post_data["day"]
    
    print(f"🚀 Processing Day {day_num} for Digital Guardian (Wilma) on {platform.upper()}...")

    # 1. Refined Caption Generation
    system_identity = f"""You are the lead strategist for Digital Guardian, a professional digital safety platform.
Your mission: {DIGITAL_GUARDIAN_MISSION}
Tone: Empathetic, expert, and professional."""

    # Add character limits for Bluesky
    max_chars = 180 if platform == "bluesky" else 2000
    
    prompt = f"""Write a professional post for {platform.upper()} regarding {post_data['audience']}. 
Topic: '{post_data['topic']}'. Type: '{post_data['type']}'. 
{f'LIMIT: {max_chars} characters.' if platform == 'bluesky' else 'Formula: Hook + Body + CTA.'}
Include #DigitalGuardian #DigitalParenting."""

    if platform == "bluesky":
        prompt += " BE EXTREMELY CONCISE. No hashtags."
    
    try:
        caption = generate_caption(prompt, platform=platform, system_prompt=system_identity)
        with open("caption.txt", "w", encoding="utf-8") as f:
            f.write(caption)
        print("✓ Caption generated.")
    except Exception as e:
        print(f"❌ Caption generation failed: {e}")
        sys.exit(1)

    # 2. Image Generation (Only once per post index, reuse for multiple runs if needed)
    image_path = os.path.join("images", f"day{day_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    try:
        image_prompt = f"{WILMA_BRAND_BASE}, {post_data['topic']}, {WILMA_BRAND_SUFFIX}"
        raw_image = generate_image(image_prompt)
        processed = _write_output_jpg(raw_image, "output.jpg")
        
        # Add Logo
        add_logo_watermark("output.jpg", str(LOGO_PATH))
        
        # Move to persistent storage
        shutil.copy("output.jpg", image_path)
        print(f"✓ Image saved to {image_path}")
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        sys.exit(1)

    # Success! Advance the schedule
    state["current_day_index"] += 1
    state["history"].append({
        "day": day_num,
        "platform": platform,
        "timestamp": datetime.now().isoformat(),
        "image": image_path
    })
    _write_state(state)
    print(f"✅ Day {day_num} complete. Advanced to index {state['current_day_index']}")

if __name__ == "__main__":
    main()
