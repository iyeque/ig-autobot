import os
import sys
import json
import time
import shutil
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
    state = _read_state()
    _write_state(state) # Initialize immediately
    
    os.chdir(str(FORWILMA_DIR))
    schedule = _read_schedule()
    
    if state["current_day_index"] >= len(schedule):
        print("🎉 Schedule complete! Restarting...")
        state["current_day_index"] = 0

    post_data = schedule[state["current_day_index"]]
    day_num = post_data["day"]
    
    print(f"🚀 Processing Day {day_num} for Digital Guardian (Wilma)...")

    # 1. Refined Caption Generation (Curiosity + Social Proof + Promised Benefit + CTA)
    system_identity = f"""You are the lead strategist for Digital Guardian, a professional digital safety platform.
Your mission: {DIGITAL_GUARDIAN_MISSION}
Tone: Empathetic, expert, and professional.

CAPTION FORMULA:
1. CURIOSITY: Start with a hook that makes parents stop scrolling.
2. SOCIAL PROOF: Mention how "hundreds of families" or "proactive parents" are doing this.
3. PROMISED BENEFIT: What will they gain (peace of mind, closer bonds, safety).
4. CTA: Clear instruction to comment or save."""

    prompt = f"""Write a professional LinkedIn post for {post_data['audience']}. 
Topic: '{post_data['topic']}'. Type: '{post_data['type']}'. 
Follow the formula: Curiosity + Social Proof + Promised Benefit + CTA.
Include #DigitalGuardian #DigitalParenting #ScreenTime #CyberSafety."""
    
    try:
        caption = generate_caption(prompt, system_prompt=system_identity)
        with open("caption.txt", "w", encoding="utf-8") as f:
            f.write(caption)
        print("✅ Caption ready.")
    except Exception as e:
        print(f"❌ Caption failed: {e}")
        return

    # 2. Faithful Image Prompt (Based on Schedule Description)
    graphics_direction = post_data['graphics']
    
    # Enrich the schedule description with brand aesthetics
    safe_image_prompt = (
        f"{graphics_direction}, {WILMA_BRAND_BASE}, {WILMA_BRAND_SUFFIX}"
    )
    
    try:
        print(f"Generating image based on schedule: {graphics_direction[:60]}...")
        raw_path = generate_image(safe_image_prompt)
        processed_path = _write_output_jpg(raw_path, "output.jpg")
        
        # Add Logo Watermark
        if LOGO_PATH.exists():
            add_logo_watermark(processed_path, str(LOGO_PATH))
            print("✓ Logo added.")

        # Add a BOLD professional text overlay of the topic
        add_static_text_overlay(processed_path, post_data['topic'])
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"day{day_num}_{timestamp}.jpg"
        shutil.copy(processed_path, Path("images") / archive_name)
        print(f"✅ Graphics ready: {archive_name}")

    except Exception as e:
        print(f"❌ Graphics failed: {e}")
        return

    # 3. Update Progress
    state["current_day_index"] += 1
    state["history"].append({
        "day": day_num,
        "date": datetime.now().isoformat(),
        "topic": post_data["topic"]
    })
    _write_state(state)
    print(f"✅ Day {day_num} complete.")

if __name__ == "__main__":
    main()
