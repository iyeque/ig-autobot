import os
import sys
import json
import time
import argparse
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
        generate_reel
    )
except ImportError:
    print("❌ Error: Could not import core logic from bot.py.")
    sys.exit(1)

# Wilma Specific Config
SCHEDULE_FILE = FORWILMA_DIR / "schedule.json"
STATE_FILE = FORWILMA_DIR / "state.json"
WILMA_IMAGES_DIR = FORWILMA_DIR / "images"
WILMA_REELS_DIR = FORWILMA_DIR / "reels"

# Ensure directories exist
WILMA_IMAGES_DIR.mkdir(exist_ok=True)
WILMA_REELS_DIR.mkdir(exist_ok=True)

# WILMA BRAND SETTINGS (Safe & Professional)
WILMA_BRAND_BASE = (
    "clean professional photography, soft natural daylight, bright and airy, "
    "high key lighting, minimalist composition, pastel color palette, 8k resolution"
)
WILMA_BRAND_SUFFIX = (
    "no humans, no faces, no text, clean textures, soft focus background, high quality"
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
    # Default state if file doesn't exist or is empty
    return {"current_day_index": 0, "history": []}

def _write_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def main():
    # 1. Initialize State immediately (Fixes Git pathspec error)
    state = _read_state()
    _write_state(state)
    
    # Switch directory
    os.chdir(str(FORWILMA_DIR))
    
    schedule = _read_schedule()
    
    if state["current_day_index"] >= len(schedule):
        print("🎉 30-day schedule complete! Restarting...")
        state["current_day_index"] = 0

    post_data = schedule[state["current_day_index"]]
    day_num = post_data["day"]
    
    print(f"🚀 Processing Day {day_num} for Wilma...")

    # 2. Generate Caption
    prompt = (
        f"Write a professional LinkedIn post for {post_data['audience']}. "
        f"Topic: '{post_data['topic']}'. Type: '{post_data['type']}'. "
        f"CTA: {post_data['cta']}. Tone: Empathetic & Helpful. #Parenting #DigitalWellbeing"
    )
    
    try:
        caption = generate_caption(prompt, book_context="", book_insights=None)
        with open("caption.txt", "w", encoding="utf-8") as f:
            f.write(caption)
        print("✅ Caption ready.")
    except Exception as e:
        print(f"❌ Caption failed: {e}")
        return

    # 3. Generate Graphics with "Safe Brand" (Fixes NSFW/Censorship)
    # We explicitly construct a prompt that avoids "Moody/Dark" keywords
    safe_image_prompt = (
        f"{WILMA_BRAND_BASE}, {post_data['graphics']}, {WILMA_BRAND_SUFFIX}"
    )
    
    try:
        print(f"Generating safe image: {post_data['topic']}")
        # We override the main bot's brand mode by passing a very specific prompt
        raw_path = generate_image(safe_image_prompt)
        processed_path = _write_output_jpg(raw_path, "output.jpg")
        
        if post_data['type'] in ["Insight", "Authority", "Community"]:
            add_static_text_overlay(processed_path, post_data['topic'])
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"day{day_num}_{timestamp}.jpg"
        shutil.copy(processed_path, Path("images") / archive_name)
        print(f"✅ Graphics ready: {archive_name}")

    except Exception as e:
        print(f"❌ Graphics failed: {e}")
        # We don't increment day index if image fails
        return

    # 4. Success - Update Progress
    state["current_day_index"] += 1
    state["history"].append({
        "day": day_num,
        "date": datetime.now().isoformat(),
        "topic": post_data["topic"]
    })
    _write_state(state)
    print(f"✅ Day {day_num} saved to state.json")

if __name__ == "__main__":
    main()
