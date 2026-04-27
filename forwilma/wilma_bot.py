import os
import sys
import json
import time
import argparse
import shutil
from datetime import datetime
from pathlib import Path

# Important: Setup paths BEFORE importing core logic
BASE_DIR = Path(__file__).parent.parent
FORWILMA_DIR = Path(__file__).parent

# Add parent directory to path so we can import from the main bot
sys.path.append(str(BASE_DIR))

# Import core logic from the main bot
try:
    from bot import (
        generate_caption, 
        generate_image, 
        _write_output_jpg, 
        add_static_text_overlay,
        generate_reel,
        generate_story_image
    )
except ImportError:
    print("❌ Error: Could not import core logic from bot.py. Ensure you are running from the project root.")
    sys.exit(1)

# Wilma Specific Config
SCHEDULE_FILE = FORWILMA_DIR / "schedule.json"
STATE_FILE = FORWILMA_DIR / "state.json"
WILMA_IMAGES_DIR = FORWILMA_DIR / "images"
WILMA_REELS_DIR = FORWILMA_DIR / "reels"

# Ensure directories exist
WILMA_IMAGES_DIR.mkdir(exist_ok=True)
WILMA_REELS_DIR.mkdir(exist_ok=True)

def _read_schedule():
    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"current_day_index": 0, "history": []}

def _write_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def main():
    # Change working directory to forwilma to keep outputs local
    os.chdir(str(FORWILMA_DIR))
    print(f"📍 Working directory switched to: {os.getcwd()}")

    state = _read_state()
    schedule = _read_schedule()
    
    if state["current_day_index"] >= len(schedule):
        print("🎉 30-day schedule complete! Restarting from Day 1...")
        state["current_day_index"] = 0

    post_data = schedule[state["current_day_index"]]
    day_num = post_data["day"]
    
    print(f"🚀 Processing Day {day_num} for Wilma...")
    print(f"Topic: {post_data['topic']} | Audience: {post_data['audience']}")

    # 1. Generate LinkedIn-specific Caption
    prompt = (
        f"Write a professional LinkedIn post for an audience of {post_data['audience']}. "
        f"The topic is: '{post_data['topic']}'. "
        f"The content type should be an '{post_data['type']}'. "
        f"End with this Call to Action: {post_data['cta']}. "
        f"Tone: Empathetic, authoritative, and helpful. Use line breaks for readability. "
        f"Include 3-5 relevant hashtags like #Parenting #ScreenTime #DigitalWellbeing."
    )
    
    try:
        print("Generating LinkedIn caption...")
        caption = generate_caption(prompt, book_context="", book_insights=None)
        
        with open("caption.txt", "w", encoding="utf-8") as f:
            f.write(caption)
        print("✅ Caption saved to forwilma/caption.txt")
    except Exception as e:
        print(f"❌ Caption generation failed: {e}")
        return

    # 2. Generate Graphics
    image_prompt = (
        f"{post_data['graphics']}, high quality, professional lighting, "
        f"soft colors, clean composition, minimalist aesthetic, 4k"
    )
    
    try:
        print(f"Generating image: {image_prompt}")
        raw_path = generate_image(image_prompt)
        
        # Local output path inside forwilma
        processed_path = _write_output_jpg(raw_path, "output.jpg")
        
        if post_data['type'] in ["Insight", "Authority", "Community"]:
            add_static_text_overlay(processed_path, post_data['topic'])
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"day{day_num}_{timestamp}.jpg"
        archive_path = Path("images") / archive_name
        shutil.copy(processed_path, archive_path)
        
        print(f"✅ Image saved and archived to forwilma/images/{archive_name}")

        if post_data['type'] == "Short Video":
            print("Generating Video placeholder...")
            reel_name = f"day{day_num}_{timestamp}.mp4"
            generate_reel(processed_path, post_data['topic'], "reel.mp4", duration_s=6.0)
            shutil.copy("reel.mp4", Path("reels") / reel_name)
            print(f"✅ Video archived to forwilma/reels/{reel_name}")

    except Exception as e:
        print(f"❌ Graphics generation failed: {e}")
        return

    # 3. Update State
    state["current_day_index"] += 1
    state["history"].append({
        "day": day_num,
        "date": datetime.now().isoformat(),
        "topic": post_data["topic"]
    })
    _write_state(state)
    print(f"✅ Day {day_num} complete. Progress saved.")

if __name__ == "__main__":
    main()
