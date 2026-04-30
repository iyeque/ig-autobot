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
        add_static_text_overlay
    )
except ImportError:
    print("❌ Error: Could not import core logic from bot.py.")
    sys.exit(1)

# Guardd / Wilma Specific Config
SCHEDULE_FILE = FORWILMA_DIR / "schedule.json"
STATE_FILE = FORWILMA_DIR / "state.json"
WILMA_IMAGES_DIR = FORWILMA_DIR / "images"

# Mission Context for AI
# Redefine GUARDD_MISSION as a single-line string to simplify embedding in f-strings
GUARDD_MISSION = (
    "Guardd is a digital safety platform that simplifies parenting. "
    "Mission: Bridge the gap between children's online exploration and well-being. "
    "Values: Healthy digital habits, fostering family bonds, open conversations, and proactive safety."
)

# WILMA BRAND SETTINGS (Safe, Trustworthy, Modern)
WILMA_BRAND_BASE = (
    "modern minimalist fine-art photography, soft ethereal lighting, warm and safe atmosphere, "
    "professional corporate aesthetic, clean geometric metaphors, pastel blue and soft teal accents"
)
WILMA_BRAND_SUFFIX = (
    "no humans, no faces, no text, ultra-sharp detail, 8k resolution, minimalist style"
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
    
    print(f"🚀 Processing Day {day_num} for Guardd (Wilma)...")

    # 1. Refined Caption Generation (Using Guardd Context)
    prompt = f"""Context: {GUARDD_MISSION}
Write a professional LinkedIn post for {post_data['audience']}. 
Topic: '{post_data['topic']}'. Type: '{post_data['type']}'. 
CTA: {post_data.get('cta', '')}. Tone: Empathetic, expert, and proactive. 
Include #Guardd #DigitalParenting #ScreenTime #CyberSafety."""
    
    try:
        caption = generate_caption(prompt, book_context="", book_insights=None)
        with open("caption.txt", "w", encoding="utf-8") as f:
            f.write(caption)
        print("✅ Caption ready.")
    except Exception as e:
        print(f"❌ Caption failed: {e}")
        return

    # 2. Refined Image Prompt (Abstract Metaphors to avoid Censorship)
    # Instead of literal "Screens", we use safe metaphors like "guiding light" or "woven protection"
    graphics_direction = post_data['graphics'].lower()
    metaphor = "abstract representation of digital safety"
    
    if "tutorial" in graphics_direction or "step-by-step" in graphics_direction:
        metaphor = "a gentle glowing path leading through soft blue clouds, symbolic of guidance"
    elif "mistakes" in graphics_direction or "boundaries" in graphics_direction:
        metaphor = "soft translucent interlocking geometric layers, symbolic of protection and structure"
    elif "lifestyle" in graphics_direction or "fix" in graphics_direction:
        metaphor = "two warm lights blending together in a calm space, symbolic of family connection"
    elif "diagram" in graphics_direction or "causes" in graphics_direction:
        metaphor = "minimalist ripple effects on calm water, clean and organized composition"
    
    safe_image_prompt = (
        f"{WILMA_BRAND_BASE}, {metaphor}, {WILMA_BRAND_SUFFIX}"
    )
    
    try:
        print(f"Generating safe visual metaphor: {metaphor[:50]}...")
        raw_path = generate_image(safe_image_prompt)
        processed_path = _write_output_jpg(raw_path, "output.jpg")
        
        # Add a professional text overlay of the topic
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
