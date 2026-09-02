import sys, json, os, shutil, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '..')
from wilma_bot import _generate_wilma_visual_prompt, generate_image, _write_output_jpg, apply_logo_watermark, add_static_text_overlay, _save_pending, STATE_FILE, LOGO_PATH, WILMA_BRAND_BASE, WILMA_BRAND_SUFFIX

state = json.load(open('state.json'))
schedule = json.load(open('schedule.json'))
day_data = schedule[3]
print('Topic:', day_data['topic'])

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_name = f"day4_repub_{timestamp}.jpg"
image_path = f"images/{image_name}"

visual_metaphor = _generate_wilma_visual_prompt(day_data["topic"])
image_prompt = f"{WILMA_BRAND_BASE}, {visual_metaphor}, {WILMA_BRAND_SUFFIX}"
print('Prompt:', image_prompt)

raw_image = generate_image(image_prompt)
processed = _write_output_jpg(raw_image, "temp_output.jpg")
apply_logo_watermark("temp_output.jpg", str(LOGO_PATH))
add_static_text_overlay("temp_output.jpg", day_data["topic"])
shutil.copy("temp_output.jpg", image_path)
print('Saved:', image_path)

pending = {
    "post_id": "day_4",
    "timestamp": timestamp,
    "post": day_data,
    "image": image_path,
    "master_reflection": None,
    "bundle_captions": {},
    "carousel": [],
    "platforms_posted": [],
    "platforms_prepared": [],
}
_save_pending(state, pending)
print('Pending bundle saved.')
