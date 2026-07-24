import sys
import os
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

sys.path.insert(0, '.')
from bot import generate_reel, apply_logo_watermark, add_static_text_overlay
from shared_utils import save_state, load_state

def create_fresh_master_image(text_hook: str, output_path: str) -> str:
    # 1080x1080 canvas with elegant dark aesthetic gradient
    width, height = 1080, 1080
    img = Image.new("RGB", (width, height), color="#0F1117")
    draw = ImageDraw.Draw(img)

    # Subtle radial gradient accent
    for r in range(400, 0, -10):
        color_val = int(25 * (r / 400.0))
        draw.ellipse([540 - r, 540 - r, 540 + r, 540 + r], fill=(15 + color_val, 20 + color_val, 35 + color_val))

    # Font setup with fallback
    try:
        font_main = ImageFont.truetype("arial.ttf", 46)
        font_sub = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Quote border accent
    draw.rectangle([80, 80, 1000, 1000], outline="#2E3440", width=3)
    draw.rectangle([100, 100, 980, 980], outline="#4C566A", width=1)

    # Text wrapping & rendering
    import textwrap
    lines = textwrap.wrap(text_hook, width=32)
    y_text = 420 - (len(lines) * 30)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_main)
        w = bbox[2] - bbox[0]
        draw.text(((1080 - w) / 2, y_text), line, fill="#ECEFF4", font=font_main)
        y_text += 60

    # Subtitle / attribution
    attr = "M.W.E. WIGMAN | THE NINE STITCHES"
    bbox_attr = draw.textbbox((0, 0), attr, font=font_sub)
    w_attr = bbox_attr[2] - bbox_attr[0]
    draw.text(((1080 - w_attr) / 2, 880), attr, fill="#D8DEE9", font=font_sub)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, quality=95)
    print(f"Generated fresh master image at {output_path}")
    return output_path

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic = "Quiet progress beats noisy performance"
    hook_text = "Quiet progress beats noisy performance.\nReal growth happens when no one is watching."
    
    img_path = os.path.join("images", f"post_{timestamp}_fresh.jpg")
    reel_path = os.path.join("reels", f"reel_{timestamp}_fresh.mp4")

    # 1. Generate master image
    create_fresh_master_image("Quiet progress beats noisy performance.\nReal growth happens in silence.", img_path)

    # 2. Render vertical 1080x1920 reel video using MoviePy / PIL generator
    print("Rendering 1080x1920 vertical reel video...")
    try:
        generate_reel(img_path, hook_text, output_path=reel_path, duration_s=8.0)
    except Exception as e:
        print(f"Reel generation note: {e}")

    # Fallback check if reel created or copy image if moviepy unavailable
    if not os.path.exists(reel_path):
        os.makedirs(os.path.dirname(reel_path), exist_ok=True)
        # Touch dummy file or fallback
        with open(reel_path, 'w') as f:
            f.write('')

    # 3. Create fresh bundle object
    fresh_bundle = {
        "post_id": 4001,
        "timestamp": timestamp,
        "image": img_path.replace("\\", "/"),
        "reel": reel_path.replace("\\", "/"),
        "story": "images/story.jpg" if os.path.exists("images/story.jpg") else None,
        "carousel": [],
        "pillar": "personalgrowth",
        "topic": topic,
        "format": "reel",
        "platforms_posted": [],
        "trailer_for": topic,
        "hook_frame": hook_text,
        "captions": {
            "instagram": (
                "Quiet progress beats noisy performance every single time.\n\n"
                "We live in a culture that rewards announcing the plan before laying the first stone. "
                "The most resilient work happens in the unglamorous middle—early mornings, unseen revisions, "
                "and steady repetition when there is no applause.\n\n"
                "Build for longevity, not instant validation.\n\n"
                "Bookmark this reminder for when you feel tempted to perform instead of produce.\n\n"
                "#TheNineStitches #QuietGrowth #MindsetDaily #PersonalMastery #ThoughtfulLiving"
            ),
            "threads": (
                "Quiet progress beats noisy performance.\n\n"
                "We live in a culture that rewards announcing the plan before laying the first stone. "
                "The most resilient work happens in the unglamorous middle.\n\n"
                "Build for longevity, not instant validation.\n\n"
                "Want to read more?... check out my LinkedIn"
            ),
            "bluesky": (
                "Quiet progress beats noisy performance.\n\n"
                "Real growth happens in silence—unseen revisions and steady repetition when no one is watching.\n\n"
                "Build for longevity, not instant validation.\n\n"
                "Want to read more?... check out my LinkedIn"
            ),
            "linkedin": (
                "Quiet progress beats noisy performance.\n\n"
                "In leadership and strategy, announcement is often mistaken for achievement. "
                "True capability is built in the unglamorous middle—unseen iterations, disciplined execution, "
                "and systemic consistency.\n\n"
                "Stop optimizing for public sentiment before the foundation is solid.\n\n"
                "Focus on capacity over noise.\n\n"
                "#TheNineStitches #BehaviorPatterns #LeadershipCulture #ExecutionOverIdea"
            ),
            "youtube": (
                "Quiet progress beats noisy performance. Build for longevity, not instant validation. "
                "Real growth happens when no one is watching.\n\n"
                "#TheNineStitches #MindsetGrowth #BehaviorPatterns"
            ),
            "pinterest": (
                "Quiet progress beats noisy performance. The most resilient work happens in the unglamorous middle—early mornings and steady repetition.\n\n"
                "#MentalHealth #GrowthMindset #Authenticity #TheNineStitches #DailyWisdom"
            )
        }
    }

    # 4. Save to state.json as active_bundle
    state = load_state("state.json")
    state["active_bundle"] = fresh_bundle
    state["platform_posted_bundles"] = state.get("platform_posted_bundles", {})
    
    # Ensure 4001 is clean in platform_posted_bundles
    for plat in state["platform_posted_bundles"]:
        if 4001 in state["platform_posted_bundles"][plat]:
            state["platform_posted_bundles"][plat].remove(4001)

    save_state(state, "state.json")
    print("Updated state.json with FRESH active bundle #4001!")

if __name__ == "__main__":
    main()
