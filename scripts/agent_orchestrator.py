#!/usr/bin/env python3
"""
Agent Orchestrator for ig-autobot.

Reads specialist agents from agents/ and uses their instructions as LLM context
to generate content bundles. Replaces the monolithic template-based generation
in bot.py with multi-agent orchestration.

Usage:
    python scripts/agent_orchestrator.py --brand main --dry-run
    python scripts/agent_orchestrator.py --brand wilma --day 12
    python scripts/agent_orchestrator.py --brand main --output state.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO / "agents"
FORWILMA_DIR = REPO / "forwilma"
STATE_PATH = REPO / "state.json"
WILMA_STATE_PATH = FORWILMA_DIR / "state.json"
SCHEDULE_PATH = FORWILMA_DIR / "schedule.json"
IMAGES_DIR = REPO / "images"
WILMA_IMAGES_DIR = FORWILMA_DIR / "images"

# ── Brand-specific constants ─────────────────────────────────────────────
BRAND = {
    "main": {
        "name": "WP WIGMAN",
        "book_title": "The Nine Stitches",
        "book_author": "M.W.E. WIGMAN",
        "hashtag": "#TheNineStitches",
        "persona": "Professional Failure Expert",
        "state_path": STATE_PATH,
        "images_dir": IMAGES_DIR,
        "platforms": ["instagram", "linkedin", "pinterest", "youtube", "threads", "bluesky"],
    },
    "wilma": {
        "name": "DigitalGuard",
        "book_title": "",
        "book_author": "",
        "hashtag": "#DigitalGuard",
        "persona": "Digital wellness guide for families",
        "state_path": WILMA_STATE_PATH,
        "images_dir": WILMA_IMAGES_DIR,
        "platforms": ["linkedin", "bluesky"],
    },
}


# ── Agent loading ────────────────────────────────────────────────────────
def load_agent(slug: str) -> dict | None:
    """Load one agent by slug from agents/ directory."""
    for path in AGENTS_DIR.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        parts = text.split("---", 2)
        if len(parts) < 3:
            continue
        fm = {}
        for line in parts[1].splitlines():
            if ":" in line and not line.startswith((" ", "\t")):
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip().strip('"').strip("'")
        name = fm.get("name", "")
        agent_slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") if name else ""
        if agent_slug == slug:
            return {
                "slug": agent_slug,
                "name": name,
                "description": fm.get("description", ""),
                "vibe": fm.get("vibe", ""),
                "body": parts[2].lstrip("\n"),
            }
    return None


def load_all_agents() -> list[dict]:
    """Load all agent .md files."""
    agents = []
    for path in sorted(AGENTS_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        parts = text.split("---", 2)
        if len(parts) < 3:
            continue
        fm = {}
        for line in parts[1].splitlines():
            if ":" in line and not line.startswith((" ", "\t")):
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip().strip('"').strip("'")
        name = fm.get("name", "")
        if not name:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        agents.append({
            "slug": slug,
            "name": name,
            "description": fm.get("description", ""),
            "vibe": fm.get("vibe", ""),
            "body": parts[2].lstrip("\n"),
        })
    return agents


# ── LLM helpers ──────────────────────────────────────────────────────────
def _load_env():
    """Load .env file if present."""
    env_path = REPO / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def llm_call(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """
    Call the LLM using AI Horde (same backend as bot.py).
    Falls back to a stub if no API key is available.
    """
    _load_env()
    api_key = os.environ.get("AI_HORDE_API_KEY", "")
    if not api_key:
        return ""  # No key — return empty for caller to handle

    try:
        import requests

        # AI Horde text generation endpoint
        url = "https://aihorde.net/api/v2/generate/text/async"
        headers = {"apikey": api_key, "Content-Type": "application/json"}
        payload = {
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "max_length": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()

        # Poll for completion (simplified)
        if "id" in result:
            poll_url = f"https://aihorde.net/api/v2/generate/text/status/{result['id']}"
            for _ in range(20):  # max ~100s
                import time
                time.sleep(5)
                pr = requests.get(poll_url, headers={"apikey": api_key}, timeout=15)
                pr.raise_for_status()
                pdata = pr.json()
                if pdata.get("done"):
                    generations = pdata.get("generations", [])
                    if generations:
                        return generations[0].get("text", "")
                    return ""
        return ""
    except Exception as e:
        print(f"  ⚠ LLM call failed: {e}")
        return ""


# ── Content generation via agents ───────────────────────────────────────
def generate_topic_with_content_creator(brand: dict, day: int | None = None, stub: bool = False) -> dict:
    """Use Content Creator agent to pick a fresh topic."""
    if stub:
        return {
            "pillar": "micro_philosophy",
            "title": "The Quiet Rebellion of Enough",
            "topic": "In a world that demands more — more speed, more output, more perfection — what if the most radical act is to stop?",
        }
    agent = load_agent("content-creator") or load_agent("marketing-content-creator")
    system = f"""You are a content strategist for {brand['name']}.
{agent['body'] if agent else ''}

Pick ONE specific, fresh topic for today's social media post.
The topic should be a single, specific idea — not broad.
Return ONLY a JSON object: {{"pillar": "...", "title": "...", "topic": "..."}}
Where pillar is one of: micro_philosophy, nature_metaphor, systems_psychology, author_voice, quote
and title is the post title (max 8 words) and topic is a one-sentence description."""

    user = f"Generate a unique topic for day {day or 'today'}. Avoid repeating common themes."
    result = llm_call(system, user, max_tokens=256)
    try:
        data = json.loads(result)
        return data
    except Exception:
        pass
    # Fallback
    return {
        "pillar": "micro_philosophy",
        "title": "The Art of Starting Over",
        "topic": "productive failure and wabi-sabi in the age of algorithms",
    }


def generate_stub_caption(platform: str, topic: dict, master_reflection: str, brand: dict) -> str:
    """Generate a stub caption for dry-run testing (no LLM)."""
    title = topic.get("title", "")
    ref = (master_reflection or topic.get("topic", ""))[:150].strip()
    hashtag = brand["hashtag"]

    if platform == "linkedin":
        return (
            f"I used to think more was the answer — more effort, more polish, more output.\n\n"
            f"Then I learned about kintsugi: the Japanese art of repairing broken pottery with gold.\n\n"
            f"The break is not a flaw. It becomes the most valuable part.\n\n"
            f"{ref}\n\n"
            f"What if your cracks are not bugs to fix but features to frame?\n\n"
            f"{hashtag} #WabiSabi"
        )
    elif platform == "instagram":
        return (
            f"The gold repair lines tell the story.\n\n"
            f"{ref}\n\n"
            f"{hashtag} #TheNineStitches"
        )
    elif platform == "threads":
        return (
            f"The gold repair lines tell the story.\n\n"
            f"{ref}"
        )
    elif platform == "bluesky":
        return (
            f"The gold repair lines tell the story.\n\n"
            f"{ref}"
        )
    elif platform == "pinterest":
        return (
            f"The Japanese art of kintsugi: repairing broken pottery with gold. "
            f"The break becomes the most valuable part.\n\n"
            f"{hashtag}"
        )
    elif platform == "youtube":
        return (
            f"What if your cracks are not bugs to fix but features to frame?\n\n"
            f"{ref}\n\n"
            f"{hashtag}"
        )
    else:
        return f"{title}\n\n{ref}\n\n{hashtag}"


def generate_image_prompt(topic: dict, brand: dict) -> str:
    """Use Image Prompt Engineer agent to craft a detailed image prompt."""
    agent = load_agent("image-prompt-engineer") or load_agent("design-image-prompt-engineer")
    system = f"""You are an Image Prompt Engineer.
{agent['body'] if agent else ''}

Craft a detailed, evocative prompt for AI image generation.
The prompt should be 50-150 words, highly specific, and include:
subject, environment, lighting, style, and technical specs.
Return ONLY the prompt text — no explanation."""

    user = f"Create an image prompt for: {topic.get('title', '')} — {topic.get('topic', '')}\nBrand: {brand['name']}\nStyle: dark, textured, gold accents, cinematic."
    return llm_call(system, user, max_tokens=512) or "dark background, textured surface, gold kintsugi lines, cinematic lighting"


def generate_caption_for_platform(
    platform: str,
    topic: dict,
    master_reflection: str,
    brand: dict,
) -> str:
    """Use platform-specific agent to generate a caption."""
    # Pick the right agent for the platform
    if platform == "linkedin":
        agent = load_agent("linkedin-content-creator") or load_agent("marketing-linkedin-content-creator")
    elif platform == "instagram":
        agent = load_agent("instagram-curator") or load_agent("marketing-instagram-curator")
    elif platform in ("bluesky", "threads"):
        agent = load_agent("twitter-engager") or load_agent("marketing-twitter-engager")
    elif platform == "pinterest":
        agent = load_agent("seo-specialist") or load_agent("marketing-seo-specialist")
    elif platform == "youtube":
        agent = load_agent("video-optimization-specialist") or load_agent("marketing-video-optimization-specialist")
    else:
        agent = load_agent("content-creator") or load_agent("marketing-content-creator")

    limits = {
        "bluesky": 250, "threads": 450, "instagram": 1400,
        "linkedin": 1800, "pinterest": 450, "youtube": 400,
    }
    max_c = limits.get(platform, 1800)

    system = f"""You are a {platform.upper()} content specialist for {brand['name']}.
{agent['body'] if agent else ''}

Write a {platform} post about: {topic.get('title', '')}
Master reflection: {master_reflection[:500] if master_reflection else topic.get('topic', '')}

Rules:
- Stay under {max_c} characters
- Use brand hashtag {brand['hashtag']}
- Be specific, never vague
- Have a point of view
- End with a question or CTA"""

    user = f"Write a {platform} caption for: {topic.get('title', '')}"
    return llm_call(system, user, max_tokens=max_c) or f"{topic.get('title', '')}\n\n{topic.get('topic', '')}"


def generate_master_reflection(topic: dict, brand: dict) -> str:
    """Generate a master reflection using Content Creator + brand persona."""
    agent = load_agent("content-creator") or load_agent("marketing-content-creator")
    system = f"""You are the '{brand['persona']}' persona for {brand['name']}.
Write a deep, witty, and cynical reflection on the topic below.

Style rules:
- If the content naturally connects to {brand.get('book_title', '')}, plant a subtle nod — never a hard sales pitch.
- Let ideas breathe. Do not summarize or truncate.

{agent['body'] if agent else ''}"""

    user = f"Topic: {topic.get('title', '')} — {topic.get('topic', '')}"
    return llm_call(system, user, max_tokens=2048) or f"Reflection on {topic.get('title', '')}."


# ── State management ─────────────────────────────────────────────────────
def load_state(path: Path) -> dict:
    """Load state.json."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(path: Path, state: dict):
    """Write state.json atomically."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def get_next_post_id(state: dict) -> int:
    """Get the next post ID."""
    used = state.get("used_ids", [])
    # Filter to only integer IDs (skip Wilma-style "day_N" strings)
    int_ids = [i for i in used if isinstance(i, int)]
    if int_ids:
        return max(int_ids) + 1
    pending = state.get("pending_bundle", {})
    if pending and pending.get("post_id"):
        return pending["post_id"] + 1
    active = state.get("active_bundle", {})
    if active and active.get("post_id"):
        return active["post_id"] + 1
    return 299


def get_next_wilma_day(state: dict) -> int:
    """Get the next Wilma day index."""
    return state.get("current_day_index", 0) + 1


# ── Main orchestration ──────────────────────────────────────────────────
def orchestrate_main(brand: dict, state: dict, dry_run: bool = False, stub: bool = False) -> dict:
    """Generate a main brand bundle using specialist agents."""
    post_id = get_next_post_id(state)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[orchestrator] Generating main bundle post_id={post_id} (stub={stub})")

    # Step 1: Pick a topic
    print("[orchestrator] Step 1: Content Creator → topic selection...")
    topic = generate_topic_with_content_creator(brand, stub=stub)
    print(f"  → Topic: {topic.get('title', '')}")

    # Step 2: Generate image prompt
    print("[orchestrator] Step 2: Image Prompt Engineer → image prompt...")
    image_prompt = generate_image_prompt(topic, brand) if not stub else "dark background, textured ceramic, gold kintsugi lines, cinematic lighting, 4k"
    print(f"  → Prompt: {image_prompt[:80]}...")

    # Step 3: Generate master reflection
    print("[orchestrator] Step 3: Master Reflection generation...")
    master_reflection = generate_master_reflection(topic, brand) if not stub else f"In a world obsessed with optimization, the Japanese art of kintsugi reminds us that breakage and repair are part of history — not something to hide. The golden seams are the story. {topic.get('title', '')} is not a problem to solve but a tension to inhabit. The algorithms want us smooth. Wabi-sabi says the crack is where the light gets in."
    print(f"  → Reflection: {master_reflection[:80]}...")

    # Step 4: Generate platform-specific captions
    print("[orchestrator] Step 4: Platform caption generation...")
    captions = {}
    for platform in brand["platforms"]:
        if stub:
            cap = generate_stub_caption(platform, topic, master_reflection, brand)
        else:
            cap = generate_caption_for_platform(platform, topic, master_reflection, brand)
        captions[platform] = cap
        print(f"  → {platform}: {len(cap)} chars")

    # Build bundle
    bundle = {
        "post_id": post_id,
        "timestamp": timestamp,
        "post": {
            "pillar": topic.get("pillar", "quote"),
            "title": topic.get("title", ""),
            "image_prompt": image_prompt,
            "caption_prompt": f"Reflect on {topic.get('title', '').lower()}. What does it teach us about productive failure? {brand['hashtag']}",
            "id": post_id,
        },
        "platforms": brand["platforms"],
        "image": f"images/post_{timestamp}.jpg",
        "image_clean": f"images/post_{timestamp}_clean.jpg",
        "reel": f"reels/reel_{timestamp}.mp4",
        "story": f"images/story_{timestamp}.jpg",
        "carousel": [],
        "master_reflection": master_reflection,
        "captions": captions,
        "platforms_prepared": [],
        "platforms_posted": [],
    }

    if dry_run:
        print("\n[orchestrator] DRY RUN — not writing to state.json")
        print(json.dumps(bundle, indent=2)[:2000])
        return bundle

    # Write to state.json
    state["pending_bundle"] = bundle
    state["content_queue"] = []
    save_state(brand["state_path"], state)
    print(f"\n[orchestrator] ✓ Written to {brand['state_path']}")
    return bundle


def orchestrate_wilma(brand: dict, state: dict, day: int | None = None, dry_run: bool = False, stub: bool = False) -> dict:
    """Generate a Wilma bundle using specialist agents."""
    target_day = day or get_next_wilma_day(state)
    print(f"[orchestrator] Generating Wilma bundle day={target_day} (stub={stub})")

    # Read schedule for the day's topic hint
    schedule_topic = ""
    if SCHEDULE_PATH.exists():
        try:
            schedule = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
            if isinstance(schedule, list):
                for item in schedule:
                    if item.get("day") == target_day:
                        schedule_topic = item.get("topic", "")
                        break
        except Exception:
            pass

    # Step 1: Pick angle via Social Media Strategist
    if stub:
        angle_data = {"angle": schedule_topic or "digital wellness for families", "topic": schedule_topic or "screen time management", "hook": f"Day {target_day}"}
    else:
        agent = load_agent("social-media-strategist") or load_agent("marketing-social-media-strategist")
        system = f"""You are a social media strategist for {brand['name']} (DigitalGuard).
{agent['body'] if agent else ''}

Given the schedule topic, pick a specific angle for today's post.
Return ONLY a JSON object: {{"angle": "...", "topic": "...", "hook": "..."}}"""

        user = f"Schedule topic: {schedule_topic or 'digital wellness for families'}"
        result = llm_call(system, user, max_tokens=256)
        try:
            angle_data = json.loads(result)
        except Exception:
            angle_data = {"angle": schedule_topic, "topic": schedule_topic, "hook": f"Day {target_day}"}

    # Step 2: Image prompt
    print("[orchestrator] Step 2: Image Prompt Engineer → image prompt...")
    image_prompt = generate_image_prompt({"title": angle_data.get("topic", ""), "topic": angle_data.get("angle", "")}, brand) if not stub else "dark background, teal/cyan gradient, digital shield icon, cinematic"
    print(f"  → Prompt: {image_prompt[:80]}...")

    # Step 3: Master reflection
    print("[orchestrator] Step 3: Master Reflection generation...")
    master_reflection = generate_master_reflection({"title": angle_data.get("topic", ""), "topic": angle_data.get("angle", "")}, brand) if not stub else f"When the classroom banned phones for a semester, something unexpected happened. The kids didn't just survive — they remembered what boredom felt like. And boredom, it turns out, is where creativity begins."
    print(f"  → Reflection: {master_reflection[:80]}...")

    # Step 4: Captions
    print("[orchestrator] Step 4: Platform caption generation...")
    captions = {}
    for platform in brand["platforms"]:
        if stub:
            cap = generate_stub_caption(platform, angle_data, master_reflection, brand)
        else:
            cap = generate_caption_for_platform(platform, angle_data, master_reflection, brand)
        captions[platform] = cap
        print(f"  → {platform}: {len(cap)} chars")

    bundle = {
        "post_id": f"day_{target_day}",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "day": target_day,
        "image": f"images/day{target_day}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
        "image_clean": f"images/day{target_day}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_clean.jpg",
        "bundle_captions": captions,
        "angle": angle_data.get("angle", ""),
        "master_reflection": master_reflection,
        "platforms_posted": [],
    }

    if dry_run:
        print("\n[orchestrator] DRY RUN — not writing to state.json")
        print(json.dumps(bundle, indent=2)[:2000])
        return bundle

    # Write to Wilma state.json
    state.setdefault("content_queue", [])
    state["content_queue"].append(bundle)
    state["current_day_index"] = target_day
    save_state(brand["state_path"], state)
    print(f"\n[orchestrator] ✓ Written to {brand['state_path']}")
    return bundle


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-agent content orchestrator for ig-autobot")
    parser.add_argument("--brand", choices=["main", "wilma"], default="main", help="Brand to generate for")
    parser.add_argument("--day", type=int, default=None, help="Wilma day number (auto-detects if omitted)")
    parser.add_argument("--dry-run", action="store_true", help="Print bundle without writing to state.json")
    parser.add_argument("--stub", action="store_true", help="Use stub LLM responses (no API calls)")
    parser.add_argument("--agents", action="store_true", help="List available agents and exit")
    args = parser.parse_args()

    if args.agents:
        agents = load_all_agents()
        print(f"Available agents ({len(agents)}):")
        for a in agents:
            print(f"  {a['slug']:45s} {a['name']}")
        return

    brand = BRAND[args.brand]
    state = load_state(brand["state_path"])

    if args.brand == "main":
        orchestrate_main(brand, state, dry_run=args.dry_run)
    else:
        orchestrate_wilma(brand, state, day=args.day, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
