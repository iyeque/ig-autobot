#!/usr/bin/env python3
"""
Generate pages data for the static site:
- _site/dashboard_data.json   — metrics for main + Wilma identities
- _site/gallery.json           — enriched gallery with metadata
- _site/agents.json            — workflow/agent descriptions
- _site/privacy.html           — generated from PRIVACY.md
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SITE = ROOT / "_site"

# ── helpers ────────────────────────────────────────────────────────────────────

def load(path, default=None):
    p = ROOT / path
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def write(path, data, indent=2):
    (_SITE / path).write_text(json.dumps(data, indent=indent), encoding="utf-8")

def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fmt_date(ts):
    """ISO or epoch → readable."""
    if not ts:
        return "—"
    try:
        s = str(ts)
        if s.isdigit():
            return datetime.fromtimestamp(int(s), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return s[:16] if len(s) >= 16 else s
    except Exception:
        return str(ts)


# ── platform display ───────────────────────────────────────────────────────────

PLATFORM_LABELS = {
    "instagram": "Instagram",
    "linkedin": "LinkedIn",
    "pinterest": "Pinterest",
    "youtube": "YouTube",
    "threads": "Threads",
    "bluesky": "Bluesky",
    "facebook": "Facebook",
}

PLATFORM_ICONS = {
    "instagram": "📷",
    "linkedin": "💼",
    "pinterest": "📌",
    "youtube": "🎬",
    "threads": "🧵",
    "bluesky": "🦋",
    "facebook": "📘",
}

POST_TYPE_MAP = {
    "image": "Image",
    "reel": "Reel",
    "carousel": "Carousel",
    "quote": "Quote",
    "story": "Story",
}


# ── 1.  Dashboard data  ────────────────────────────────────────────────────────

def build_dashboard(main_state, wilma_state, posts_data, used_ids):
    """
    main_state  : root state.json dict
    wilma_state : forwilma/state.json dict
    posts_data  : posts.json list
    used_ids    : state.json['used_ids'] dict
    """
    now = utcnow()

    # --- main identity ---
    active = main_state.get("active_bundle") or {}
    queue = main_state.get("content_queue") or []
    posted = main_state.get("platform_posted_bundles") or {}
    pillars = main_state.get("pillar_history") or []
    ctas = main_state.get("cta_history") or []
    series = main_state.get("active_series") or {}

    main_posts_by_platform = {}
    for plat, ids in (used_ids or {}).items():
        main_posts_by_platform[plat] = len(ids)

    total_main_posted = sum(main_posts_by_platform.values())

    main_recent = []
    for plat, ids in (used_ids or {}).items():
        recent_ids = ids[-5:] if len(ids) >= 5 else ids
        for pid in recent_ids:
            main_recent.append({"platform": plat, "post_id": pid})
    main_recent.sort(key=lambda r: r["post_id"], reverse=True)

    main_queue_items = []
    for item in queue[:10]:
        main_queue_items.append({
            "post_id": item.get("post_id"),
            "platforms": [PLATFORM_LABELS.get(p, p) for p in (item.get("platforms") or [])],
            "format": POST_TYPE_MAP.get(item.get("format"), item.get("format", "?")),
        })

    # --- Wilma identity ---
    wilma_current_day = wilma_state.get("current_day_index", "?")
    wilma_history = wilma_state.get("history") or []
    wilma_posted = wilma_state.get("platform_posted_bundles") or {}
    wilma_queue = wilma_state.get("content_queue") or []
    wilma_active = wilma_state.get("active_bundle")
    wilma_last_topic = wilma_state.get("last_topic", "—")

    wilma_posts_by_platform = {}
    for plat, ids in wilma_posted.items():
        wilma_posts_by_platform[plat] = len(ids)

    total_wilma_posted = sum(wilma_posts_by_platform.values())

    wilma_recent = []
    for plat, ids in wilma_posted.items():
        recent_ids = ids[-5:] if len(ids) >= 5 else ids
        for pid in recent_ids:
            wilma_recent.append({"platform": plat, "post_id": pid})
    wilma_recent.sort(key=lambda r: r["post_id"], reverse=True)

    wilma_queue_items = []
    for item in wilma_queue[:10]:
        wilma_queue_items.append({
            "post_id": item.get("post_id"),
            "platforms": [PLATFORM_LABELS.get(p, p) for p in (item.get("platforms") or [])],
            "format": POST_TYPE_MAP.get(item.get("format"), item.get("format", "?")),
        })

    # --- post counts by pillar (from posts.json) ---
    pillar_counts = {}
    for p in posts_data:
        pk = p.get("pillar", "unknown")
        pillar_counts[pk] = pillar_counts.get(pk, 0) + 1

    return {
        "generated_at": now,
        "main": {
            "identity": "M.W.E. Wigman — The Nine Stitches",
            "posts_by_platform": main_posts_by_platform,
            "total_posted": total_main_posted,
            "active_bundle": {
                "post_id": active.get("post_id"),
                "image": Path(active.get("image", "")).name if active.get("image") else None,
                "reel": Path(active.get("reel", "")).name if active.get("reel") else None,
                "carousel": Path(active.get("carousel", "")).name if active.get("carousel") else None,
                "platforms_posted": [PLATFORM_LABELS.get(p, p) for p in (active.get("platforms_posted") or [])],
                "platforms_prepared": [PLATFORM_LABELS.get(p, p) for p in (active.get("platforms_prepared") or [])],
                "captions": {k: v for k, v in (active.get("captions") or {}).items()},
            },
            "queue": main_queue_items,
            "queue_count": len(queue),
            "recent": main_recent[:10],
            "pillars_used": list(dict.fromkeys(pillars))[-8:],
            "ctas_used": list(dict.fromkeys(ctas))[-8:],
            "pillar_counts": pillar_counts,
            "series": {k: len(v) if isinstance(v, list) else v for k, v in series.items()},
        },
        "wilma": {
            "identity": "Wilma — Mindful Reflections",
            "current_day": wilma_current_day,
            "posts_by_platform": wilma_posts_by_platform,
            "total_posted": total_wilma_posted,
            "last_topic": wilma_last_topic,
            "queue": wilma_queue_items,
            "queue_count": len(wilma_queue),
            "active_bundle": wilma_active,
            "recent": wilma_recent[:10],
            "history_days": len(wilma_history),
        },
    }


# ── 2.  Gallery data  ─────────────────────────────────────────────────────────

def build_gallery(posts_data, main_state):
    """
    Produce enriched gallery entries from posts.json.
    Each entry: id, title, pillar, platform, image, date, caption, type
    """
    posted_ids = set()
    for plat, ids in (main_state.get("used_ids") or {}).items():
        posted_ids.update(ids)

    gallery = []
    for p in posts_data:
        pid = p.get("id")
        if pid not in posted_ids:
            continue

        image = p.get("image") or p.get("image_path") or f"images/post_{pid}.jpg"
        caption = p.get("caption") or p.get("caption_prompt") or p.get("title", "")
        pillar = p.get("pillar", "unknown")
        title = p.get("title", "")
        platform = p.get("platform") or "instagram"
        post_type = p.get("type") or POST_TYPE_MAP.get(pillar, "Image")

        gallery.append({
            "id": pid,
            "title": title,
            "pillar": pillar,
            "platform": platform,
            "image": image,
            "caption": caption,
            "type": post_type,
            "date": p.get("date") or f"Post #{pid}",
        })

    # Sort newest first
    gallery.sort(key=lambda e: e["id"], reverse=True)
    return gallery


# ── 3.  Agents / workflows data  ──────────────────────────────────────────────

def build_agents():
    """
    Read .github/workflows/*.yml (active only) and produce a structured
    agent/workflow catalog for the visualization.
    """
    wf_dir = ROOT / ".github" / "workflows"
    archived = wf_dir / "archive"
    agents = []

    if not wf_dir.exists():
        return agents

    schedule_map = {
        "master_publish.yml":        {"name": "Master Publish",         "schedule": "Daily 17:00 UTC",   "platform": "All (auto)",     "kind": "publish"},
        "master_carousel.yml":       {"name": "Main IG Carousel",       "schedule": "Mon/Wed/Fri 13:30 UTC", "platform": "Instagram",  "kind": "carousel"},
        "master_linkedin_carousel.yml":{"name": "Main LinkedIn Carousel","schedule": "Wed 16:00 UTC",    "platform": "LinkedIn",       "kind": "carousel"},
        "master_quote.yml":          {"name": "IG Quote Poster",        "schedule": "5×/day (02,06,10,18,22 UTC)", "platform": "Instagram", "kind": "quote"},
        "wilma_publish.yml":         {"name": "Wilma Publish",         "schedule": "Daily 16:30 UTC",   "platform": "Wilma (LinkedIn/Bluesky)", "kind": "publish"},
        "wilma_carousel.yml":        {"name": "Wilma LinkedIn Carousel","schedule": "Fri+Sun 15:00 UTC","platform": "Wilma LinkedIn", "kind": "carousel"},
        "master_wilma_gen.yml":      {"name": "Wilma Generator",       "schedule": "Daily 11:30 UTC",   "platform": "Wilma (generation)", "kind": "generate"},
        "master_content_gen.yml":    {"name": "Content Generator",     "schedule": "Mon/Tue/Thu/Sat 02:00 UTC", "platform": "All (generation)", "kind": "generate"},
    }

    for fname, info in schedule_map.items():
        fpath = wf_dir / fname
        if not fpath.exists():
            continue
        agents.append({
            "file": fname,
            "name": info["name"],
            "schedule": info["schedule"],
            "platform": info["platform"],
            "kind": info["kind"],
            "status": "active",
            "trigger": "scheduled",
        })

    # Also add archived workflows for awareness
    if archived.exists():
        for f in archived.iterdir():
            if f.suffix == ".yml":
                agents.append({
                    "file": f"archive/{f.name}",
                    "name": f.name.replace("auto_", "").replace(".yml", "").replace("_", " ").title(),
                    "schedule": "⏸ archived",
                    "platform": "—",
                    "kind": f.name.split("_", 1)[-1].replace(".yml", "").replace("_", " ").title() if "_" in f.name else "—",
                    "status": "archived",
                    "trigger": "none",
                })

    return agents


# ── 4.  Privacy HTML  ─────────────────────────────────────────────────────────

def build_privacy_html():
    """Convert PRIVACY.md → _site/privacy.html"""
    privacy_md = ROOT / "PRIVACY.md"
    if not privacy_md.exists():
        return None

    md_text = privacy_md.read_text(encoding="utf-8")

    # Very simple markdown → HTML (headers + paragraphs + bullets)
    html_parts = ['<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">',
                  '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
                  '<title>Privacy Policy — ig-autobot</title>',
                  '<link rel="stylesheet" href="style.css">',
                  '</head><body><div class="container">',
                  '<nav><a href="index.html">← Back to Gallery</a></nav>',
                  '<h1>Privacy Policy</h1>']

    in_list = False
    for line in md_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue

        if stripped.startswith("# "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h2>{stripped[2:].strip()}</h2>")
        elif stripped.startswith("## "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h3>{stripped[3:].strip()}</h3>")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{stripped[2:].strip()}</li>")
        elif re.match(r"^\d+\.\s", stripped):
            if not in_list:
                html_parts.append("<ol>")
                in_list = True
            html_parts.append(f"<li>{stripped.split('.', 1)[1].strip()}</li>")
        else:
            html_parts.append(f"<p>{stripped}</p>")

    if in_list:
        html_parts.append("</ul>")

    html_parts.append('</div></body></html>')
    (_SITE / "privacy.html").write_text("\n".join(html_parts), encoding="utf-8")
    return True


# ── 5.  main  ──────────────────────────────────────────────────────────────────

def main():
    main_state  = load("state.json", {})
    wilma_state = load("forwilma/state.json", {})
    posts_data  = load("posts.json", [])
    if isinstance(posts_data, dict):
        posts_data = posts_data.get("posts", [])
    used_ids    = main_state.get("used_ids", {})

    # Ensure _site exists
    _SITE.mkdir(exist_ok=True)

    # 1. Dashboard
    dashboard = build_dashboard(main_state, wilma_state, posts_data, used_ids)
    write("dashboard_data.json", dashboard)
    print(f"✓ dashboard_data.json — {len(dashboard['main']['pillar_counts'])} pillars, "
          f"main={dashboard['main']['total_posted']} posts, "
          f"wilma day={dashboard['wilma']['current_day']}")

    # 2. Gallery
    gallery = build_gallery(posts_data, main_state)
    write("gallery.json", gallery)
    print(f"✓ gallery.json — {len(gallery)} enriched entries")

    # 3. Agents
    agents = build_agents()
    write("agents.json", agents)
    act = sum(1 for a in agents if a["status"] == "active")
    arch = sum(1 for a in agents if a["status"] == "archived")
    print(f"✓ agents.json — {act} active, {arch} archived workflows")

    # 4. Privacy
    ok = build_privacy_html()
    if ok:
        print("✓ privacy.html generated from PRIVACY.md")
    else:
        print("⚠ PRIVACY.md not found — privacy.html not generated")

    # 5. Sync root files → _site/ (excluding generated ones)
    GENERATED = {"gallery.json", "dashboard_data.json", "agents.json", "privacy.html"}
    ROOT_FILES = {
        "index.html", "style.css", "app.js",
        "PRIVACY.md", "README.md", "roadmap.md",
        "quotes_state.json", "reel_rotation.json",
        "captions_bundle.json", "carousel_hosted_urls.json",
        "growth_data.json", "AUDIT_2026-08-18.md", "DEEP_LEARNING.md",
    }
    for name in ROOT_FILES:
        src = ROOT / name
        if src.exists():
            dst = _SITE / name
            if dst.is_file() and dst.read_bytes() == src.read_bytes():
                continue
            dst.write_bytes(src.read_bytes())
    print("✓ root files synced to _site/")


if __name__ == "__main__":
    main()
