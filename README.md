# ig-autobot 🤖📚

**Social Media Automation Bot for M.W.E. Wigman's *The Nine Stitches* Trilogy + Digital Guardian (Wilma)**

[![Instagram](https://img.shields.io/badge/Instagram-@mwewigman-E4405F?logo=instagram)](https://www.instagram.com/m.w.e_wigman/)
[![Visual Archive](https://img.shields.io/badge/Web-Visual_Archive-blue?logo=github)](https://iyeque.github.io/ig-autobot/)

An intelligent, book-aware automation system that maintains a robust multi-platform social media presence for author **M.W.E. Wigman** with philosophical depth, modern wit, and visual consistency. The **Digital Guardian / Wilma** subbrand is managed inside `forwilma/` with its own state, schedule, and LinkedIn/Bluesky workflows.

## ✨ What This Does

- 🧠 **Unified Asset Generation** — Phase 6 logic: Generates a single Master Image and platform-tailored captions in one pass.
- 🎨 **Cinematic Visuals** — High-legibility Reels and Shorts with 75px overlays (mobile-safe), visual 'Pattern Interrupts' at 3s to boost completion rates, and professional watermarking.
- 📸 **Carousel Strategy** — Instagram uses a weekday carousel/reel/static alternator. Wilma enforces carousel posts for MOFU days (Friday/Sunday) per the Digital Guardian brand brief.
- 🔄 **Self-Healing Resilience** — Built-in `stash-pull-rebase` logic and automated token refreshes for LinkedIn.
- 🌐 **Zero-Inference Publishing** — Posting workflows are decoupled from AI generation. They pick up pre-built assets, making them 100% immune to API timeouts or queue delays.
- 📈 **SEO Optimized** — Smart hashtag selection (3-5 tags) and keyword-dense captioning. No hashtags on Bluesky for a cleaner look.
- 🖼️ **Live Visual Gallery** — A [web-based archive](https://iyeque.github.io/ig-autobot/) that automatically curates and sorts all media chronologically.
- 📅 **Smart Scheduling** — 4 runs per week (Mon/Tue/Thu/Sat) optimized for peak GST engagement across 8 platforms.

## 🏗️ Architecture

```mermaid
graph TD
    subgraph "Generation Phase"
        A["Master Content Gen Workflow"] --> B["bot.py --mode generate_all"]
        B --> C["AI Horde / Local Media"]
        B --> D["Deterministic Platform Editor"]
        C --> E["Asset Bundle (JSON + Images)"]
        D --> E
        E --> F["Commit to Repository"]
    end

    subgraph "Distribution Phase"
        G["Platform Workflow (e.g. Threads)"] --> H["prepare_assets.py"]
        F -.-> H
        H --> I["Upload to Social Platform"]
    end
```

**Caption pipeline:** AI Horde generates the raw caption → `_deterministic_platform_editor()` applies platform-specific rules (voice filters, paragraph tightening, sentence-boundary truncation) → final assembly adds CTA and hashtags. No external API calls during editing. Used by both main and Wilma workflows.

**Resilience modes:** AI Horde health is pre-checked before generation. If the horde is down or rate-limited (429), text generation is skipped and the bot reuses prior-day images for media. When image generation fails and no fallback image exists, caption-only posts are produced for Bluesky and LinkedIn so publishing is never blocked by a media outage.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- GitHub account
- API keys: **AI Horde**, **OCR Space**

### 2. Local Setup
```bash
git clone https://github.com/iyeque/ig-autobot.git
cd ig-autobot
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
```

### 3. GitHub Actions Setup
Add these secrets in `Settings` → `Secrets and variables` → `Actions`:

| Secret | Platform | Description |
| :--- | :--- | :--- |
| `AI_HORDE_API_KEY` | Text | Primary Caption & Image gen |
| `OCR_SPACE_API_KEY`| Safety | Content filtering |
| `IG_ACCESS_TOKEN` | IG | Graph API token |
| `THREADS_ACCESS_TOKEN` | Threads | API Access Token |
| `THREADS_USER_ID` | Threads | User ID |
| `BLUESKY_HANDLE` | Bluesky | Handle |
| `BLUESKY_PASSWORD` | Bluesky | App Password |
| `YOUTUBE_CLIENT_ID` | YT | OAuth Client ID |
| `YOUTUBE_CLIENT_SECRET` | YT | OAuth Client Secret |
| `YOUTUBE_REFRESH_TOKEN` | YT | Refresh Token |
| `LINKEDIN_ACCESS_TOKEN` | LinkedIn | Access Token |
| `LINKEDIN_REFRESH_TOKEN` | LinkedIn | Refresh Token |
| `LINKEDIN_CLIENT_ID` | LinkedIn | OAuth Client ID |
| `LINKEDIN_CLIENT_SECRET` | LinkedIn | OAuth Client Secret |
| `LINKEDIN_URN` | LinkedIn | Person/Org URN |
| `PINTEREST_ACCESS_TOKEN` | Pinterest | Access Token |
| `PINTEREST_REFRESH_TOKEN` | Pinterest | Refresh Token |
| `PINTEREST_APP_ID` | Pinterest | App ID |
| `PINTEREST_APP_SECRET` | Pinterest | App Secret |
| `PINTEREST_BOARD_ID` | Pinterest | Board ID |
| `WILMA_LINKEDIN_REFRESH_TOKEN` | Wilma LinkedIn | Refresh Token |
| `WILMA_LINKEDIN_CLIENT_ID` | Wilma LinkedIn | OAuth Client ID |
| `WILMA_LINKEDIN_CLIENT_SECRET` | Wilma LinkedIn | OAuth Client Secret |
| `WILMA_LINKEDIN_URN` | Wilma LinkedIn | Person/Org URN |
| `WILMA_BLUESKY_HANDLE` | Wilma Bluesky | Handle |
| `WILMA_BLUESKY_PASSWORD` | Wilma Bluesky | App Password |

> 💡 **Pinterest Note**: While in "Trial" tier, the bot uses the **Pinterest Sandbox API**. Ensure your tokens are generated for a Sandbox user in the Pinterest Dev Console.

## 📁 Project Structure

- `bot.py` — Main generation + shared carousel/reel helpers + deterministic platform caption editor
- `forwilma/` — Digital Guardian / Wilma subbrand: state, schedule, publisher scripts
- `scripts/` — Platform-specific publishers and asset preparation
- `shared_utils.py` — State management helpers
- `.github/workflows/` — GitHub Actions schedules for generation and publishing
- `images/`, `reels/` — Local media storage
- `index.html`, `gallery.json` — Visual archive frontend

## 📜 License & Attribution
**© 2024–2026 Digital Guardian. All Rights Reserved.**
