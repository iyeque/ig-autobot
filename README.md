# ig-autobot 🤖📚

**Social Media Automation Bot for M.W.E. Wigman's *The Nine Stitches* Trilogy**

[![Instagram](https://img.shields.io/badge/Instagram-@mwewigman-E4405F?logo=instagram)](https://www.instagram.com/m.w.e_wigman/)
[![Visual Archive](https://img.shields.io/badge/Web-Visual_Archive-blue?logo=github)](https://iyeque.github.io/ig-autobot/)

An intelligent, book-aware automation system for M.W.E. Wigman that maintains a robust multi-platform social media presence with philosophical depth, modern wit, and visual consistency.

## ✨ What This Does

- 🧠 **Dual-Engine AI Captions** — Primary: AI Horde (targeting 120B+ models for maximum wit). Fallback: Cerebras AI (Llama 3.1).
- 🗣️ **Persona mastery** — Content is powered by a witty, self-deprecating 'Professional Failure Expert' persona tailored for deep engagement.
- 🎨 **Cinematic Visuals** — High-legibility Reels and Shorts with 85px overlays, dynamic transparency, automated watermarking, and cinematic motion.
- 🌐 **Multi-Platform Syndication** — Automated posting across **Instagram, LinkedIn, Pinterest, YouTube Shorts, Threads, and Bluesky**.
- 📈 **SEO Optimized** — Smart hashtag selection (3-5 tags for IG) and keyword-dense captioning for 2026 discovery standards.
- 🖼️ **Live Visual Gallery** — A [web-based archive](https://iyeque.github.io/ig-autobot/) that automatically curates and sorts all media chronologically with interactive hover-previews.
- 📅 **Smart Scheduling** — Optimized for 4x daily posts aligned with UAE peak-engagement windows (GST).
- 🔄 **Self-Healing Resilience** — Built-in `stash-pull-rebase` logic to handle concurrent jobs, auto-refreshes tokens, and aggressively filters AI artifacts.

## 🏗️ Architecture

```mermaid
graph TD
    A[Schedule Trigger] --> B[bot.py Executes]
    B --> C{Select Unused Post}
    C --> D[Smart Caption Generation]
    D --> E[Image Generation (SDXL)]
    E --> F[Professional Watermarking & Overlay]
    F --> G[Commit & Deploy]
    G --> H[Publish to Platforms]
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- GitHub account
- API keys: **Cerebras AI**, **AI Horde**, **OCR Space**

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
| `CEREBRAS_API_KEY` | Text | Fallback Caption generation |
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

> 💡 **Pinterest Note**: While in "Trial" tier, the bot uses the **Pinterest Sandbox API**. Ensure your tokens are generated for a Sandbox user in the Pinterest Dev Console.

---

## 📜 License & Attribution
**© 2024–2026 Digital Guardian. All Rights Reserved.**
