# ig-autobot 🤖📚

**Social Media Automation Bot for M.W.E. Wigman's *The Nine Stitches* Trilogy**

[![Instagram](https://img.shields.io/badge/Instagram-@mwewigman-E4405F?logo=instagram)](https://www.instagram.com/m.w.e_wigman/)

An intelligent, book-aware automation system for **Digital Guardian** that maintains a robust multi-platform social media presence with philosophical depth, modern wit, and visual consistency.

## ✨ What This Does

- 🧠 **AI-Powered Captions** — Primary engine: AI Horde (targeting 120B+ models for maximum wit). Fallback: Cerebras AI (Llama 3.1).
- 🗣️ **Persona Upgrade** — Content is powered by a witty, self-deprecating 'Professional Failure Expert' persona tailored for deep engagement.
- 🎨 **Cinematic Visuals** — High-legibility Reels and Stories with 100px+ high-contrast overlays, automated logo watermarking, and cinematic movement.
- 🌐 **Multi-Platform Syndication** — Automated posting across **Instagram, LinkedIn, Pinterest, YouTube Shorts, Threads, and Bluesky**.
- 📅 **Smart Scheduling** — Optimized for 4x daily posts aligned with UAE peak-engagement windows (GST).
- 📖 **Book-Integrated** — Extracts themes, quotes, and concepts directly from *The Nine Stitches* PDF.
- 🔄 **Self-Healing** — Auto-generates new concepts when the queue runs low, auto-refreshes tokens, and aggressively strips AI filler.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Schedule Trigger] --> B[bot.py Executes]
    B --> C{Select Unused Post}
    C -->|AI Horde Engine| D[Smart Caption Generation]
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

---

## 📜 License & Attribution
**© 2024–2026 Digital Guardian. All Rights Reserved.**
