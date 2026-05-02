# ig-autobot 🤖📚

**Instagram Automation Bot for M.W.E. Wigman's *The Nine Stitches* Trilogy**

[![Instagram](https://img.shields.io/badge/Instagram-@mwewigman-E4405F?logo=instagram)](https://www.instagram.com/m.w.e_wigman/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen?logo=github)](https://iyeque.github.io/ig-autobot/)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

> *"What happens if you try to fail and succeed?"*
> — The central paradox of The Nine Stitches

**[🌐 View Generated Images](https://iyeque.github.io/ig-autobot/)**

---

## ✨ What This Does

An intelligent, book-aware automation system for **Digital Guardian** that maintains social media presence with philosophical depth, modern wit, and visual consistency.

**Core Capabilities:**
- 🧠 **AI-Powered Captions** — Primary engine: AI Horde (targeting 120B+ models for maximum wit). Fallback: Cerebras AI (Llama 3.1).
- 🎨 **Persona Upgrade** — Content is now powered by a witty, self-deprecating 'Relatable Failure Expert' persona tailored for Gen Z/Millennials.
- 🎨 **Visuals** — Professional, high-legibility cinematic Reels and Stories with 100px+ high-contrast overlays, automated logo watermarking, and cinematic movement.
- 🌐 **GitHub Pages Hosting** — Instant image access via [iyeque.github.io/ig-autobot](https://iyeque.github.io/ig-autobot/)
- 📅 **Smart Scheduling** — Daily posts with intelligent content rotation, brand-specific aesthetics, and automated hashtag management.
- 📖 **Book-Integrated** — Extracts themes, quotes, and concepts directly from *The Nine Stitches* PDF.
- 🔄 **Self-Healing** — Auto-generates new concepts when the queue runs low, auto-refreshes tokens, and aggressively strips AI filler.

---

## 🏗️ Architecture
ig-autobot/
├── 📜 bot.py                    # Main automation engine
├── 📋 posts.json                # 30+ curated post concepts
├── 📝 state.json                # Tracks used posts (auto-managed)
├── 🖼️  images/                   # Generated images (auto-committed)
├── 📄 caption.txt               # Generated caption output
├── 📚 The-Nine-Stitches.pdf     # Source material
└── 🔧 _site/                    # GitHub Pages build output


**Image Flow:**
bot.py → output.jpg → images/post_TIMESTAMP.jpg
↓
Git commit & push
↓
https://iyeque.github.io/ig-autobot/images/post_TIMESTAMP.jpg
↓
Social Media APIs

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- GitHub account
- API keys: Cerebras AI, AI Horde, OCR Space

### 2. Local Setup

```bash
# Clone repository
git clone https://github.com/iyeque/ig-autobot.git
cd ig-autobot

# Install dependencies
pip install -r requirements.txt

# Create .env from example
cp .env.example .env
# Edit .env with your keys
```

### 3. GitHub Actions Setup

#### Repository Secrets

Add these secrets in `Settings` → `Secrets and variables` → `Actions`:

| Secret | Required | Description |
| :--- | :--- | :--- |
| **CEREBRAS_API_KEY** | ✅ Yes | Secondary/Fallback Caption generation |
| **AI_HORDE_API_KEY** | ✅ Yes | Primary text & image generation |
| **PDF_BOOK_FILENAME**| ✅ Yes | Name of PDF in repo |
| **OCR_SPACE_API_KEY**| ℹ️ Opt | For image content safety filtering |
| **IG_ACCESS_TOKEN**  | ✅ Yes | Instagram Graph API token |

## ⚙️ How It Works

graph TD
    A[Schedule Trigger] --> B[bot.py Executes]
    B --> C{Select Unused Post}
    C -->|AI Horde Engine| D[Smart Caption Generation]
    D --> E[Image Generation (SDXL)]
    E --> F[Professional Watermarking & Overlay]
    F --> G[Commit & Deploy]
    G --> H[Publish to Platforms]

## 📜 License & Attribution
**© 2024–2026 Digital Guardian. All Rights Reserved.**
