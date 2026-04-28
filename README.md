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

An intelligent, book-aware automation system that maintains M.W.E. Wigman's Social media presence with philosophical depth and visual consistency.

**Core Capabilities:**
- 🧠 **AI-Powered Captions** — Uses Cerebras AI with book context awareness to write in the author's voice
- 🎨 **Image Generation** — AI Horde (API) default with portrait pipeline (1080×1350)
- 🌐 **GitHub Pages Hosting** — Instant image access via [iyeque.github.io/ig-autobot](https://iyeque.github.io/ig-autobot/)
- 📅 **Smart Scheduling** — Daily posts at 10 AM UTC with intelligent content rotation
- 📖 **Book-Integrated** — Extracts themes, quotes, and concepts directly from *The Nine Stitches* PDF
- 🔄 **Self-Healing** — Auto-generates new post concepts when the queue runs low and auto-refreshes API tokens

---

## 🏗️ Architecture
ig-autobot/
├── 📜 bot.py                    # Main automation engine
├── 📋 posts.json                # 30+ curated post concepts (expandable)
├── 📝 state.json                # Tracks used posts (auto-managed)
├── 🖼️  images/                   # Generated images (auto-committed)
│   └── .gitkeep
├── 📄 caption.txt               # Generated caption output
├── 📚 The-Nine-Stitches.pdf     # Source material for context
├── 🔧 _site/                    # GitHub Pages build output
│   └── images/                  # Deployed images (instant access)
└── ⚙️ .github/
└── workflows/
└── auto_instagram.yml   # GitHub Actions orchestration


**Image Flow:**
bot.py → output.jpg → images/post_TIMESTAMP.jpg
↓
Git commit & push
↓
_site/images/ (Pages build)
↓
https://iyeque.github.io/ig-autobot/images/post_TIMESTAMP.jpg
↓
Instagram Graph API / Pinterest V5 API
---


## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- GitHub account with repository secrets access
- Instagram Business Account
- Facebook Developer App (for Graph API)
- Pinterest Developer App (Standard Access recommended)

### 2. Local Setup

```bash
# Clone repository
git clone https://github.com/iyeque/ig-autobot.git
cd ig-autobot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (choose one method)

# Method A: Export directly
export CEREBRAS_API_KEY="your_key_here"
export AI_HORDE_API_KEY="your_key_here" # For fallback image generation
export PDF_BOOK_FILENAME="The-Nine-Stitches.pdf"

# Method B: Use .env file (recommended)
cp .env.example .env
# Edit .env with your keys
```

Add to .gitignore:
```
.env
venv/
__pycache__/
*.pyc
output.jpg
caption.txt
```

Run locally
```py
python bot.py
```

Check outputs
```
cat caption.txt
open output.jpg  # or your image viewer
```

### 3. GitHub Actions Setup

#### Repository Secrets

Add these secrets in `Settings` → `Secrets and variables` → `Actions`:

| Secret | Required | Description |
| :--- | :--- | :--- |
| **CEREBRAS_API_KEY** | ✅ Yes | Caption generation via Llama 3.1 |
| **AI_HORDE_API_KEY** | ✅ Yes | Image generation via AI Horde |
| **PDF_BOOK_FILENAME** | ✅ Yes | Name of PDF in repo (e.g., *The-Nine-Stitches.pdf*) |
| **IG_USER_ID** | ✅ Yes | Instagram Business Account ID |
| **IG_ACCESS_TOKEN** | ✅ Yes | Long-lived Graph API token |
| **LINKEDIN_ACCESS_TOKEN** | ✅ Yes | LinkedIn OAuth2 Token (w_member_social) |
| **LINKEDIN_URN** | ✅ Yes | Your Person URN (e.g., `urn:li:person:ABC`) |
| **PINTEREST_ACCESS_TOKEN** | ✅ Yes | Pinterest V5 API Access Token |
| **PINTEREST_REFRESH_TOKEN**| ✅ Yes | **New:** For automated token rotation |
| **PINTEREST_APP_ID**       | ✅ Yes | **New:** Your Pinterest App ID |
| **PINTEREST_APP_SECRET**   | ✅ Yes | **New:** Your Pinterest App Secret |
| **PINTEREST_BOARD_ID**    | ✅ Yes | The ID of the board where you want to pin |
| **OCR_SPACE_API_KEY**     | ℹ️ Opt | For image content safety filtering |

### 4. Enable GitHub Pages

Settings → Pages:
Source: GitHub Actions
URL: https://iyeque.github.io/ig-autobot/
*(Crucial for Pinterest as it requires a public URL to fetch images)*

## 📘 Series Content Engine (Phase 4)
The bot now features an automated narrative engine. 
- **Sequential Storytelling:** It can run multi-part series (e.g., "The Nine Stitches") by automatically selecting the next logical part in a sequence.
- **Narrative Continuity:** Captions are dynamically prefixed with "Part X — [Title]", and Reel/Story overlays carry this branding for better watch-time and audience retention.
- **Auto-Discovery:** If no series is active, the bot has a 20% chance to start a new one automatically.

## 🌐 Multi-Platform Automation
We have expanded from Instagram-only to include:
- **LinkedIn:** Automated scheduling via `scripts/publish_linkedin.py` and GitHub Actions.
- **Pinterest:** Automated board posting via `scripts/publish_pinterest.py`.
- **Self-Healing Tokens:** Pinterest integration now includes logic to automatically use the `REFRESH_TOKEN` to generate a new `ACCESS_TOKEN` every run, ensuring 100% uptime.

## ⚙️ How It Works

graph TD
    A[Schedule Trigger] --> B[bot.py Executes]
    B --> C{Select Unused Post}
    C -->|All Used| D[Generate 10 New Posts]
    C -->|Available| E[Extract Book Context]
    E --> F[Generate Caption via Cerebras]
    F --> G[Generate Image via AI Horde]
    G -->|Success| J[Save to images/]
    J --> K[Commit & Push to GitHub]
    K --> L[Deploy to GitHub Pages]
    L --> M[Instant Image Access]
    M --> N[Post to Instagram / Pinterest / LinkedIn]
    N --> O[Update state.json]

## 🛠️ Customization

### Adding New Posts
Edit `posts.json` or let the bot auto-generate. Concepts are categorized under pillars:
*micro_philosophy*, *nature_metaphor*, *systems_psychology*, *author_voice*, *quote*.

### Modifying Post Schedule
Edit `.github/workflows/auto_instagram.yml` or the specific platform workflow.

## 🔧 Troubleshooting

### Image Generation Fails
Check your `AI_HORDE_API_KEY` and kudos balance at stablehorde.net. The bot uses 1088×1344 and crops to 1080×1350 for Instagram.

### Pinterest/Instagram Publishing Fails
- **403 FORBIDDEN (Pinterest):** Ensure you have applied for "Standard Access" in the Pinterest Developer Portal.
- **401 UNAUTHORIZED:** Check if your tokens have expired. For Pinterest, ensure the Refresh Token secrets are set correctly.

## 📊 Monitoring & Logs
View recent runs via GitHub Actions tab or use the GitHub CLI:
`gh run list --workflow=auto_instagram.yml`

## 📜 License & Attribution
**© 2024–2026 M.W.E. Wigman. All Rights Reserved.**
Proprietary and confidential. Unauthorized use is prohibited.

---
## Changelog

v.1.4.0 (2026-04-28)
| Section | Update |
|---------|--------|
| **Pinterest** | Implemented automated token refresh logic using OAuth2 refresh flow. |
| **Workflow** | Added GitHub Pages environment configuration for reliable deployments. |
| **Stability** | Added automatic fallback to Sandbox API for trial-tier Pinterest apps. |

v.1.3.0 (2026-03-09)
| Section | Update |
|---------|--------|
| **Image Generation** | Made AI Horde the default generator. Generate at 1088×1344 and crop to 1080×1350. |
| **Docs** | Updated README to reflect AI Horde default and Pinterest setup requirements. |

... (rest of history)
