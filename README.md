# ig-autobot 🤖📚

**Instagram Automation Bot for M.W.E. Wigman's *The Nine Stitches* Trilogy**

[![Instagram](https://img.shields.io/badge/Instagram-@mwewigman-E4405F?logo=instagram)](https://instagram.com/mwewigman)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen?logo=github)](https://iyeque.github.io/ig-autobot/)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

&gt; *"What happens if you try to fail and succeed?"*
&gt; — The central paradox of The Nine Stitches

**[🌐 View Generated Images](https://iyeque.github.io/ig-autobot/)**

---

## ✨ What This Does

An intelligent, book-aware automation system that maintains M.W.E. Wigman's Instagram presence with philosophical depth and visual consistency.

**Core Capabilities:**
- 🧠 **AI-Powered Captions** — Uses Cerebras AI with book context awareness to write in the author's voice
- 🎨 **Triple Image Generation** — AI Horde → DeepAI → Pollinations.ai for reliable visual creation
- 🌐 **GitHub Pages Hosting** — Instant image access via [iyeque.github.io/ig-autobot](https://iyeque.github.io/ig-autobot/)
- 📅 **Smart Scheduling** — Daily posts at 10 AM UTC with intelligent content rotation
- 📖 **Book-Integrated** — Extracts themes, quotes, and concepts directly from *The Nine Stitches* PDF
- 🔄 **Self-Healing** — Auto-generates new post concepts when the queue runs low

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
Instagram Graph API
---


## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- GitHub account with repository secrets access
- Instagram Business Account
- Facebook Developer App (for Graph API)

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
export DEEPAI_API_KEY="your_key_here"
export AI_HORDE_API_KEY="your_key_here"
export PDF_BOOK_FILENAME="The-Nine-Stitches.pdf"

# Method B: Use .env file (recommended)
cp .env.example .env
# Edit .env with your keys
```

Add to top of bot.py:

```py
from dotenv import load_dotenv
from pathlib import Path

#Load .env file
dotenv_path = Path(__file__).parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from {dotenv_path}")
```

Then add to .gitignore:
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

Add these secrets in Settings → Secrets and variables → Actions:

CEREBRAS_API_KEY	✅ Yes	Caption generation via Llama 3.1
DEEPAI_API_KEY	⚠️ Recommended	Image fallback
AI_HORDE_API_KEY	⚠️ Recommended	Primary image generation (faster)
IG_USER_ID	✅ Yes	Instagram Business Account ID
IG_ACCESS_TOKEN	✅ Yes	Long-lived Graph API token
FB_APP_ID	Facebook App ID (may be required for some Graph API permissions)
FB_APP_SECRET	Facebook App Secret (may be required for some Graph API permissions)
PDF_BOOK_FILENAME	✅ Yes	Name of PDF in repo (e.g., The-Nine-Stitches.pdf)

### 4. Enable GitHub Pages

Settings → Pages:
Source: GitHub Actions
URL: https://iyeque.github.io/ig-autobot/

## 📖 Content Philosophy

Posts are engineered around The Nine Stitches core themes:

1	Intention vs. Outcome	Compass & terrain, cognitive bias, bioluminescence
2	Adversity & Growth	Serotinous cones, antifragility, amor fati
3	Elegance of Flaws	Kintsugi, wabi-sabi, Leaning Tower of Pisa
4	Microcosm/Macrocosm	Butterfly effect, keystone species, ripple effects

Content Pillars:
*micro_philosophy* — Core philosophical concepts
*nature_metaphor* — Biological systems as human mirrors
*systems_psychology* — Cognitive science and behavior
*author_voice* — Direct M.W.E. Wigman perspective
*quote* — Book excerpts and epigraphs 

## ⚙️ How It Works

graph TD
    A[Schedule Trigger 10 AM UTC] --> B[bot.py Executes]
    B --> C{Select Unused Post}
    C -->|All Used| D[Generate 10 New Posts]
    C -->|Available| E[Extract Book Context]
    E --> F[Generate Caption via Cerebras]
    F --> G[Generate Image via AI Horde]
    G -->|Fails| H[Fallback: DeepAI]
    H -->|Fails| I[Fallback: Pollinations.ai]
    G -->|Success| J[Save to images/]
    H -->|Success| J
    I -->|Success| J
    J --> K[Commit & Push to GitHub]
    K --> L[Deploy to GitHub Pages]
    L --> M[Instant Image Access]
    M --> N[Post to Instagram]
    N --> O[Update state.json]

### Key Features:

Intelligent Rotation: Never repeats posts until pool exhausted
Context-Aware: Feeds 2000 characters of book text to AI for authentic voice
Resilient Generation: Triple-fallback image generation (AI Horde → DeepAI → Pollinations)
GitHub Pages Hosting: Instant image access, no CDN delays
Timestamped Images: post_20240210_143022.jpg organized chronologically
Concurrency Lock: Prevents duplicate posts from parallel runs

## 🛠️ Customization

### Adding New Posts

Edit posts.json (or let the bot auto-generate):

{
  "id": 31,
  "pillar": "nature_metaphor",
  "title": "Your new concept",
  "image_prompt": "Detailed description for AI image generator...",
  "caption_prompt": "Instructions for caption AI with #TheNineStitches hashtag..."
}

### Modifying Post Schedule

Edit .github/workflows/auto_instagram.yml:

on:
  schedule:
    - cron: "0 10 * * *"  # 10 AM UTC daily
    # - cron: "0 14 * * 1,3,5"  # Mon/Wed/Fri at 2 PM

### Changing Book Context

Update PDF_BOOK_FILENAME secret and ensure PDF is committed:

git add Your-New-Book.pdf
git commit -m "Add Book II content"
git push

## 🔧 Troubleshooting

### Image Generation Fails

#### AI Horde Issues:

403 FORBIDDEN → Check API key and kudos balance at stablehorde.net
No image URL → Prompt may be filtered; check logs for censored content
Timeout → Normal for complex prompts; bot retries automatically

#### DeepAI Fallback:

402 Payment Required → Free tier exhausted; add payment method or rely on Pollinations
500 Server Error → DeepAI server issue; bot will retry

#### Pollinations.ai (Third Fallback):

Always free, no API key needed
Lower quality but 99% uptime

#### Instagram Publishing Fails

Invalid token	Refresh long-lived token at developers.facebook.com
Media not found	Image URL must be public; check raw.githubusercontent.com link
Rate limit	Instagram allows ~25 posts/day; bot has built-in delays

#### GitHub Pages Issues

404 on image URL → Pages not deployed; check Settings → Pages → Source: GitHub Actions
Slow loading → Normal; CDN propagates globally within 1-2 minutes

#### Caption Generation Issues

Empty response → Check CEREBRAS_API_KEY validity
Off-brand voice → Verify PDF is extracted correctly; check `book_context` length
Missing hashtags → Bot auto-appends #TheNineStitches if absent

## 📊 Monitoring & Logs

### View recent runs:

#GitHub CLI
gh run list --workflow=auto_instagram.yml

#View specific run
gh run view <run-id> --log

### Check GitHub Pages status:

#Verify image is accessible
curl -I https://iyeque.github.io/ig-autobot/images/post_20240211_052047.jpg

### Local debugging:

#Verbose output
python bot.py --verbose 2>&1 | tee bot.log

#Test specific post
python -c "import bot; bot.main()"  # Uses current state.json

## 🗺️ Roadmap

[x] Book I (The Nine Stitches) full integration
[x] GitHub Pages hosting for instant image access
[x] Triple fallback image generation (AI Horde → DeepAI → Pollinations)
[ ] Book II (A Burden of One's Choice) content expansion
[ ] Book III (upcoming) teaser campaign mode
[ ] Carousel posts — Multi-image storytelling
[ ] Reels generation — Short-form video with AI voiceover
[ ] Engagement analytics — Track performance per pillar/theme
[ ] A/B caption testing — Auto-optimize for engagement
[ ] Multi-account — Support for author + book series accounts

## 📜 License & Attribution

**© 2024–2026 M.W.E. Wigman. All Rights Reserved.**

This software and all generated content are proprietary and confidential.
Unauthorized copying, distribution, modification, or commercial use 
is strictly prohibited without written permission.

**Generated Content:** All captions and images are derivative works 
of *The Nine Stitches* and remain intellectual property of the author.

**Third-Party APIs:**
- [Cerebras AI](https://cerebras.ai)
- [AI Horde](https://stablehorde.net)
- [DeepAI](https://deepai.org)
- [Pollinations.ai](https://pollinations.ai)
- [Instagram Graph API](https://developers.facebook.com)

## 🙏 Acknowledgments
Built with respect for the paradox: "To become, be calm. To be calm, pretend to be calm."
Questions/ Licensing inquiries? Open an issue or contact: mmmuraya@outlook.com

## View Live Images: 

iyeque.github.io/ig-autobot
---
## Changelog

v.1.2.0 (2026-02-16)
| Section | Update |
|---------|--------|
| **Image Generation** | Implemented OCR-based image filtering using OCR.space to detect and retry generation of censored or NSFW images, enhancing content safety and quality. |

v.1.1.0 (2026-02-15)
| Section | Update |
|---------|--------|
| **Scheduling** | Updated cron schedule for more frequent daily posts (10 AM, 12 PM, 2 PM, 4 PM, 6 PM, 8 PM UTC). |
| **Image Generation** | Integrated Qwen-Image-Max as primary generator, updated to synchronous API, and configured all generators for Instagram portrait (1080x1350 or closest). |
| **Caption Generation** | Enhanced hashtag selection in `generate_caption` for improved reach and relevance. |
| **Content Management** | Increased `posts.json` content with 30 additional contextually relevant ideas (total 60+ posts) to delay self-healing. |

v.1.0.0
| Section | Update |
|---------|--------|
| **Header badges** | Added GitHub Pages live badge |
| **Architecture** | Added `_site/` and Pages flow diagram |
| **Quick Start** | Added `.env` file method, Pages setup step |
| **How It Works** | Added Mermaid diagram with Pages deployment |
| **Troubleshooting** | Added Pages-specific errors and solutions |
| **Monitoring** | Added `curl` command to verify image access |
| **Roadmap** | Marked GitHub Pages and triple fallback as complete |
| **Footer** | Added live site link |

---

