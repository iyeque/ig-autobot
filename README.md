# ig-autobot 🤖📚

**Instagram Automation Bot for M.W.E. Wigman's *The Nine Stitches* Trilogy**

[![Instagram](https://img.shields.io/badge/Instagram-@mwewigman-E4405F?logo=instagram)](https://instagram.com/mwewigman)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

> *"What happens if you try to fail and succeed?"*
> — The central paradox of The Nine Stitches

---

## ✨ What This Does

An intelligent, book-aware automation system that maintains M.W.E. Wigman's Instagram presence with philosophical depth and visual consistency.

**Core Capabilities:**
- 🧠 **AI-Powered Captions** — Uses Cerebras AI with book context awareness to write in the author's voice
- 🎨 **Dual Image Generation** — AI Horde primary, DeepAI fallback for reliable visual creation
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
└── ⚙️ .github/
└── workflows/
└── auto_instagram.yml   # GitHub Actions orchestration
plain
Copy

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

# Set environment variables
export CEREBRAS_API_KEY="your_key_here"
export DEEPAI_API_KEY="your_key_here"
export AI_HORDE_API_KEY="your_key_here"
export PDF_BOOK_FILENAME="The-Nine-Stitches.pdf"

# Run locally
python bot.py

# Check outputs
cat caption.txt
open output.jpg  # or your image viewer
3. GitHub Actions Setup
Add these secrets in Settings → Secrets and variables → Actions:
Table
Copy
Secret	Required	Purpose
CEREBRAS_API_KEY	✅ Yes	Caption generation via Llama 3.1
DEEPAI_API_KEY	⚠️ Recommended	Image fallback
AI_HORDE_API_KEY	⚠️ Recommended	Primary image generation (faster)
IG_USER_ID	✅ Yes	Instagram Business Account ID
IG_ACCESS_TOKEN	✅ Yes	Long-lived Graph API token
FB_APP_ID	Facebook App ID (may be required for some Graph API permissions)
FB_APP_SECRET	Facebook App Secret (may be required for some Graph API permissions)
PDF_BOOK_FILENAME	✅ Yes	Name of PDF in repo (e.g., The-Nine-Stitches.pdf)
📖 Content Philosophy
Posts are engineered around The Nine Stitches core themes:
Table
Copy
Chapter	Theme	Sample Metaphors
1	Intention vs. Outcome	Compass & terrain, cognitive bias, bioluminescence
2	Adversity & Growth	Serotinous cones, antifragility, amor fati
3	Elegance of Flaws	Kintsugi, wabi-sabi, Leaning Tower of Pisa
4	Microcosm/Macrocosm	Butterfly effect, keystone species, ripple effects
Content Pillars:
micro_philosophy — Core philosophical concepts
nature_metaphor — Biological systems as human mirrors
systems_psychology — Cognitive science and behavior
author_voice — Direct M.W.E. Wigman perspective
quote — Book excerpts and epigraphs

Key Features:
Intelligent Rotation: Never repeats posts until pool exhausted
Context-Aware: Feeds 2000 characters of book text to AI for authentic voice
Resilient Generation: Triple-retry logic with exponential backoff
Timestamped Images: post_20240210_143022.jpg avoids CDN caching
Concurrency Lock: Prevents duplicate posts from parallel runs
🛠️ Customization
Adding New Posts
Edit posts.json (or let the bot auto-generate):
JSON
Copy
{
  "id": 31,
  "pillar": "nature_metaphor",
  "title": "Your new concept",
  "image_prompt": "Detailed description for AI image generator...",
  "caption_prompt": "Instructions for caption AI with #TheNineStitches hashtag..."
}
Modifying Post Schedule
Edit .github/workflows/auto_instagram.yml:
yaml
Copy
on:
  schedule:
    - cron: "0 10 * * *"  # 10 AM UTC daily
    # - cron: "0 14 * * 1,3,5"  # Mon/Wed/Fri at 2 PM
Changing Book Context
Update PDF_BOOK_FILENAME secret and ensure PDF is committed:
bash
Copy
git add Your-New-Book.pdf
git commit -m "Add Book II content"
git push

🔧 Troubleshooting
Image Generation Fails
AI Horde Issues:
plain
Copy
403 FORBIDDEN → Check API key and kudos balance at stablehorde.net
No image URL → Prompt may be filtered; check logs for censored content
Timeout → Normal for complex prompts; bot retries automatically
DeepAI Fallback:
plain
Copy
AttributeError → Outdated code; ensure using latest bot.py
Rate limit → Wait 60s; bot has built-in exponential backoff
Instagram Publishing Fails
Table
Copy
Error	Solution
Invalid token	Refresh long-lived token at developers.facebook.com
Media not found	Image URL must be public; check raw.githubusercontent.com link
Rate limit	Instagram allows ~25 posts/day; bot has built-in delays
Caption Generation Issues
plain
Copy
Empty response → Check CEREBRAS_API_KEY validity
Off-brand voice → Verify PDF is extracted correctly; check `book_context` length
Missing hashtags → Bot auto-appends #TheNineStitches if absent

📊 Monitoring & Logs
View recent runs:
bash
Copy
# GitHub CLI
gh run list --workflow=auto_instagram.yml

# View specific run
gh run view <run-id> --log
Local debugging:
bash
Copy
# Verbose output
python bot.py --verbose 2>&1 | tee bot.log

# Test specific post
python -c "import bot; bot.main()"  # Uses current state.json

🗺️ Roadmap
[x] Book I (The Nine Stitches) full integration
[ ] Book II (A Burden of One's Choice) content expansion
[ ] Book III (upcoming) teaser campaign mode
[ ] Carousel posts — Multi-image storytelling
[ ] Reels generation — Short-form video with AI voiceover
[ ] Engagement analytics — Track performance per pillar/theme
[ ] A/B caption testing — Auto-optimize for engagement
[ ] Multi-account — Support for author + book series accounts

📜 License & Attribution
© 2024–2026 M.W.E. Wigman. All Rights Reserved.
This automation system is proprietary software. The generated content, prompts, and underlying concepts from The Nine Stitches are protected by copyright.
Third-Party APIs:
Cerebras AI — cerebras.ai
AI Horde — stablehorde.net
DeepAI — deepai.org
Instagram Graph API — developers.facebook.com

🙏 Acknowledgments
Built with respect for the paradox: "To become, be calm. To be calm, pretend to be calm."
Questions? Open an issue or contact: mmmuraya@outlook.com