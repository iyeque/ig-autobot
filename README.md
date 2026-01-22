# ig-autobot
# Instagram Automation Bot for M.W.E. Wigman  
### Automated posting system for trilogy author page

This repository contains a fully automated Instagram posting workflow designed for the author M.W.E. Wigman.  
It uses:

- GitHub Actions (scheduler + automation)
- External AI APIs for text and image generation
- Instagram Graph API for publishing posts
- A curated JSON file of post concepts based on *The Nine Stitches*

The goal is to maintain a consistent, philosophical, nature‑driven aesthetic aligned with the themes of the trilogy.

---

## 📌 Project Structure
 ├── bot.py              # Main automation script 
 ├── posts.json          # List of 30 curated post concepts 
 ├── state.json          # Tracks which posts have been used 
 └── .github/ 
    └── workflows/ 
       └── auto_instagram.yml   # GitHub Actions workflow


---

## 🔧 How It Works

1. **GitHub Actions** triggers the workflow on a schedule (or manually).
2. `bot.py`:
   - Reads `posts.json`
   - Selects the next unused post
   - Generates:
     - an AI image (via HuggingFace)
     - an AI caption (via HuggingFace)
   - Uploads the image to a public host
   - Publishes the post to Instagram via the Graph API
3. `state.json` is updated to avoid repeating posts.

---

## 🔑 Required Secrets (GitHub → Settings → Secrets → Actions)

| Secret Name        | Description |
|--------------------|-------------|
| `HF_TOKEN`         | HuggingFace token for text & image generation |
| `IG_ACCESS_TOKEN`  | Long‑lived Instagram Graph API token |
| `IG_USER_ID`       | Instagram Business Account ID |

---

## 🧠 Content Philosophy

The posts are based on the themes of:

- *The Nine Stitches*  
- *A Burden of One’s Choice*  
- The upcoming third book in the trilogy  

They explore:

- Nature as metaphor  
- Systems thinking  
- Duality and contradiction  
- Human psychology  
- Scars, cycles, and introspection  

---

## 🚀 Running Locally (Optional)

You can test the bot locally:

```bash
python bot.py

## 📈 Future Enhancements
- Add Book II and Book III post sets
- Add carousel support
- Add reel generation
- Add multi‑account support
- Add analytics logging
- A multi‑post scheduler
- A JSON‑driven content queue upgrade
- A caption‑style enhancer
- A debugging system


## © Copyright
All content, prompts, and generated captions are © 2024–2026 M.W.E. Wigman.
Unauthorized reproduction is prohibited.
