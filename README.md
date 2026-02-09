# ig-autobot
Instagram Automation Bot for M.W.E. Wigman
Automated posting system for a trilogy author page that generates images and captions with AI and publishes to Instagram on a schedule.

## Project Overview
This repository contains a fully automated Instagram posting workflow designed for the author M.W.E. Wigman. The system:
- Schedules and runs via GitHub Actions
- Generates captions using the Cerebras AI API
- Generates images using AI Horde, with DeepAI as a fallback
- Publishes posts using the Instagram Graph API
- Uses a curated posts.json of post concepts inspired by The Nine Stitches
The goal is to maintain a consistent, philosophical, nature‑driven aesthetic aligned with the trilogy’s themes.

## Project Structure
      
   |bot.py — Main automation script that selects posts, generates caption and image, and writes outputs
   |posts.json — Curated list of post concepts and prompts
   |state.json — Tracks which posts have been used to avoid repeats
   |images/ — Generated images (committed by workflow)
   |.github  
    └── workflows/ 
       └── auto_instagram.yml   # GitHub Actions workflow

## How It Works
- Trigger
- GitHub Actions runs on a schedule or via manual dispatch.
- Selection
- bot.py reads posts.json and picks the next unused post using state.json.
- Generation
- Caption: generated via Cerebras AI API.
- Image: generated via AI Horde API; if AI Horde fails, it falls back to DeepAI API.
- Publish
- The workflow moves the generated image into images/, commits it, and posts the image and caption to Instagram using the Graph API.
- State Update
- state.json is updated so the same post is not reused until the pool cycles.

## Required Secrets
Add these secrets in GitHub Settings → Secrets → Actions.
| Secret Name           | Description |
|-----------------------|-------------|
| `CEREBRAS_API_KEY`    | API key for Cerebras AI (for caption generation) |
| `DEEPAI_API_KEY`      | API key for DeepAI (for image generation fallback) |
| `IG_ACCESS_TOKEN`     | Long‑lived Instagram Graph API token |
| `IG_USER_ID`          | Instagram Business Account ID |
| `FB_APP_ID`           | Facebook App ID (may be required for some Graph API permissions) |
| `FB_APP_SECRET`       | Facebook App Secret (may be required for some Graph API permissions) |
| `PDF_BOOK_FILENAME`   | The filename of the PDF to use for context (e.g., "The-Nine-Stitches.pdf") |


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

## Setup and Configuration
- Local prerequisites
- Python 3.11 or later
- pip available
- Install dependencies
pip install requests pillow deepai

- Environment for local testing
export CEREBRAS_API_KEY="your_cerebras_api_key_here"
export DEEPAI_API_KEY="your_deepai_api_key_here"
export PDF_BOOK_FILENAME="The-Nine-Stitches.pdf" # Specify the PDF file to use for context

- Workflow configuration
- Ensure .github/workflows/auto_instagram.yml contains persist-credentials: true in the checkout step so the workflow can push commits.
- Add the required secrets to the repository.

Running Locally
You can test the bot locally to validate caption and image generation:
python bot.py


Expected outputs after a successful run:
- caption.txt — generated caption
- output.jpg — generated image saved locally

## Troubleshooting
- Caption generation fails
- Confirm `CEREBRAS_API_KEY` is set and valid.
- Check logs for any API errors from Cerebras.
- Image generation fails
- Check logs for errors from AI Horde. If AI Horde fails, it will attempt DeepAI.
- Confirm `DEEPAI_API_KEY` is set and valid if DeepAI is being used as a fallback.
- Git push fails in workflow
- Ensure `persist-credentials: true` is set in the checkout step and the workflow uses the default `GITHUB_TOKEN`. If push still fails, verify repository permissions for the token.
- Instagram publish fails
- Confirm `IG_ACCESS_TOKEN` is valid and long‑lived and that `IG_USER_ID` is the correct Business Account ID. The raw GitHub URL used for the image must be publicly accessible.
- Debugging tips
- Run `python bot.py` locally to reproduce errors.
- Inspect workflow logs for printed messages from the generation services.

## Future Enhancements
- Add Book II and Book III post sets
- Carousel and reel generation support
- Multi‑account posting and scheduling per account
- Analytics logging and engagement tracking
- JSON driven content queue with priorities and tags
- Caption style enhancer and A/B caption testing
- Improved retry and backoff for rate limits and provider errors

## License and Copyright
All content, prompts, and generated captions are © 2024–2026 M.W.E. Wigman. Unauthorized reproduction is prohibited.
