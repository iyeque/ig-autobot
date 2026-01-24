# ig-autobot
Instagram Automation Bot for M.W.E. Wigman
Automated posting system for a trilogy author page that generates images and captions with AI and publishes to Instagram on a schedule.

## Project Overview
This repository contains a fully automated Instagram posting workflow designed for the author M.W.E. Wigman. The system:
- Schedules and runs via GitHub Actions
- Generates captions and images using Hugging Face inference (router or api-inference)
- Publishes posts using the Instagram Graph API
- Uses a curated posts.json of post concepts inspired by The Nine Stitches
The goal is to maintain a consistent, philosophical, nature‑driven aesthetic aligned with the trilogy’s themes.

## Project Structure
      
   |bot.py — Main automation script that selects posts, generates caption and image, and writes outputs
   |posts.json — Curated list of post concepts and prompts
   |state.json — Tracks which posts have been used to avoid repeats
   |working_model.txt — Persisted working model slug (best-effort; optional)
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
- Caption: generated via Hugging Face router or api-inference; the first successful model is persisted to working_model.txt.
- Image: generated via Hugging Face provider-backed inference (e.g., replicate) using huggingface-hub and falls back to api-inference if needed.
- Publish
- The workflow moves the generated image into images/, commits it, and posts the image and caption to Instagram using the Graph API.
- State Update
- state.json is updated so the same post is not reused until the pool cycles.

## Required Secrets
Add these secrets in GitHub Settings → Secrets → Actions.
| Secret Name        | Description |
|--------------------|-------------|
| `HF_TOKEN`         | HuggingFace token for text & image generation |
| `IG_ACCESS_TOKEN`  | Long‑lived Instagram Graph API token |
| `IG_USER_ID`       | Instagram Business Account ID |
| `HF_MODEL`         |Optional default caption model slug (e.g., meta-llama/Llama-3.1-8B-Instruct)
| `SD_MODEL`         |Optional image model slug (e.g., stabilityai/stable-diffusion-xl-base-1.0)
| `REPLICATE_API_KEY`|Optional provider key if the chosen provider requires a separate API key


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
pip install --upgrade pip
pip install requests pillow openai huggingface-hub


- Environment for local testing
export HF_TOKEN="hf_xxx"
export HF_MODEL="meta-llama/Llama-3.1-8B-Instruct"   # optional
export SD_MODEL="stabilityai/stable-diffusion-xl-base-1.0"   # optional


- Workflow configuration
- Ensure .github/workflows/auto_instagram.yml contains persist-credentials: true in the checkout step so the workflow can push commits.
- Add the required secrets to the repository.

Running Locally
You can test the bot locally to validate caption and image generation:
python bot.py


Expected outputs after a successful run:
- caption.txt — generated caption
- output.jpg — generated image saved locally
If working_model.txt exists, the bot will try that model first. Delete it to force trying the configured default model.

## Troubleshooting
- Caption generation fails
- Confirm HF_TOKEN is set and has inference scope.
- Check logs for which model was tried and any HTTP status codes. The bot retries transient errors automatically.
- Image generation returns 410 or 404
- The image model slug may be removed or gated. Try a provider-backed call by setting SD_MODEL and adding a provider key if required. The bot falls back to alternate slugs if configured.
- Git push fails in workflow
- Ensure persist-credentials: true is set in the checkout step and the workflow uses the default GITHUB_TOKEN. If push still fails, verify repository permissions for the token.
- Instagram publish fails
- Confirm IG_ACCESS_TOKEN is valid and long‑lived and that IG_USER_ID is the correct Business Account ID. The raw GitHub URL used for the image must be publicly accessible.
- Debugging tips
- Run python bot.py locally to reproduce errors.
- Inspect workflow logs for printed model names and endpoints.
- Add SD_MODEL and HF_MODEL secrets to pin working slugs.

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
