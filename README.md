# Farfetch Scraper

Production-grade fashion product scraper for Farfetch, built for the Finds app.

## What it does

- Scrapes all products from 14 Farfetch category pages (men + women)
- Extracts full product metadata (brand, title, description, prices, sizes, materials, images)
- Generates 768-d SigLIP image embeddings for visual search
- Generates text embeddings for hybrid search
- Supports dual-view (front + back) embeddings
- Smart upsert with diffing — only re-embeds changed products
- Stale product cleanup after 2 consecutive missed runs

## Setup

```bash
cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_KEY, HF_TOKEN
pip install -r requirements.txt
python main.py
```

## GitHub Actions

Runs automatically 3x/week (Tue, Thu, Sun at 8AM UTC) via `.github/workflows/scrape.yml`.

Set these repository secrets:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `HF_TOKEN` (optional, HuggingFace token for higher rate limits)

## Architecture

| File | Purpose |
|------|---------|
| `main.py` | Orchestrator — category scraping, detail scraping, embedding, upsert |
| `config.py` | Environment-based configuration |
| `parser.py` | HTML parsing — extracts `window.__HYDRATION_STATE__` + JSON-LD |
| `embeddings.py` | SigLIP image + text embeddings via HuggingFace API |
| `supabase_client.py` | Batch upsert, smart diffing, stale cleanup |

## Back-View Detection

The scraper detects back-view images using:
1. Alt text containing "back" / "rear" keywords
2. URL patterns with "back" / "rear" suffixes
3. Image position heuristics in the gallery

Back images are stored in `back_image_url` + `back_image_embedding`. The primary `image_url` always remains the front packshot.
