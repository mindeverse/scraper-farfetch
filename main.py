import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

import httpx

from config import load_config, Config
from parser import (
    extract_hydration_state,
    parse_listing_products,
    parse_product_detail,
    build_product_id,
)
from embeddings import (
    generate_image_embedding,
    generate_text_embedding,
    build_info_text,
)
from supabase_client import SupabaseClient

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/scraper.log", mode="a"),
    ],
)
logger = logging.getLogger("farfetch-scraper")


@dataclass
class RunStats:
    new_products: int = 0
    updated_products: int = 0
    unchanged_products: int = 0
    front_embeddings: int = 0
    back_embeddings: int = 0
    text_embeddings: int = 0
    stale_deleted: int = 0
    errors: int = 0
    total_scraped: int = 0


def scrape_category(
    category_url: str,
    client: httpx.Client,
    config: Config,
) -> list[dict]:
    """Scrape all products from a single category, paginating through all pages."""
    products = []
    page = 1
    url = category_url

    while True:
        logger.info("Fetching category page %d: %s", page, url)
        try:
            resp = client.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Failed to fetch %s: %s", url, e)
            break

        html = resp.text
        hydration = extract_hydration_state(html)
        if not hydration:
            logger.warning("No hydration state found on %s", url)
            break

        page_products, pagination = parse_listing_products(hydration)
        if not page_products:
            logger.info("No products found on page %d, stopping pagination", page)
            break

        products.extend(page_products)
        logger.info("Found %d products on page %d (total so far: %d)", len(page_products), page, len(products))

        if not pagination or not pagination.get("has_next"):
            logger.info("No more pages for this category")
            break

        page += 1
        url = f"{category_url}?page={page}"
        time.sleep(config.REQUEST_DELAY)

    return products


def scrape_product_detail(
    product_url: str,
    client: httpx.Client,
    config: Config,
) -> dict | None:
    """Scrape full product detail from a product page."""
    try:
        resp = client.get(product_url, timeout=30)
        resp.raise_for_status()
        return parse_product_detail(resp.text, product_url)
    except Exception as e:
        logger.error("Failed to scrape product %s: %s", product_url, e)
        return None


def run():
    os.makedirs("logs", exist_ok=True)
    config = load_config()
    stats = RunStats()

    logger.info("=== Farfetch Scraper Starting ===")
    logger.info("Source: %s", config.SOURCE)

    sb = SupabaseClient(config)
    sb.load_existing_products()

    seen_urls: set[str] = set()

    with httpx.Client(headers=config.HEADERS, follow_redirects=True) as client:
        # Phase 1: Collect all product URLs from category listings
        all_listing_products: list[dict] = []
        for cat_url in config.CATEGORY_URLS:
            logger.info("--- Scraping category: %s ---", cat_url)
            cat_products = scrape_category(cat_url, client, config)
            all_listing_products.extend(cat_products)
            time.sleep(config.REQUEST_DELAY)

        # Deduplicate by product_url
        seen_in_run: dict[str, dict] = {}
        for p in all_listing_products:
            url = p["product_url"]
            if url not in seen_in_run:
                seen_in_run[url] = p
                stats.total_scraped += 1

        logger.info("Total unique products from listings: %d", len(seen_in_run))

        # Phase 2: Scrape product details
        products_to_upsert: list[dict] = []

        for i, (url, listing_data) in enumerate(seen_in_run.items()):
            seen_urls.add(url)
            logger.info("Scraping detail %d/%d: %s", i + 1, len(seen_in_run), url)

            detail = scrape_product_detail(url, client, config)
            if not detail:
                logger.warning("Could not scrape detail for %s, using listing data", url)
                detail = listing_data
                detail["metadata_json"] = {"scraped_at": datetime.now(timezone.utc).isoformat()}

            # Merge listing data as fallback for missing fields
            for key in ["brand", "title", "image_url", "price", "sale", "gender", "category"]:
                if not detail.get(key) and listing_data.get(key):
                    detail[key] = listing_data[key]

            products_to_upsert.append(detail)
            time.sleep(config.REQUEST_DELAY)

            # Batch upsert every BATCH_SIZE products
            if len(products_to_upsert) >= config.BATCH_SIZE:
                _process_batch(products_to_upsert, sb, config, stats)
                products_to_upsert = []

        # Process remaining
        if products_to_upsert:
            _process_batch(products_to_upsert, sb, config, stats)

        # Phase 3: Cleanup stale products
        logger.info("--- Cleaning up stale products ---")
        # Mark all seen products
        for url in seen_urls:
            sb.mark_seen(url)

        stats.stale_deleted = sb.cleanup_stale_products()

    # Print summary
    logger.info("=== RUN SUMMARY ===")
    logger.info("New products added: %d", stats.new_products)
    logger.info("Products updated: %d", stats.updated_products)
    logger.info("Products unchanged (skipped): %d", stats.unchanged_products)
    logger.info("Front embeddings generated: %d", stats.front_embeddings)
    logger.info("Back embeddings generated: %d", stats.back_embeddings)
    logger.info("Text embeddings generated: %d", stats.text_embeddings)
    logger.info("Stale products deleted: %d", stats.stale_deleted)
    logger.info("Errors / failures: %d", stats.errors)
    logger.info("=== RUN COMPLETE ===")


def _process_batch(
    products: list[dict],
    sb: SupabaseClient,
    config: Config,
    stats: RunStats,
):
    """Process a batch of products: diff, embed, upsert."""
    with httpx.Client(headers=config.HEADERS, timeout=60) as hf_client:
        for product in products:
            url = product["product_url"]
            try:
                is_new = sb.get_existing(url) is None

                if not is_new:
                    needs_update, changed = sb.should_update(url, product)
                    if not needs_update:
                        stats.unchanged_products += 1
                        continue
                    stats.updated_products += 1
                    logger.info("Product changed: %s (%d fields)", url, len(changed))
                else:
                    stats.new_products += 1
                    logger.info("New product: %s", url)

                # Determine what needs re-embedding
                new_image_url = product.get("image_url", "")
                new_back_url = product.get("back_image_url")
                do_front_embed = sb.needs_image_reembed(url, new_image_url)
                do_back_embed = sb.needs_back_reembed(url, new_back_url)
                do_info_embed = sb.needs_info_reembed(url, product)

                # Generate embeddings
                if do_front_embed and new_image_url:
                    embedding = generate_image_embedding(
                        new_image_url, config.HF_TOKEN, hf_client, cfg=config
                    )
                    if embedding:
                        product["_image_embedding"] = embedding
                        product["_embedding_version"] = 2
                        stats.front_embeddings += 1
                    else:
                        stats.errors += 1

                if do_back_embed and new_back_url:
                    back_embedding = generate_image_embedding(
                        new_back_url, config.HF_TOKEN, hf_client, cfg=config
                    )
                    if back_embedding:
                        product["_back_image_embedding"] = back_embedding
                        stats.back_embeddings += 1
                    else:
                        stats.errors += 1

                if do_info_embed:
                    info_text = build_info_text(product)
                    if info_text:
                        info_embedding = generate_text_embedding(
                            info_text, config.HF_TOKEN, hf_client, cfg=config
                        )
                        if info_embedding:
                            product["_info_embedding"] = info_embedding
                            stats.text_embeddings += 1
                        else:
                            stats.errors += 1

                # Build upsert row
                row = sb._build_row(product)

                # Add embedding fields
                if "_image_embedding" in product:
                    row["image_embedding"] = product["_image_embedding"]
                    row["embedding_version"] = 2
                if "_back_image_embedding" in product:
                    row["back_image_embedding"] = product["_back_image_embedding"]
                if "_info_embedding" in product:
                    row["info_embedding"] = product["_info_embedding"]

                # Upsert
                try:
                    sb.client.table("products").upsert(
                        row, on_conflict="source,product_url"
                    ).execute()
                except Exception as e:
                    logger.error("Upsert failed for %s: %s", url, e)
                    stats.errors += 1
                    _log_failed_product(url, str(config))

            except Exception as e:
                logger.error("Error processing %s: %s", url, e)
                stats.errors += 1
                _log_failed_product(url, str(e))


def _log_failed_product(url: str, error: str):
    """Log a failed product to the failures file."""
    os.makedirs("logs", exist_ok=True)
    with open("logs/failed_products.log", "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {url} | {error}\n")


if __name__ == "__main__":
    run()
