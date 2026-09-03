import json
import logging
from typing import Optional
from datetime import datetime, timezone

from supabase import create_client, Client

from config import Config
from parser import build_product_id

logger = logging.getLogger(__name__)


class SupabaseClient:
    def __init__(self, config: Config):
        self.config = config
        self.client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        self._existing_products: dict[str, dict] = {}  # key: product_url -> row

    def load_existing_products(self):
        """Load all existing products for this source into memory for diffing."""
        logger.info("Loading existing products from Supabase...")
        offset = 0
        limit = 1000
        while True:
            resp = (
                self.client.table("products")
                .select("*")
                .eq("source", self.config.SOURCE)
                .range(offset, offset + limit - 1)
                .execute()
            )
            rows = resp.data
            if not rows:
                break
            for row in rows:
                self._existing_products[row["product_url"]] = row
            offset += limit
            if len(rows) < limit:
                break
        logger.info("Loaded %d existing products", len(self._existing_products))

    def get_existing(self, product_url: str) -> Optional[dict]:
        return self._existing_products.get(product_url)

    def upsert_batch(self, products: list[dict]):
        """Upsert a batch of products to Supabase."""
        if not products:
            return

        rows = []
        for p in products:
            row = self._build_row(p)
            rows.append(row)

        for attempt in range(self.config.MAX_RETRIES):
            try:
                self.client.table("products").upsert(
                    rows,
                    on_conflict="source,product_url",
                ).execute()
                logger.info("Upserted batch of %d products", len(rows))
                return
            except Exception as e:
                logger.error("Upsert batch attempt %d failed: %s", attempt + 1, e)
                if attempt < self.config.MAX_RETRIES - 1:
                    import time
                    time.sleep(2 ** attempt)

        logger.error("Failed to upsert batch after %d retries", self.config.MAX_RETRIES)
        raise RuntimeError("Batch upsert failed")

    def mark_seen(self, product_url: str):
        """Mark a product as seen in the existing products map."""
        if product_url in self._existing_products:
            self._existing_products[product_url]["_seen"] = True

    def cleanup_stale_products(self) -> int:
        """
        Remove products not seen in this run.
        Returns count of deleted products.
        """
        deleted = 0
        for url, row in list(self._existing_products.items()):
            if row.get("_seen"):
                continue

            metadata = {}
            try:
                metadata = json.loads(row.get("metadata") or "{}")
            except (json.JSONDecodeError, TypeError):
                pass

            miss_count = metadata.get("scrape_miss_count", 0)

            if miss_count >= self.config.STALE_MISS_THRESHOLD:
                # Delete the product
                try:
                    self.client.table("products").delete().eq("id", row["id"]).execute()
                    deleted += 1
                    del self._existing_products[url]
                    logger.info("Deleted stale product: %s", url)
                except Exception as e:
                    logger.error("Failed to delete stale product %s: %s", url, e)
            else:
                # Increment miss count
                metadata["scrape_miss_count"] = miss_count + 1
                try:
                    self.client.table("products").update(
                        {"metadata": json.dumps(metadata)}
                    ).eq("id", row["id"]).execute()
                except Exception as e:
                    logger.error("Failed to update miss count for %s: %s", url, e)

        return deleted

    def _build_row(self, product: dict) -> dict:
        """Build a Supabase row from a product dict."""
        pid = build_product_id(self.config.SOURCE, product["product_url"])

        # Parse price strings
        price = product.get("price", "")
        sale = product.get("sale", "")

        # Build additional_images string
        additional = product.get("additional_images", [])
        back_url = product.get("back_image_url")
        if back_url and back_url not in additional:
            additional.append(back_url)
        additional_str = " , ".join(additional) if additional else None

        # Build metadata JSON
        metadata = product.get("metadata_json", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["scraped_at"] = datetime.now(timezone.utc).isoformat()

        # Reset miss count since we're seeing it
        metadata["scrape_miss_count"] = 0

        row = {
            "id": pid,
            "source": self.config.SOURCE,
            "product_url": product["product_url"],
            "affiliate_url": product.get("affiliate_url"),
            "image_url": product.get("image_url", ""),
            "compressed_image_url": product.get("compressed_image_url"),
            "back_image_url": back_url,
            "brand": product.get("brand", self.config.BRAND),
            "title": product.get("title", ""),
            "description": product.get("description"),
            "category": product.get("category"),
            "gender": product.get("gender", self.config.GENDER),
            "price": price or None,
            "sale": sale or None,
            "metadata": json.dumps(metadata),
            "size": ", ".join(product.get("sizes", [])) if product.get("sizes") else None,
            "second_hand": self.config.SECOND_HAND,
            "tags": product.get("tags"),
            "additional_images": additional_str,
            "other": product.get("other"),
        }

        return row

    def should_update(self, product_url: str, scraped: dict) -> tuple[bool, dict]:
        """
        Compare scraped product with existing DB row.
        Returns (needs_update, changed_fields).
        """
        existing = self.get_existing(product_url)
        if not existing:
            return True, {}

        changed = {}
        compare_fields = [
            "title", "description", "price", "sale", "category",
            "gender", "image_url", "back_image_url", "additional_images",
            "affiliate_url", "size",
        ]

        # Build additional_images comparison
        additional = scraped.get("additional_images", [])
        back_url = scraped.get("back_image_url")
        if back_url and back_url not in additional:
            additional_list = list(additional)
            additional_list.append(back_url)
        else:
            additional_list = list(additional)
        scraped_additional = " , ".join(additional_list) if additional_list else None

        scraped_map = {
            "title": scraped.get("title", ""),
            "description": scraped.get("description"),
            "price": scraped.get("price") or None,
            "sale": scraped.get("sale") or None,
            "category": scraped.get("category"),
            "gender": scraped.get("gender", self.config.GENDER),
            "image_url": scraped.get("image_url", ""),
            "back_image_url": scraped.get("back_image_url"),
            "additional_images": scraped_additional,
            "affiliate_url": scraped.get("affiliate_url"),
            "size": ", ".join(scraped.get("sizes", [])) if scraped.get("sizes") else None,
        }

        for field in compare_fields:
            scraped_val = scraped_map.get(field)
            existing_val = existing.get(field)
            if scraped_val != existing_val:
                changed[field] = scraped_val

        # Check metadata changes
        existing_meta = {}
        try:
            existing_meta = json.loads(existing.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            pass
        scraped_meta = scraped.get("metadata_json", {})
        if isinstance(scraped_meta, dict):
            for mk in scraped_meta:
                if mk == "scraped_at":
                    continue
                if scraped_meta.get(mk) != existing_meta.get(mk):
                    changed[f"metadata_{mk}"] = scraped_meta[mk]

        return bool(changed), changed

    def needs_image_reembed(self, product_url: str, new_image_url: str) -> bool:
        """Check if image_embedding needs regeneration."""
        existing = self.get_existing(product_url)
        if not existing:
            return True
        return existing.get("image_url") != new_image_url

    def needs_back_reembed(self, product_url: str, new_back_url: Optional[str]) -> bool:
        """Check if back_image_embedding needs regeneration."""
        existing = self.get_existing(product_url)
        if not existing:
            return bool(new_back_url)
        existing_back = existing.get("back_image_url")
        return existing_back != new_back_url

    def needs_info_reembed(self, product_url: str, scraped: dict) -> bool:
        """Check if info_embedding needs regeneration."""
        existing = self.get_existing(product_url)
        if not existing:
            return True

        text_fields = ["title", "description", "category", "gender", "price", "sale"]
        for f in text_fields:
            if scraped.get(f) != existing.get(f):
                return True

        # Check metadata changes
        existing_meta = {}
        try:
            existing_meta = json.loads(existing.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            pass
        scraped_meta = scraped.get("metadata_json", {})
        if isinstance(scraped_meta, dict):
            for mk in ["materials", "brandStyleId"]:
                if scraped_meta.get(mk) != existing_meta.get(mk):
                    return True

        return False

    def update_product(self, product_url: str, updates: dict):
        """Update specific fields of a product."""
        pid = build_product_id(self.config.SOURCE, product_url)
        try:
            self.client.table("products").update(updates).eq("id", pid).execute()
        except Exception as e:
            logger.error("Failed to update product %s: %s", product_url, e)
