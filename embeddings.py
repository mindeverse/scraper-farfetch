from __future__ import annotations

import base64
import io
import logging
import time
from typing import Optional

import httpx
from PIL import Image

from config import Config

logger = logging.getLogger(__name__)

_last_hf_call: float = 0.0


def _rate_limit(cfg: Config) -> None:
    global _last_hf_call
    elapsed = time.time() - _last_hf_call
    if elapsed < cfg.EMBEDDING_DELAY:
        time.sleep(cfg.EMBEDDING_DELAY - elapsed)
    _last_hf_call = time.time()


def _download_image(url: str, client: httpx.Client) -> bytes | None:
    try:
        resp = client.get(url, timeout=15)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.error("Failed to download image %s: %s", url, e)
        return None


def _process_image(image_bytes: bytes, max_side: int = 1280) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _call_hf_api(payload: dict, cfg: Config) -> list[float] | None:
    url = f"https://api-inference.huggingface.co/models/{cfg.SIGLIP_MODEL}"
    headers = {}
    if cfg.HF_TOKEN:
        headers["Authorization"] = f"Bearer {cfg.HF_TOKEN}"
    for attempt in range(cfg.MAX_RETRIES):
        try:
            _rate_limit(cfg)
            resp = httpx.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 503:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, list) and len(result) > 0:
                embedding = result[0] if isinstance(result[0], list) else result
                if isinstance(embedding, list) and len(embedding) == cfg.SIGLIP_DIM:
                    return embedding
                if isinstance(embedding, list) and len(embedding) > cfg.SIGLIP_DIM:
                    return embedding[:cfg.SIGLIP_DIM]
            logger.warning("Unexpected HF response shape: %s", type(result))
            return None
        except httpx.HTTPStatusError as e:
            logger.error("HF API error (attempt %d/%d): %s", attempt + 1, cfg.MAX_RETRIES, e)
            time.sleep(2 ** (attempt + 1))
        except Exception as e:
            logger.error("HF API call failed (attempt %d/%d): %s", attempt + 1, cfg.MAX_RETRIES, e)
            time.sleep(2 ** (attempt + 1))
    return None


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm < 1e-9:
        return vec
    return [v / norm for v in vec]


def generate_image_embedding(
    image_url: str,
    hf_token: str,
    client: httpx.Client,
    delay: float = 0.5,
    cfg: Config | None = None,
) -> Optional[list[float]]:
    """Download image and generate SigLIP embedding via HuggingFace Inference API."""
    if cfg is None:
        cfg = Config()

    image_bytes = _download_image(image_url, client)
    if not image_bytes:
        return None

    image_b64 = _process_image(image_bytes)
    payload = {
        "inputs": {
            "image": f"data:image/jpeg;base64,{image_b64}"
        },
        "options": {"wait_for_model": True}
    }

    embedding = _call_hf_api(payload, cfg)
    if embedding is None:
        return None

    return _l2_normalize(embedding)


def generate_text_embedding(
    text: str,
    hf_token: str,
    client: httpx.Client,
    delay: float = 0.5,
    cfg: Config | None = None,
) -> Optional[list[float]]:
    """Generate text embedding using SigLIP via HuggingFace API."""
    if cfg is None:
        cfg = Config()

    if not text or not text.strip():
        return None

    payload = {
        "inputs": text[:512],
        "options": {"wait_for_model": True}
    }

    embedding = _call_hf_api(payload, cfg)
    if embedding is None:
        return None

    return _l2_normalize(embedding)


def build_info_text(product: dict) -> str:
    """Build text representation for info_embedding from product metadata."""
    parts = []
    if product.get("brand"):
        parts.append(product["brand"])
    if product.get("title"):
        parts.append(product["title"])
    if product.get("description"):
        parts.append(product["description"])
    if product.get("category"):
        parts.append(f"Category: {product['category']}")
    if product.get("gender"):
        parts.append(f"For: {product['gender']}")
    if product.get("color"):
        parts.append(f"Color: {product['color']}")
    if product.get("price"):
        parts.append(f"Price: {product['price']}")

    metadata = product.get("metadata_json", {})
    if isinstance(metadata, dict):
        materials = metadata.get("materials", [])
        if materials:
            parts.append(f"Materials: {', '.join(materials)}")
        brand_style = metadata.get("brandStyleId", "")
        if brand_style:
            parts.append(f"Style: {brand_style}")

    sizes = product.get("sizes", [])
    if sizes:
        parts.append(f"Sizes: {', '.join(sizes)}")

    return " | ".join(parts)
