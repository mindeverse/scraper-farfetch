import base64
import io
import logging
import time
import math
from typing import Optional

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

SIGLIP_API_URL = "https://api-inference.huggingface.co/models/google/siglip-base-patch16-384"
GEMINI_EMBED_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"


def _l2_normalize(vector: list[float]) -> list[float]:
    """L2-normalize a vector."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def _prepare_image(image_bytes: bytes) -> bytes:
    """
    Decode image, resize longest side to 1280px, encode as JPEG quality 85.
    Returns JPEG bytes.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")

    max_side = max(img.size)
    if max_side > 1280:
        ratio = 1280 / max_side
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def generate_image_embedding(
    image_url: str,
    hf_token: str,
    client: httpx.Client,
    delay: float = 0.5,
) -> Optional[list[float]]:
    """Download image and generate SigLIP embedding via HuggingFace Inference API."""
    try:
        # Download image
        resp = client.get(image_url, timeout=30)
        resp.raise_for_status()

        # Prepare image
        jpeg_bytes = _prepare_image(resp.content)
        b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")

        # Call HF API
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        payload = {
            "inputs": {
                "image": b64_image,
            },
        }

        time.sleep(delay)
        resp = client.post(
            SIGLIP_API_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # Handle different response formats
        embedding = _extract_embedding(data)
        if embedding is None:
            logger.warning("No embedding returned for image %s", image_url)
            return None

        # L2-normalize
        embedding = _l2_normalize(embedding)

        if len(embedding) != 768:
            logger.warning("Embedding dim %d != 768 for %s", len(embedding), image_url)
            return None

        return embedding

    except Exception as e:
        logger.error("Failed to generate image embedding for %s: %s", image_url, e)
        return None


def generate_text_embedding(
    text: str,
    hf_token: str,
    client: httpx.Client,
    delay: float = 0.5,
) -> Optional[list[float]]:
    """Generate text embedding using sentence-transformers via HuggingFace API."""
    try:
        if not text or not text.strip():
            return None

        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        payload = {"inputs": text}

        time.sleep(delay)
        resp = client.post(
            GEMINI_EMBED_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        embedding = _extract_embedding(data)
        if embedding is None:
            logger.warning("No text embedding returned")
            return None

        embedding = _l2_normalize(embedding)

        if len(embedding) != 768:
            logger.warning("Text embedding dim %d != 768", len(embedding))
            return None

        return embedding

    except Exception as e:
        logger.error("Failed to generate text embedding: %s", e)
        return None


def _extract_embedding(data) -> Optional[list[float]]:
    """Extract embedding vector from various HF API response formats."""
    if isinstance(data, list):
        # Could be batch format: [[...]] or flat: [...]
        if len(data) > 0:
            if isinstance(data[0], list):
                # Nested - average if batch
                if len(data) == 1:
                    return data[0]
                # Average across batch
                dim = len(data[0])
                avg = [0.0] * dim
                for vec in data:
                    for i in range(dim):
                        avg[i] += vec[i]
                return [x / len(data) for x in avg]
            else:
                # Flat 768-d vector
                return data

    elif isinstance(data, dict):
        # Could be {"embeddings": [...]} or {"vector": [...]}
        for key in ["embeddings", "vector", "embedding", "data"]:
            if key in data:
                val = data[key]
                return _extract_embedding(val)

    return None


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
