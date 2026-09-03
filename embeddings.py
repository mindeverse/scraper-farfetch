"""Image and text embeddings using SigLIP (768-dim).

Runs the model locally via transformers + torch — no HF API token needed.
Model: google/siglip-base-patch16-384 (free on HuggingFace).
"""

import io
import logging
import math
from typing import Optional

import httpx
import torch
from PIL import Image

try:
    from transformers import SiglipImageProcessorPil as SiglipImageProcessor
except ImportError:
    from transformers import SiglipImageProcessor
from transformers import SiglipModel, SiglipTokenizer

from config import Config

logger = logging.getLogger(__name__)

MODEL_NAME = "google/siglip-base-patch16-384"

_model = None
_image_processor = None
_tokenizer = None
_device = None


def _get_device():
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    return _device


def _load_model():
    global _model, _image_processor, _tokenizer
    if _model is None:
        logger.info("Loading SigLIP model %s...", MODEL_NAME)
        _image_processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
        _tokenizer = SiglipTokenizer.from_pretrained(MODEL_NAME)
        _model = SiglipModel.from_pretrained(MODEL_NAME)
        _model.to(_get_device())
        _model.eval()
        logger.info("SigLIP model loaded on %s", _get_device())
    return _model, _image_processor, _tokenizer


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm < 1e-10:
        return vector
    return [v / norm for v in vector]


def get_image_embedding(image_url: str) -> Optional[list[float]]:
    """Generate 768-dim L2-normalized embedding for an image using SigLIP."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    try:
        resp = httpx.get(image_url, timeout=15, headers=headers)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to load image %s: %s", image_url, e)
        return None

    model, image_processor, _ = _load_model()
    device = _get_device()

    try:
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        emb_tensor = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        embedding = emb_tensor.cpu().float().numpy().flatten().tolist()

        if len(embedding) != 768:
            logger.warning("Unexpected embedding dim %d", len(embedding))
            return None

        return _l2_normalize(embedding)

    except Exception as e:
        logger.warning("Failed to embed image: %s", e)
        return None


def get_text_embedding(text: str) -> Optional[list[float]]:
    """Generate 768-dim L2-normalized embedding for text using SigLIP text encoder."""
    if not text or not text.strip():
        return None

    model, _, tokenizer = _load_model()
    device = _get_device()

    try:
        inputs = tokenizer(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)

        emb_tensor = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        embedding = emb_tensor.cpu().float().numpy().flatten().tolist()

        if len(embedding) != 768:
            logger.warning("Unexpected text embedding dim %d", len(embedding))
            return None

        return _l2_normalize(embedding)

    except Exception as e:
        logger.warning("Failed to embed text: %s", e)
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
