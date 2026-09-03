import json
import re
import hashlib
import logging
from html.parser import HTMLParser
from typing import Any

logger = logging.getLogger(__name__)


class HydrationStateParser(HTMLParser):
    """Extract window.__HYDRATION_STATE__ from Farfetch HTML."""

    def __init__(self):
        super().__init__()
        self._in_script = False
        self._script_id = None
        self._capture = False
        self._buffer: list[str] = []
        self.hydration_state: dict | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag == "script":
            attr_dict = dict(attrs)
            sid = attr_dict.get("id", "")
            if sid == "script_pageSettings":
                self._in_script = True
                self._script_id = sid
            # Also look for inline scripts with __HYDRATION_STATE__
            self._in_script = True
            self._script_id = attr_dict.get("id", "")

    def handle_data(self, data: str):
        if self._in_script and "window.__HYDRATION_STATE__" in data:
            self._capture = True
            self._buffer.append(data)

    def handle_endtag(self, tag: str):
        if tag == "script" and self._capture:
            raw = "".join(self._buffer)
            self._parse_hydration(raw)
            self._capture = False
            self._buffer.clear()
        if tag == "script":
            self._in_script = False
            self._script_id = None

    def _parse_hydration(self, raw: str):
        match = re.search(r'window\.__HYDRATION_STATE__\s*=\s*(\{.*?\});?\s*(?:</script>|$)', raw, re.DOTALL)
        if not match:
            # Try alternative pattern
            match = re.search(r'__HYDRATION_STATE__\s*=\s*(\{.*?\});?\s*$', raw, re.DOTALL)
        if match:
            try:
                self.hydration_state = json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse __HYDRATION_STATE__ JSON")


def extract_hydration_state(html: str) -> dict | None:
    """Extract window.__HYDRATION_STATE__ from HTML.

    The value can be either:
    - A JSON-encoded string: window.__HYDRATION_STATE__="{...}"
    - A raw JSON object: window.__HYDRATION_STATE__={...}
    """
    # Try JSON-encoded string first (most common on Farfetch)
    match = re.search(
        r'window\.__HYDRATION_STATE__\s*=\s*"((?:[^"\\]|\\.)*)"\s*;',
        html,
        re.DOTALL,
    )
    if match:
        try:
            raw = match.group(1)
            unescaped = raw.encode("utf-8").decode("unicode_escape")
            return json.loads(unescaped)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Failed to decode JSON-string hydration state: %s", e)

    # Fallback: raw JSON object
    match = re.search(
        r'window\.__HYDRATION_STATE__\s*=\s*(\{.*?\})\s*;\s*(?:</script>)',
        html,
        re.DOTALL,
    )
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("Fallback regex parse failed")

    return None


def extract_jsonld(html: str) -> list[dict]:
    """Extract JSON-LD structured data from HTML."""
    results = []
    for match in re.finditer(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', html, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
        except json.JSONDecodeError:
            continue
    return results


def resolve_apollo_ref(apollo_cache: dict, ref_str: str) -> dict | None:
    """Resolve an Apollo __ref string to its cached object.
    
    The ref value can be either:
    - "Brand:13198" (direct key)
    - "__ref:Brand:13198" (prefixed)
    """
    if not ref_str:
        return None
    # Try direct lookup first
    obj = apollo_cache.get(ref_str)
    if obj:
        return obj
    # Try stripping __ref: prefix
    if ref_str.startswith("__ref:"):
        return apollo_cache.get(ref_str[6:])
    return None


def resolve_brand(apollo_cache: dict, brand_ref: dict | None) -> str | None:
    """Resolve brand name from Apollo cache ref."""
    if not brand_ref:
        return None
    ref_str = brand_ref.get("__ref", "")
    if not ref_str:
        return None
    brand_obj = resolve_apollo_ref(apollo_cache, ref_str)
    if brand_obj:
        return brand_obj.get("name")
    return None


def build_product_id(source: str, product_url: str) -> str:
    """Generate stable product ID from source + URL."""
    raw = f"{source}:{product_url}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def extract_category_from_path(path: str) -> str | None:
    """Extract category from product URL path like /shopping/women/brand-category-item-xxx.aspx"""
    if not path:
        return None
    # Try to get the last meaningful segment before item ID
    parts = path.rstrip("/").split("/")
    if len(parts) >= 4:
        # e.g. /shopping/women/brand-productname-item-xxx.aspx
        slug = parts[-1]
        # Remove item-XXXXXX.aspx suffix
        slug = re.sub(r'-item-\d+\.aspx$', '', slug)
        # Remove brand prefix (first word or two)
        # This is heuristic; the category info from hydration is better
        return slug
    return None


def _extract_price(node: dict, apollo: dict) -> tuple[str | None, str | None]:
    """Extract full price and sale price from a product node."""
    price_obj = node.get("productPrice", {})
    full_price = None
    sale_price = None

    if not price_obj:
        return full_price, sale_price

    def _get_currency_code(price_obj: dict) -> str:
        currency = price_obj.get("currency", {})
        if isinstance(currency, dict):
            ref = currency.get("__ref", "")
            if ref:
                m = re.search(r'"isoCode":"(\w+)"', ref)
                if m:
                    return m.group(1)
        return ""

    def _format_price(raw_val: float, curr_code: str) -> str:
        return f"{raw_val:.2f}{curr_code}"

    curr_code = _get_currency_code(price_obj)

    full_info = price_obj.get("full", {})
    if full_info:
        raw_val = full_info.get("value", {}).get("raw", 0)
        if raw_val:
            full_price = _format_price(raw_val, curr_code)

    final_info = price_obj.get("final", {})
    discounts = price_obj.get("discounts", [])
    if final_info and discounts:
        raw_val = final_info.get("value", {}).get("raw", 0)
        full_raw = full_info.get("value", {}).get("raw", 0) if full_info else 0
        if raw_val and raw_val != full_raw:
            sale_price = _format_price(raw_val, curr_code)

    return full_price, sale_price


def _normalize_product_url(url: str) -> str:
    """Normalize product URL to always include /cz/ prefix for consistency."""
    if not url:
        return url
    # Ensure /cz/ prefix for canonical URLs
    if "/cz/" not in url and "farfetch.com/shopping/" in url:
        url = url.replace("farfetch.com/shopping/", "farfetch.com/cz/shopping/")
    return url


def _parse_product_edge(edge: dict, apollo: dict, gender: str) -> dict | None:
    """Parse a single product edge into a product dict."""
    node = edge.get("node", {})
    if not node:
        return None

    pid = node.get("id", "")
    path_obj = node.get("resourceIdentifier", {})
    path = path_obj.get("path", "") if isinstance(path_obj, dict) else ""
    product_url = _normalize_product_url(f"https://www.farfetch.com{path}") if path else ""

    brand_obj = node.get("brand", {})
    brand = resolve_brand(apollo, brand_obj) if isinstance(brand_obj, dict) else None

    images = node.get("images", [])
    image_url = ""
    if images:
        first_img = images[0]
        size_obj = first_img.get("size1000") or first_img.get("size480") or first_img.get("size600")
        if size_obj and isinstance(size_obj, dict):
            image_url = size_obj.get("url", "")

    full_price, sale_price = _extract_price(node, apollo)
    short_desc = node.get("shortDescription", "")

    return {
        "id": pid,
        "product_url": product_url,
        "brand": brand,
        "title": short_desc,
        "image_url": image_url,
        "price": full_price,
        "sale": sale_price,
        "gender": gender.capitalize() if gender else "",
        "category": extract_category_from_path(path),
    }


def parse_listing_products(hydration: dict) -> tuple[list[dict], dict | None]:
    """
    Parse product catalog from listing page hydration state.
    Returns (list of product summaries, pagination info).
    """
    apollo = hydration.get("apolloInitialState", {})
    products = []
    pagination = None

    # The product catalog is inside ROOT_QUERY under a key like "productCatalog:{...}"
    root_query = apollo.get("ROOT_QUERY", {})

    # Search in ROOT_QUERY first, then in top-level apollo keys
    sources = [root_query]
    sources.extend(v for k, v in apollo.items() if isinstance(v, dict) and k != "ROOT_QUERY")

    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if "productCatalog" not in key:
                continue
            if not isinstance(value, dict):
                continue

            edges = value.get("edges", [])
            if not edges:
                continue

            gender_obj = hydration.get("page", {}).get("routeValues", {})
            gender = gender_obj.get("currentRootCategory", "") if isinstance(gender_obj, dict) else ""

            for edge in edges:
                product = _parse_product_edge(edge, apollo, gender)
                if product:
                    products.append(product)

            page_info = value.get("pageInfo", {})
            if page_info:
                pagination = {
                    "has_next": page_info.get("hasNextPage", False),
                    "end_cursor": page_info.get("endCursor", ""),
                    "total_count": value.get("totalCount", 0),
                }

            if products:
                break
        if products:
            break

    return products, pagination


def parse_product_detail(html: str, product_url: str) -> dict | None:
    """
    Parse full product detail from a product page HTML.
    Uses JSON-LD as primary (simpler, more reliable) with hydration fallback.
    """
    # Try JSON-LD first
    jsonld_items = extract_jsonld(html)
    product_group = None
    for item in jsonld_items:
        if item.get("@type") == "ProductGroup":
            product_group = item
            break

    # Also extract hydration state for richer data
    hydration = extract_hydration_state(html)

    result = _parse_from_jsonld(product_group, product_url)
    if result:
        # Enrich with hydration data if available
        if hydration:
            _enrich_from_hydration(result, hydration)
        return result

    # Fallback to hydration only
    if hydration:
        return _parse_from_hydration(hydration, product_url)

    return None


def _parse_from_jsonld(product_group: dict | None, product_url: str) -> dict | None:
    """Parse product from JSON-LD ProductGroup."""
    if not product_group:
        return None

    name = product_group.get("name", "")
    brand_obj = product_group.get("brand", {})
    brand = brand_obj.get("name", "") if isinstance(brand_obj, dict) else ""
    color = product_group.get("color", "")
    description = product_group.get("description", "")

    # Images
    images = product_group.get("image", [])
    image_urls = []
    for img in images:
        if isinstance(img, dict):
            url = img.get("contentUrl", "")
            if url:
                image_urls.append(url)
        elif isinstance(img, str):
            image_urls.append(img)

    # Variants for sizes and prices
    variants = product_group.get("hasVariant", [])
    sizes = []
    prices = []
    skus = []
    for v in variants:
        size = v.get("size", "")
        if size:
            sizes.append(size)
        sku = v.get("sku", "")
        if sku:
            skus.append(sku)
        offers = v.get("offers", {})
        price_specs = offers.get("priceSpecification", [])
        for ps in price_specs:
            p = ps.get("price", 0)
            c = ps.get("priceCurrency", "")
            pt = ps.get("priceType", "")
            if p:
                prices.append({"price": p, "currency": c, "type": pt})

    # Determine full price and sale price
    full_price = None
    sale_price = None
    for p in prices:
        pt = p["type"]
        if "StrikethroughPrice" in pt:
            full_price = p
        elif not pt or pt == "":
            if not sale_price:
                sale_price = p

    # If no strikethrough, use the regular price as full
    if not full_price and sale_price:
        full_price = sale_price
        sale_price = None
    elif not full_price and prices:
        full_price = prices[0]

    price_str = ""
    sale_str = ""
    if full_price:
        price_str = f"{full_price['price']:.2f}{full_price['currency']}"
    if sale_price:
        sale_str = f"{sale_price['price']:.2f}{sale_price['currency']}"

    # Category from breadcrumbs
    breadcrumbs = product_group.get("mainEntity", {})
    if not breadcrumbs:
        breadcrumbs = {}

    category = ""
    # Try to extract from description or breadcrumbs
    bc_list = product_group.get("@graph", [])

    # Build product ID
    # Extract item ID from URL
    item_match = re.search(r'item-(\d+)', product_url)
    item_id = item_match.group(1) if item_match else ""

    return {
        "id": item_id,
        "product_url": product_url,
        "brand": brand,
        "title": name,
        "description": description,
        "color": color,
        "image_url": image_urls[0] if image_urls else "",
        "additional_images": image_urls[1:] if len(image_urls) > 1 else [],
        "sizes": sizes,
        "price": price_str,
        "sale": sale_str,
        "skus": skus,
        "metadata_json": {},
    }


def _enrich_from_hydration(result: dict, hydration: dict):
    """Enrich product result from hydration state data."""
    apollo = hydration.get("apolloInitialState", {})
    route = hydration.get("page", {}).get("routeValues", {})

    product_id = result.get("id", "")
    product_key = f"Product:{product_id}"
    product_obj = apollo.get(product_key, {})

    if not product_obj:
        return

    # Get gender from route
    gender = route.get("currentRootCategory", "")
    if gender:
        result["gender"] = gender.capitalize()

    # Get category from path array
    path_refs = product_obj.get("path", [])
    categories = []
    for pref in path_refs:
        if isinstance(pref, dict):
            ref = pref.get("__ref", "")
            cat_obj = apollo.get(ref, {})
            if cat_obj:
                cat_name = cat_obj.get("name", "")
                # Skip brand and gender categories
                if cat_name and cat_name.lower() not in [result.get("brand", "").lower(), gender.lower()]:
                    categories.append(cat_name)
    if categories:
        result["category"] = ", ".join(categories)

    # Get richer variation data
    variations = product_obj.get("variations", {})
    if isinstance(variations, dict):
        var_edges = variations.get("edges", [])
        # Get first variation for detailed data
        if var_edges:
            first_var_ref = var_edges[0] if isinstance(var_edges[0], dict) else {}
            ref_str = first_var_ref.get("__ref", "")
            var_obj = apollo.get(ref_str, {}) if ref_str else {}

            if var_obj:
                # Main color
                main_color = var_obj.get("mainColor", "")
                if main_color:
                    result["color"] = main_color

                # Brand style ID
                brand_style_id = var_obj.get("brandStyleId", "")
                if brand_style_id:
                    result["metadata_json"]["brandStyleId"] = brand_style_id

                # Composition
                composition = var_obj.get("composition", {})
                if isinstance(composition, dict):
                    parts = composition.get("parts", [])
                    materials = []
                    for part in parts:
                        for mat in part.get("materials", []):
                            mat_name = mat.get("name", "")
                            pct = mat.get("percentage")
                            if mat_name:
                                materials.append(f"{mat_name} {pct}%" if pct else mat_name)
                    if materials:
                        result["metadata_json"]["materials"] = materials

                # Description from variation
                desc_obj = var_obj.get("description", {})
                if isinstance(desc_obj, dict):
                    highlight = desc_obj.get("highlight", {})
                    if isinstance(highlight, dict):
                        rich_text = highlight.get("richTextContent", [])
                        bullets = []
                        for rt in rich_text:
                            if isinstance(rt, dict):
                                blocks = rt.get("blocks", [])
                                for block in blocks:
                                    items = block.get("items", [])
                                    for item in items:
                                        if isinstance(item, dict) and item.get("type") == "text":
                                            bullets.append(item.get("value", ""))
                        if bullets:
                            result["description"] = " | ".join(bullets)

                    short_desc = desc_obj.get("short", {})
                    if isinstance(short_desc, dict):
                        short_text = short_desc.get("textContent", "")
                        if short_text:
                            result["title"] = short_text

                # Price from variation (more accurate)
                var_price = var_obj.get("productPrice", {})
                if isinstance(var_price, dict):
                    full_info = var_price.get("full", {})
                    final_info = var_price.get("final", {})
                    discounts = var_price.get("discounts", [])

                    if full_info:
                        val = full_info.get("value", {})
                        raw = val.get("raw", 0)
                        curr_obj = var_price.get("currency", {})
                        curr_code = ""
                        if isinstance(curr_obj, dict):
                            ref = curr_obj.get("__ref", "")
                            m = re.search(r'"isoCode":"(\w+)"', ref) if ref else None
                            if m:
                                curr_code = m.group(1)
                        if raw:
                            result["price"] = f"{raw:.2f}{curr_code}"

                    if final_info and discounts:
                        val = final_info.get("value", {})
                        raw = val.get("raw", 0)
                        curr_obj = var_price.get("currency", {})
                        curr_code = ""
                        if isinstance(curr_obj, dict):
                            ref = curr_obj.get("__ref", "")
                            m = re.search(r'"isoCode":"(\w+)"', ref) if ref else None
                            if m:
                                curr_code = m.group(1)
                        if raw and raw != (full_info.get("value", {}).get("raw", 0) if full_info else 0):
                            result["sale"] = f"{raw:.2f}{curr_code}"

                # Images from variation (richer - has multiple sizes)
                var_images = var_obj.get("images", [])
                if var_images:
                    all_image_urls = []
                    for img in var_images:
                        order = img.get("order", 999)
                        size_obj = img.get("size1000") or img.get("size2048") or img.get("size600")
                        if size_obj and isinstance(size_obj, dict):
                            url = size_obj.get("url", "")
                            alt = size_obj.get("alt", "")
                            if url:
                                all_image_urls.append({"url": url, "alt": alt, "order": order})

                    if all_image_urls:
                        all_image_urls.sort(key=lambda x: x["order"])
                        result["image_url"] = all_image_urls[0]["url"]
                        result["additional_images"] = [img["url"] for img in all_image_urls[1:]]

                        # Detect back view
                        back_url = _detect_back_view(all_image_urls)
                        if back_url:
                            result["back_image_url"] = back_url

    # Enrich metadata
    result["metadata_json"]["scraped_at"] = _now_iso()


def _detect_back_view(images: list[dict]) -> str | None:
    """Detect back-view image from image list using alt text, URL patterns, or position."""
    back_keywords = ["back", "rear", "背面", "rückseite"]
    for img in images:
        url = img.get("url", "").lower()
        alt = img.get("alt", "").lower()
        for kw in back_keywords:
            if kw in url or kw in alt:
                return img.get("url")

    # If 2+ images and second has no "front" indicators, it might be back
    if len(images) >= 2:
        second = images[1]
        alt = second.get("alt", "").lower()
        if "front" not in alt and "side" not in alt:
            # Heuristic: second image might be back view
            pass

    return None


def _parse_from_hydration(hydration: dict, product_url: str) -> dict | None:
    """Parse product detail from hydration state only."""
    apollo = hydration.get("apolloInitialState", {})
    route = hydration.get("page", {}).get("routeValues", {})
    product_id = route.get("productId", "")

    if not product_id:
        item_match = re.search(r'item-(\d+)', product_url)
        product_id = item_match.group(1) if item_match else ""

    product_obj = apollo.get(f"Product:{product_id}", {})
    if not product_obj:
        return None

    # Get first variation
    variations = product_obj.get("variations", {})
    var_edges = variations.get("edges", []) if isinstance(variations, dict) else []
    first_var = {}
    if var_edges:
        ref_str = var_edges[0].get("__ref", "") if isinstance(var_edges[0], dict) else ""
        first_var = apollo.get(ref_str, {}) if ref_str else {}

    brand_ref = product_obj.get("brand", {})
    brand = resolve_brand(apollo, brand_ref) if isinstance(brand_ref, dict) else ""

    images = first_var.get("images", [])
    image_url = ""
    additional = []
    back_url = None
    if images:
        sorted_imgs = sorted(images, key=lambda x: x.get("order", 999))
        for i, img in enumerate(sorted_imgs):
            size_obj = img.get("size1000") or img.get("size600")
            if size_obj and isinstance(size_obj, dict):
                url = size_obj.get("url", "")
                if i == 0:
                    image_url = url
                else:
                    additional.append(url)
        # Detect back
        all_imgs = []
        for img in sorted_imgs:
            size_obj = img.get("size1000") or img.get("size600")
            if size_obj:
                all_imgs.append({
                    "url": size_obj.get("url", ""),
                    "alt": size_obj.get("alt", ""),
                    "order": img.get("order", 999),
                })
        back_url = _detect_back_view(all_imgs)

    gender = route.get("currentRootCategory", "").capitalize()

    # Get full price and sale
    var_price = first_var.get("productPrice", {})
    full_price = ""
    sale_price = ""
    if isinstance(var_price, dict):
        full_info = var_price.get("full", {})
        final_info = var_price.get("final", {})
        discounts = var_price.get("discounts", [])
        curr_obj = var_price.get("currency", {})
        curr_code = ""
        if isinstance(curr_obj, dict):
            ref = curr_obj.get("__ref", "")
            m = re.search(r'"isoCode":"(\w+)"', ref) if ref else None
            if m:
                curr_code = m.group(1)

        if full_info:
            raw = full_info.get("value", {}).get("raw", 0)
            if raw:
                full_price = f"{raw:.2f}{curr_code}"

        if final_info and discounts:
            raw = final_info.get("value", {}).get("raw", 0)
            full_raw = full_info.get("value", {}).get("raw", 0) if full_info else 0
            if raw and raw != full_raw:
                sale_price = f"{raw:.2f}{curr_code}"

    # Sizes
    var_props = variations.get("variationProperties", []) if isinstance(variations, dict) else []
    sizes = []
    for vp in var_props:
        for val in vp.get("values", []):
            desc = val.get("description", "")
            if desc:
                sizes.append(desc)

    # Description
    desc_obj = first_var.get("description", {})
    description = ""
    title = ""
    if isinstance(desc_obj, dict):
        short = desc_obj.get("short", {})
        if isinstance(short, dict):
            title = short.get("textContent", "")
        highlight = desc_obj.get("highlight", {})
        if isinstance(highlight, dict):
            bullets = []
            for rt in highlight.get("richTextContent", []):
                for block in rt.get("blocks", []):
                    for item in block.get("items", []):
                        if item.get("type") == "text":
                            bullets.append(item.get("value", ""))
            description = " | ".join(bullets)

    return {
        "id": product_id,
        "product_url": product_url,
        "brand": brand,
        "title": title,
        "description": description,
        "image_url": image_url,
        "additional_images": additional,
        "back_image_url": back_url,
        "sizes": sizes,
        "price": full_price,
        "sale": sale_price,
        "gender": gender,
        "metadata_json": {"scraped_at": _now_iso()},
    }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
