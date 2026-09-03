import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # Brand
    BRAND: str = "Farfetch"
    SOURCE: str = "scraper-farfetch"
    BRAND_COLUMN: str = "Farfetch"
    SECOND_HAND: bool = False
    GENDER: str = "Unisex"

    # Supabase
    SUPABASE_URL: str = field(default_factory=lambda: os.environ["SUPABASE_URL"])
    SUPABASE_KEY: str = field(default_factory=lambda: os.environ["SUPABASE_KEY"])

    # HuggingFace
    HF_TOKEN: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))
    SIGLIP_MODEL: str = "google/siglip-base-patch16-384"
    SIGLIP_DIM: int = 768

    # Embedding rate limit (seconds between HF API calls)
    EMBEDDING_DELAY: float = 0.5

    # Scraping
    REQUEST_DELAY: float = 0.8
    MAX_RETRIES: int = 3
    BATCH_SIZE: int = 50
    ITEMS_PER_PAGE: int = 96
    STALE_MISS_THRESHOLD: int = 2

    # HTTP
    BASE_URL: str = "https://www.farfetch.com"
    USER_AGENT: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    HEADERS: dict = field(default_factory=lambda: {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
        "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
    })

    # Category URLs to scrape
    CATEGORY_URLS: list[str] = field(default_factory=lambda: [
        "https://www.farfetch.com/cz/shopping/men/sale/all/items.aspx",
        "https://www.farfetch.com/cz/shopping/men/clothing-2/items.aspx",
        "https://www.farfetch.com/cz/shopping/men/shoes-2/items.aspx",
        "https://www.farfetch.com/cz/shopping/men/bags-purses-2/items.aspx",
        "https://www.farfetch.com/cz/shopping/men/accessories-all-2/items.aspx",
        "https://www.farfetch.com/cz/shopping/women/items.aspx",
        "https://www.farfetch.com/cz/shopping/men/activewear-2/items.aspx",
        "https://www.farfetch.com/cz/shopping/women/sale/all/items.aspx",
        "https://www.farfetch.com/cz/shopping/women/clothing-1/items.aspx",
        "https://www.farfetch.com/cz/shopping/women/shoes-1/items.aspx",
        "https://www.farfetch.com/cz/shopping/women/bags-purses-1/items.aspx",
        "https://www.farfetch.com/cz/shopping/women/accessories-all-1/items.aspx",
        "https://www.farfetch.com/cz/shopping/women/jewellery-1/items.aspx",
        "https://www.farfetch.com/cz/shopping/women/lifestyle-1/items.aspx",
    ])

    # Logs directory
    LOG_DIR: str = "logs"


def load_config() -> Config:
    return Config()
