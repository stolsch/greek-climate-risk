"""Orchestrate all source scrapers and persist records."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from greek_climate_risk.database import initialize_database, upsert_articles
from greek_climate_risk.scrapers.base import AsyncArticleScraper
from greek_climate_risk.scrapers.sources import SOURCES

LOGGER = logging.getLogger(__name__)


async def _run_async_scraping(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run all scrapers asynchronously and collect article dictionaries."""
    scraper_cfg = config["scraping"]
    start_date = datetime.strptime(config["global"]["start_date"], "%Y-%m-%d")
    terms = scraper_cfg["seed_queries"]
    scrapers = [
        AsyncArticleScraper(
            source=source,
            user_agent=scraper_cfg["user_agent"],
            min_delay_seconds=scraper_cfg["min_delay_seconds"],
            max_delay_seconds=scraper_cfg["max_delay_seconds"],
            request_timeout_seconds=scraper_cfg["request_timeout_seconds"],
            retries=scraper_cfg["retries"],
            max_search_pages_per_term=int(scraper_cfg.get("max_search_pages_per_term", 25)),
            block_cooldown_seconds=float(scraper_cfg.get("block_cooldown_seconds", 120.0)),
            search_concurrency=int(scraper_cfg.get("search_concurrency", 4)),
            article_concurrency=int(scraper_cfg.get("article_concurrency", 12)),
        )
        for source in SOURCES
    ]
    try:
        batches = await asyncio.gather(*(scraper.scrape(terms, start_date) for scraper in scrapers))
    finally:
        await asyncio.gather(*(scraper.close() for scraper in scrapers))
    all_articles = [article for batch in batches for article in batch]
    LOGGER.info("Total scraped across all sources: %s", len(all_articles))
    return all_articles


def run_scraping_pipeline(config: dict[str, Any], db_path: Path) -> None:
    """Initialize DB and run async scraping collection."""
    initialize_database(db_path)
    articles = asyncio.run(_run_async_scraping(config))
    if not articles:
        LOGGER.warning("No articles scraped; downstream steps may be sparse.")
        return
    upsert_articles(db_path, articles)
