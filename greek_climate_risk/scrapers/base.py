"""Shared async scraping utilities."""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from greek_climate_risk.scrapers.sources import SourceConfig

LOGGER = logging.getLogger(__name__)


class AsyncArticleScraper:
    """Scrape source search pages and parse individual article pages."""

    def __init__(
        self,
        source: SourceConfig,
        user_agent: str,
        min_delay_seconds: float,
        max_delay_seconds: float,
        request_timeout_seconds: float,
        retries: int,
    ) -> None:
        """Initialize scraper settings and robot rules."""
        self.source = source
        self.retries = retries
        self.min_delay_seconds = min_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.client = httpx.AsyncClient(
            timeout=request_timeout_seconds,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )
        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(urljoin(source.base_url, "/robots.txt"))

    async def close(self) -> None:
        """Close underlying HTTP client."""
        await self.client.aclose()

    async def _delay(self) -> None:
        """Sleep between requests to avoid aggressive crawling."""
        await asyncio.sleep(random.uniform(self.min_delay_seconds, self.max_delay_seconds))

    async def _load_robots(self) -> None:
        """Load robots.txt in a thread-safe way."""
        await asyncio.to_thread(self.robot_parser.read)

    def _is_allowed(self, url: str) -> bool:
        """Check if robots allow crawling this URL."""
        path = urlparse(url).path or "/"
        return self.robot_parser.can_fetch("*", path)

    async def _fetch(self, url: str) -> str | None:
        """Fetch URL with retries and graceful error handling."""
        if not self._is_allowed(url):
            LOGGER.warning("Robots disallow URL: %s", url)
            return None
        for attempt in range(1, self.retries + 1):
            try:
                response = await self.client.get(url)
                response.raise_for_status()
                await self._delay()
                return response.text
            except httpx.HTTPError as exc:
                LOGGER.warning(
                    "Fetch failed for %s (attempt %s/%s): %s",
                    url,
                    attempt,
                    self.retries,
                    exc,
                )
                await self._delay()
        return None

    async def scrape(self, search_terms: list[str], start_date: datetime) -> list[dict[str, Any]]:
        """Scrape all candidate articles matching source query and date window."""
        await self._load_robots()
        links: set[str] = set()
        for term in search_terms:
            encoded = quote_plus(term)
            search_url = self.source.search_url_template.format(query=encoded)
            html = await self._fetch(search_url)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            for anchor in soup.select(self.source.article_link_selector):
                href = anchor.get("href")
                if not href:
                    continue
                links.add(urljoin(self.source.base_url, href))

        parsed_articles: list[dict[str, Any]] = []
        for link in sorted(links):
            article = await self._parse_article(link)
            if not article:
                continue
            if article["date"] < start_date.date().isoformat():
                continue
            parsed_articles.append(article)
        LOGGER.info("%s scraped %s articles", self.source.name, len(parsed_articles))
        return parsed_articles

    async def _parse_article(self, url: str) -> dict[str, Any] | None:
        """Parse article page into normalized dictionary."""
        html = await self._fetch(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        title_node = soup.select_one(self.source.article_title_selector)
        body_nodes = soup.select(self.source.article_body_selector)
        date_node = soup.select_one(self.source.article_date_selector)

        title = title_node.get_text(" ", strip=True) if title_node else ""
        body = " ".join(node.get_text(" ", strip=True) for node in body_nodes).strip()
        if not title or not body:
            return None

        date_text = ""
        if date_node:
            if self.source.date_attr:
                date_text = (date_node.get(self.source.date_attr) or "").strip()
            if not date_text:
                date_text = date_node.get_text(" ", strip=True)
        parsed_date = self._parse_date(date_text)
        if not parsed_date:
            parsed_date = datetime.utcnow().date().isoformat()

        return {
            "title": title,
            "body": body,
            "date": parsed_date,
            "source": self.source.name,
            "url": url,
        }

    @staticmethod
    def _parse_date(raw_date: str) -> str | None:
        """Parse loose date strings and return ISO date."""
        if not raw_date:
            return None
        cleaned = raw_date.replace("Z", "+00:00")
        for fmt in (
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%d/%m/%Y",
            "%d-%m-%Y",
        ):
            try:
                return datetime.strptime(cleaned, fmt).date().isoformat()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(cleaned).date().isoformat()
        except ValueError:
            return None
