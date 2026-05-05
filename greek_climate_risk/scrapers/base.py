"""Shared async scraping utilities."""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, quote_plus, urlencode, urljoin, urlparse, urlunparse
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
        max_search_pages_per_term: int,
        block_cooldown_seconds: float,
        search_concurrency: int,
        article_concurrency: int,
    ) -> None:
        """Initialize scraper settings and robot rules."""
        self.source = source
        self.retries = retries
        self.min_delay_seconds = min_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.max_search_pages_per_term = max(1, int(max_search_pages_per_term))
        self.block_cooldown_seconds = float(block_cooldown_seconds)
        self.search_concurrency = max(1, int(search_concurrency))
        self.article_concurrency = max(1, int(article_concurrency))
        self.client = httpx.AsyncClient(
            timeout=request_timeout_seconds,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )
        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(urljoin(source.base_url, "/robots.txt"))
        self._robots_disallow_logged: set[str] = set()

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
        allowed = self.robot_parser.can_fetch("*", path)
        if not allowed and path not in self._robots_disallow_logged:
            LOGGER.warning("Robots disallow URL path on %s: %s", self.source.name, path)
            self._robots_disallow_logged.add(path)
        return allowed

    async def _adaptive_backoff(self, attempt: int, retry_after: float | None = None) -> None:
        """Sleep with increasing cooldown after transient failures."""
        base = random.uniform(self.min_delay_seconds, self.max_delay_seconds)
        exp = min(60.0, base * (2 ** max(0, attempt - 1)))
        wait_for = max(exp, retry_after or 0.0)
        await asyncio.sleep(wait_for)

    async def _fetch(self, url: str) -> str | None:
        """Fetch URL with retries and graceful error handling."""
        if not self._is_allowed(url):
            return None
        for attempt in range(1, self.retries + 1):
            try:
                response = await self.client.get(url)
                if response.status_code in (429, 403, 503):
                    retry_after_raw = response.headers.get("Retry-After")
                    retry_after = float(retry_after_raw) if retry_after_raw and retry_after_raw.isdigit() else None
                    LOGGER.warning(
                        "Possible anti-bot block (%s) for %s (attempt %s/%s). Cooling down.",
                        response.status_code,
                        url,
                        attempt,
                        self.retries,
                    )
                    await self._adaptive_backoff(attempt, retry_after=retry_after)
                    continue
                response.raise_for_status()
                await self._delay()
                return response.text
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in (404, 410):
                    LOGGER.info("Skipping non-retryable %s for %s", status_code, url)
                    return None
                LOGGER.warning(
                    "HTTP status failure for %s (attempt %s/%s): %s",
                    url,
                    attempt,
                    self.retries,
                    exc,
                )
                await self._adaptive_backoff(attempt)
            except httpx.HTTPError as exc:
                LOGGER.warning(
                    "Fetch failed for %s (attempt %s/%s): %s",
                    url,
                    attempt,
                    self.retries,
                    exc,
                )
                await self._adaptive_backoff(attempt)
        LOGGER.warning("Retries exhausted for %s; entering cooldown of %.1fs.", url, self.block_cooldown_seconds)
        await asyncio.sleep(self.block_cooldown_seconds)
        return None

    @staticmethod
    def _set_query_param(url: str, key: str, value: int) -> str:
        """Return URL with one query parameter replaced/inserted."""
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query[key] = str(value)
        return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))

    def _candidate_search_urls(self, base_search_url: str, page_number: int) -> list[str]:
        """Generate likely paginated search URLs for one result page."""
        if page_number <= 1:
            return [base_search_url]
        return [self._set_query_param(base_search_url, "page", page_number)]

    def _extract_next_search_pages(self, soup: BeautifulSoup) -> list[str]:
        """Extract explicit next-page links from search results page."""
        next_urls: list[str] = []
        for selector in ("a[rel='next']", "a.next", ".next a", "a[aria-label*='Next']"):
            for anchor in soup.select(selector):
                href = anchor.get("href")
                if href:
                    next_urls.append(urljoin(self.source.base_url, href))
        return next_urls

    async def _collect_links_for_search_page(self, search_url: str) -> tuple[set[str], list[str], bool]:
        """Fetch one search page and return article links plus next-page candidates."""
        html = await self._fetch(search_url)
        if not html:
            return set(), [], False
        soup = BeautifulSoup(html, "html.parser")
        page_links: set[str] = set()
        for anchor in soup.select(self.source.article_link_selector):
            href = anchor.get("href")
            if not href:
                continue
            page_links.add(urljoin(self.source.base_url, href))
        return page_links, self._extract_next_search_pages(soup), True

    async def scrape(self, search_terms: list[str], start_date: datetime) -> list[dict[str, Any]]:
        """Scrape all candidate articles matching source query and date window."""
        await self._load_robots()
        links: set[str] = set()
        for term in search_terms:
            encoded = quote_plus(term)
            base_search_url = self.source.search_url_template.format(query=encoded)
            if not self._is_allowed(base_search_url):
                continue
            visited_search_urls: set[str] = set()
            pending_search_urls: list[str] = [base_search_url]
            pages_processed = 0

            while pending_search_urls and pages_processed < self.max_search_pages_per_term:
                batch: list[str] = []
                while pending_search_urls and len(batch) < self.search_concurrency:
                    candidate = pending_search_urls.pop(0)
                    if candidate in visited_search_urls:
                        continue
                    visited_search_urls.add(candidate)
                    batch.append(candidate)
                if not batch:
                    continue
                page_results = await asyncio.gather(
                    *(self._collect_links_for_search_page(url) for url in batch)
                )
                for page_links, discovered_next_urls, fetch_ok in page_results:
                    if not fetch_ok:
                        continue
                    if pages_processed >= self.max_search_pages_per_term:
                        break
                    pages_processed += 1
                    links.update(page_links)
                    page_number = pages_processed + 1
                    for candidate in self._candidate_search_urls(base_search_url, page_number):
                        if candidate not in visited_search_urls and candidate not in pending_search_urls:
                            pending_search_urls.append(candidate)
                    for next_url in discovered_next_urls:
                        if next_url not in visited_search_urls and next_url not in pending_search_urls:
                            pending_search_urls.append(next_url)

        parsed_articles: list[dict[str, Any]] = []
        parse_semaphore = asyncio.Semaphore(self.article_concurrency)

        async def _parse_with_limit(link: str) -> dict[str, Any] | None:
            async with parse_semaphore:
                return await self._parse_article(link)

        parsed_batch = await asyncio.gather(*(_parse_with_limit(link) for link in sorted(links)))
        min_date_iso = start_date.date().isoformat()
        for article in parsed_batch:
            if not article:
                continue
            if article["date"] < min_date_iso:
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
