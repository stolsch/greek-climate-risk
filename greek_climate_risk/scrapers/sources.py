"""Source definitions and parsing selectors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SourceConfig:
    """Configuration for a single newspaper source."""

    name: str
    base_url: str
    search_url_template: str
    article_link_selector: str
    article_title_selector: str
    article_body_selector: str
    article_date_selector: str
    date_attr: str | None = None


# Greek outlets: exactly the four newspapers used in Hardouvelis–Karalas–Karanastasis–Samartzis
# (HKKS) to trace mainstream Greek print/online news — no other domains.
SOURCES: list[SourceConfig] = [
    SourceConfig(
        name="kathimerini",
        base_url="https://www.kathimerini.gr",
        search_url_template="https://www.kathimerini.gr/search/?search={query}",
        article_link_selector="a[href*='/economy/'], a[href*='/world/'], a[href*='/society/']",
        article_title_selector="h1",
        article_body_selector="article p, .article-content p, .entry-content p",
        article_date_selector="time",
        date_attr="datetime",
    ),
    SourceConfig(
        name="tovima",
        base_url="https://www.tovima.gr",
        search_url_template="https://www.tovima.gr/?s={query}",
        article_link_selector="a[href*='/202'], a[href*='/article/']",
        article_title_selector="h1",
        article_body_selector="article p, .entry-content p",
        article_date_selector="time",
        date_attr="datetime",
    ),
    SourceConfig(
        name="tanea",
        base_url="https://www.tanea.gr",
        search_url_template="https://www.tanea.gr/?s={query}",
        article_link_selector="a[href*='www.tanea.gr/20']",
        article_title_selector="h1",
        article_body_selector="article p, .entry-content p",
        article_date_selector="time",
        date_attr="datetime",
    ),
    SourceConfig(
        name="naftemporiki",
        base_url="https://www.naftemporiki.gr",
        search_url_template="https://www.naftemporiki.gr/search/?q={query}",
        article_link_selector="a[href*='/story/'], a[href*='/finance/']",
        article_title_selector="h1",
        article_body_selector="article p, .article-content p",
        article_date_selector="time",
        date_attr="datetime",
    ),
]
