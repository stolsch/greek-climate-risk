"""SQLite utilities for article storage and retrieval."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def initialize_database(db_path: Path) -> None:
    """Create articles table if missing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                date TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE
            )
            """
        )
        connection.commit()
    LOGGER.info("Database initialized at %s", db_path)


def upsert_articles(db_path: Path, articles: list[dict[str, Any]]) -> int:
    """Insert article rows while skipping duplicate URLs."""
    inserted = 0
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        for article in articles:
            cursor.execute(
                """
                INSERT OR IGNORE INTO articles (title, body, date, source, url)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    article["title"],
                    article["body"],
                    article["date"],
                    article["source"],
                    article["url"],
                ),
            )
            inserted += cursor.rowcount
        connection.commit()
    LOGGER.info("Inserted %s new articles", inserted)
    return inserted
