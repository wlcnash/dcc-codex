"""
Royal Road scraper for Dungeon Crawler Carl.
Scrapes chapter text from royalroad.com and stores in PostgreSQL.
"""

import time
import re
import logging
from dataclasses import dataclass
from typing import Optional
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

FICTION_ID = 12518
BASE_URL = "https://www.royalroad.com"
CHAPTER_LIST_URL = f"{BASE_URL}/fiction/{FICTION_ID}/dungeon-crawler-carl/chapters"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DCCCodexBot/1.0; +https://github.com/wlcnash/dcc-codex)"
    )
}

RATE_LIMIT_SECONDS = 2.0  # be polite to Royal Road


@dataclass
class Chapter:
    book_id: int
    chapter_number: int
    chapter_title: str
    url: str
    raw_text: str
    word_count: int


def get_chapter_list() -> list[dict]:
    """Return list of {title, url} for all DCC chapters on Royal Road."""
    resp = requests.get(CHAPTER_LIST_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    chapters = []
    for row in soup.select("table.table tbody tr"):
        link = row.select_one("a[href*='/chapter/']")
        if not link:
            continue
        chapters.append({
            "title": link.get_text(strip=True),
            "url": BASE_URL + link["href"].split("?")[0],  # strip query params
        })

    logger.info(f"Found {len(chapters)} chapters on Royal Road")
    return chapters


def scrape_chapter(url: str) -> Optional[str]:
    """Scrape a single chapter page and return cleaned text."""
    time.sleep(RATE_LIMIT_SECONDS)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Royal Road chapter text lives in div.chapter-content
    content_div = soup.select_one("div.chapter-content")
    if not content_div:
        logger.warning(f"No chapter-content div found at {url}")
        return None

    # Remove author notes (usually in blockquote or .spoiler elements)
    for tag in content_div.select("blockquote, .spoiler, .author-note-portlet"):
        tag.decompose()

    # Get text, normalize whitespace
    text = content_div.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def assign_book_id(chapter_number: int, book_boundaries: list[int]) -> int:
    """
    Map a chapter number (1-indexed) to a book ID.
    book_boundaries: list of chapter numbers where each new book starts.
    e.g. [1, 52, 105, 158, 211, 262, 313, 364]
    """
    for i, start in enumerate(reversed(book_boundaries)):
        book_index = len(book_boundaries) - 1 - i
        if chapter_number >= start:
            return book_index + 1  # 1-indexed book id
    return 1


def run_scraper(conn, book_boundaries: list[int]) -> int:
    """
    Main scrape loop. Fetches all chapters and inserts into DB.
    Skips chapters already present. Returns count of new chapters scraped.
    """
    cur = conn.cursor()

    # Fetch existing chapter URLs to skip
    cur.execute("SELECT url FROM chapters")
    existing_urls = {row[0] for row in cur.fetchall()}

    chapters_raw = get_chapter_list()
    new_count = 0

    for idx, chap_meta in enumerate(chapters_raw, start=1):
        url = chap_meta["url"]
        if url in existing_urls:
            logger.debug(f"Skipping already-scraped chapter: {chap_meta['title']}")
            continue

        logger.info(f"Scraping chapter {idx}: {chap_meta['title']}")
        text = scrape_chapter(url)
        if text is None:
            logger.error(f"Skipping chapter {idx} due to scrape failure")
            continue

        book_id = assign_book_id(idx, book_boundaries)
        word_count = len(text.split())

        cur.execute(
            """
            INSERT INTO chapters (book_id, chapter_number, chapter_title, url, raw_text, word_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (book_id, chapter_number) DO NOTHING
            """,
            (book_id, idx, chap_meta["title"], url, text, word_count),
        )
        conn.commit()
        new_count += 1

    cur.close()
    logger.info(f"Scraping complete. {new_count} new chapters added.")
    return new_count
