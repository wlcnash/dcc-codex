"""
DCC Codex — floor-transition extraction.

Floors, not books, are the correct narrative unit for tracking how an entity looks
"right now." Confirmed against the actual corpus before building this:
  - Mordecai takes on a new form for every guildhall/floor -- three different forms
    show up in Book 1 alone (Rat Hooligan ch.3, Bugaboo ch.35, Incubus at the start
    of Book 2). Book-level snapshots collapse these into one wrong image.
  - Floor numbers progress monotonically with no revisits: every one of 73 sampled
    "previous floor" references in the corpus is retrospective narration, never a
    "we went back to floor N." No backtracking language was found.
  - There is no clean structural marker for floor transitions (chapter titles are
    just "Chapter N"; no recurring inline header). Detecting them requires reading
    comprehension, not a keyword search -- crawlers move floors by finding a
    staircase before the current floor "collapses" on a countdown.

This is a ONE-TIME pass. The source text is finalized, published books -- it will
not change retroactively. Once floor boundaries are found and validated here, they
are permanent hard points stored in the `floors` table, not re-derived on every
pipeline run.
"""

import json
import logging
import re
import time

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

WINDOW_SIZE = 10       # chapters per Gemini call
WINDOW_OVERLAP = 2     # chapters of re-shown context between windows, for continuity
RATE_LIMIT_SECONDS = 2.0

FLOOR_SYSTEM_PROMPT = (
    "You are identifying dungeon-floor transitions in the LitRPG novel Dungeon Crawler Carl. "
    "Crawlers descend through numbered dungeon floors. They move to the next floor by finding "
    "a staircase, usually after defeating a boss or reaching an exit, before the current floor "
    "'collapses' on a timer. Each floor has its own guildhall with its own shape-shifted "
    "incarnation of the NPC Mordecai.\n\n"
    "You will be given several consecutive chapters of raw text, each labeled with its chapter_id. "
    "You are also told what floor number the party is on as of the START of this excerpt "
    "(established from prior chapters, not shown here).\n\n"
    "Find every point WITHIN this excerpt where the party moves to a NEW floor for the first time "
    "(i.e. floor number increases by exactly one from whatever the current floor is). For each "
    "one, report:\n"
    "  - new_floor_number: the floor they arrive on (must be current_floor + 1, then + 1 again for "
    "the next transition found, etc. -- floors are never skipped and never revisited)\n"
    "  - chapter_id: the chapter_id where this transition occurs\n"
    "  - evidence_quote: a short (under 30 words) VERBATIM quote from the text that is your direct "
    "evidence this is the moment of arrival on the new floor. Do not paraphrase.\n\n"
    "Be conservative: only report a transition if the text clearly shows the party arriving on a "
    "new floor (e.g. descending a staircase, a system message confirming a new floor, explicit "
    "narration of arrival). A character merely mentioning a floor number in conversation, "
    "reminiscing, or a rumor is NOT a transition. If you find no transitions in this excerpt, "
    "return an empty list.\n\n"
    "Respond with ONLY a JSON array (empty array if none), no other text. Each element: "
    '{"new_floor_number": int, "chapter_id": int, "evidence_quote": "..."}'
)


def run_migrate(conn) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS floors (
            id SERIAL PRIMARY KEY,
            floor_number INTEGER NOT NULL UNIQUE,
            start_chapter_id INTEGER NOT NULL REFERENCES chapters(id),
            book_id INTEGER NOT NULL REFERENCES books(id),
            evidence_quote TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    logger.info("Migration: floors table ensured.")


def _extract_json_array(text):
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def _call_window(chapters_window, current_floor, client):
    """chapters_window: list of (chapter_id, book_number, chapter_number, raw_text)."""
    labeled = "\n\n".join(
        f"=== chapter_id={cid} (Book {bn}, Chapter {cn}) ===\n{text}"
        for cid, bn, cn, text in chapters_window
    )
    prompt = (
        FLOOR_SYSTEM_PROMPT
        + f"\n\nThe party is currently on floor {current_floor} as of the start of this excerpt.\n\n"
        + labeled
    )
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return _extract_json_array(response.text)


def run_floor_extraction(conn, gemini_api_key: str) -> int:
    """One-time pass. Returns number of floor transitions recorded (floor 1's start plus
    every detected transition)."""
    run_migrate(conn)
    client = genai.Client(api_key=gemini_api_key)

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM floors")
    if cur.fetchone()[0] > 0:
        logger.info("floors table already populated -- this is a one-time extraction, skipping. "
                    "Delete rows manually first if you intend to re-run.")
        cur.close()
        return 0

    cur.execute("""
        SELECT c.id, b.book_number, c.chapter_number, c.raw_text, c.book_id
        FROM chapters c JOIN books b ON b.id = c.book_id
        ORDER BY b.book_number, c.chapter_number
    """)
    rows = cur.fetchall()
    cur.close()
    logger.info("Loaded %d chapters for floor extraction.", len(rows))

    # Floor 1 always starts at the very first chapter -- the dungeon activates at the
    # start of the book and everyone is immediately on floor 1.
    first_chapter_id, _, _, _, first_book_id = rows[0]
    floors_found = [(1, first_chapter_id, first_book_id, "Dungeon activation -- floor 1 start (seeded, not model-detected).")]
    current_floor = 1

    i = 0
    while i < len(rows):
        window = rows[i:i + WINDOW_SIZE]
        window_for_call = [(cid, bn, cn, text) for cid, bn, cn, text, bid in window]
        try:
            results = _call_window(window_for_call, current_floor, client)
        except Exception as e:
            logger.warning("Floor extraction window at chapter index %d failed: %s", i, e)
            i += WINDOW_SIZE - WINDOW_OVERLAP
            time.sleep(RATE_LIMIT_SECONDS)
            continue

        # chapter_id -> book_id lookup for this window
        cid_to_book = {cid: bid for cid, bn, cn, text, bid in window}

        for item in sorted(results, key=lambda r: r.get("new_floor_number", 0)):
            fn = item.get("new_floor_number")
            cid = item.get("chapter_id")
            quote = item.get("evidence_quote", "")
            if fn != current_floor + 1:
                logger.warning(
                    "Skipping non-sequential floor claim: got floor %s while on floor %d "
                    "(chapter_id=%s, quote=%r). Hard rule: floors must increase by exactly 1.",
                    fn, current_floor, cid, quote,
                )
                continue
            if cid not in cid_to_book:
                logger.warning("Skipping floor %s claim: chapter_id %s not in this window.", fn, cid)
                continue
            floors_found.append((fn, cid, cid_to_book[cid], quote))
            current_floor = fn
            logger.info("Floor %d starts at chapter_id=%s: %r", fn, cid, quote)

        i += WINDOW_SIZE - WINDOW_OVERLAP
        time.sleep(RATE_LIMIT_SECONDS)

    cur = conn.cursor()
    for fn, cid, bid, quote in floors_found:
        cur.execute(
            "INSERT INTO floors (floor_number, start_chapter_id, book_id, evidence_quote) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (floor_number) DO NOTHING",
            (fn, cid, bid, quote),
        )
    conn.commit()
    cur.close()

    logger.info("Floor extraction complete. %d floors recorded (final floor: %d).", len(floors_found), current_floor)
    return len(floors_found)
