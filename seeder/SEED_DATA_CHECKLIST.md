# Seed / hand-entered data verification checklist

Any fact typed by hand into this codebase -- schema.sql seed INSERTS, a
hardcoded evidence_quote, a manually-transcribed title, a constant baked into
a script -- is a claim about the source material, not a fact just because it
compiles. This project has been burned by this exact failure mode before:

**The Book 7/8 title bug (2026-07-27):** `schema.sql`'s seed `INSERT INTO
books` had placeholder titles for Book 7 ("The Hive") and Book 8 ("The Great
Undying Interstitial Realm of Nightmare") that were never checked against the
real published titles ("This Inevitable Ruin" and "A Parade of Horribles").
They'd been running in production, unchallenged, since initial DB setup.
Nobody had a reason to doubt them until someone asked directly "did you make
up the titles to book 7 and 8?" and an actual check was run.

**The lesson, generalized:** the fact that seed data has been sitting in a
working system for a long time is not evidence it's correct. It's only ever
been exercised by code, not verified against source truth.

## Rule

Before committing any new hand-entered fact into `schema.sql`, a script
constant, a hardcoded doc string, or a `floors.evidence_quote`-style ground
truth record, verify it against the actual source (the scraped
`chapters.raw_text`, an official book listing, whatever the real source of
truth is for that fact) -- don't trust a placeholder, a memory of the fact, or
an LLM's confident-sounding output without checking. This is the same
principle as [[feedback_never_guess_read_source]], applied specifically to
seed/constant data instead of entity identity decisions.

## How to check a claim against `chapters.raw_text` directly

This is the strongest verification available in this project, since it
bypasses even the LLM-extracted `passages` table and hits the actual scraped
book text:

```sql
SELECT substring(c.raw_text from position('<exact phrase>' in c.raw_text) - 100 for 300)
FROM chapters c WHERE c.book_id = <N> AND c.chapter_number = <N>;
```

If the phrase isn't found verbatim, the claim is wrong or the citation
(book/chapter) is wrong -- either way, don't ship it until resolved. This
exact pattern is what confirmed the floor names ("The Bubbles", "Don't Come
in Last", "Court of the Ascendency") were real and not invented, 2026-07-27.

## What counts as "seed / hand-entered data" in this project

- `schema.sql`'s seed `INSERT` statements (book titles, book numbers, URLs).
- `floors` table rows added by hand or reviewed by hand (`evidence_quote`,
  `start_chapter_id`, `book_id`).
- Any hardcoded list of names/IDs baked into a one-off script (e.g. the
  book->floor mapping used during the floor-image test-migration, or the
  merge group tuples used with `merge_entities.py`).
- Docstring claims about what a script found or fixed, when those claims cite
  specific chapters/books/quotes -- if you write it down as fact, it should
  have actually been checked, not paraphrased from memory.

## What does NOT need this treatment

- Data produced by the extraction/classification pipeline itself (passages,
  entity aliases, permanence classifications) -- that's a different failure
  mode (LLM hallucination during extraction) covered by the grounding
  guardrails in `imager.py`/`extractor.py`/`permanence.py`, not this
  checklist. This checklist is specifically about facts a human or an agent
  typed in directly, asserting them as ground truth.
