"""
Gemini-powered entity and passage extractor for DCC Codex.

For each chapter, asks Gemini Flash to:
1. Identify named entities (characters, creatures, items, locations, abilities, factions)
2. Extract exact passages that describe each entity's physical appearance
3. Return structured JSON

Entity deduplication is handled by normalizing names and checking the DB.
"""

import json
import logging
import re
import time
from typing import Optional
import psycopg2
from psycopg2.extras import execute_values
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

VALID_ENTITY_TYPES = {"character", "creature", "item", "location", "floor", "ability", "faction", "other"}
VALID_PASSAGE_TYPES = {"physical", "personality", "backstory", "ability", "action", "other"}

# Note: {{ and }} are escaped braces for str.format(); {chapter_text} is the real placeholder
EXTRACTION_PROMPT = """You are a precise literary analyst processing the LitRPG web novel "Dungeon Crawler Carl."

Analyze the chapter text below and extract ALL named entities — characters, creatures, items, locations, floors, abilities, and factions.

For each entity, identify passages from the text that describe:
- Physical appearance (most important — size, color, shape, materials, anatomy, AND current clothing/gear/equipment status,
  even if that detail is embedded in dialogue or an action beat rather than stated as plain description.
  Examples of physical passages you MUST catch: a character putting on, taking off, receiving, losing, or declining an item
  of clothing/armor/gear ("You have boots now!" / "I pulled my scorched boxers off and slipped the new ones on");
  a wound, injury, or debuff being described, worsening, OR resolving/healing ("I sat up" after being critically hurt is a
  physical-state passage just as much as the injury itself was); a character noting they are barefoot, dirty, bloodied,
  bandaged, freshly healed, etc. Do not restrict "physical" to static, isolated description sentences — capture status
  CHANGES to a character's body or gear wherever they occur in the text, dialogue included.
- Personality traits
- Backstory or origin
- Abilities or powers
- Notable actions

IMPORTANT RULES:
- Include ONLY text that literally appears in the chapter (exact quotes)
- Only extract entities with at least one descriptive passage
- Use exact entity names as used by the author
- Include aliases if the text uses multiple names for the same entity

Return a JSON object matching this exact schema:
{{
  "entities": [
    {{
      "name": "string (canonical name)",
      "entity_type": "character|creature|item|location|floor|ability|faction|other",
      "aliases": ["list of alternate names used in this chapter"],
      "is_major": true/false,
      "passages": [
        {{
          "passage_text": "exact verbatim text from chapter",
          "passage_type": "physical|personality|backstory|ability|action|other",
          "context_before": "up to 200 chars before the passage",
          "context_after": "up to 200 chars after the passage"
        }}
      ]
    }}
  ]
}}

Only return the JSON object, no other text.

CHAPTER TEXT:
{chapter_text}
"""

RATE_LIMIT_SECONDS = 1.0  # Gemini Flash has generous rate limits


def slugify(name: str) -> str:
    """Convert entity name to URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def extract_from_chapter(chapter_text: str, client) -> Optional[dict]:
    """Call Gemini to extract entities and passages from a chapter."""
    prompt = EXTRACTION_PROMPT.format(chapter_text=chapter_text[:50000])  # cap at ~50k chars

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,  # low temp for factual extraction
            ),
        )
        time.sleep(RATE_LIMIT_SECONDS)
        return json.loads(response.text)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Gemini extraction failed: {e}")
        return None


def upsert_entity(cur, entity_data: dict, first_book_id: int, first_chapter_id: int) -> Optional[int]:
    """Insert or update an entity. Returns entity_id. Raises on DB error (caller uses savepoint)."""
    name = entity_data["name"].strip()
    if not name:
        return None

    entity_type = entity_data.get("entity_type", "other")
    if entity_type not in VALID_ENTITY_TYPES:
        entity_type = "other"

    slug = slugify(name)
    aliases = entity_data.get("aliases", [])
    is_major = entity_data.get("is_major", False)

    # Try to find existing entity by name or alias
    cur.execute(
        "SELECT id FROM entities WHERE name = %s OR %s = ANY(aliases)",
        (name, name),
    )
    row = cur.fetchone()
    if row:
        entity_id = row[0]
        # Update aliases if new ones found
        if aliases:
            cur.execute(
                "UPDATE entities SET aliases = array(SELECT DISTINCT unnest(aliases || %s::text[])) WHERE id = %s",
                (aliases, entity_id),
            )
        return entity_id

    # Ensure slug is unique by appending a counter if needed
    base_slug = slug
    counter = 1
    while True:
        cur.execute("SELECT id FROM entities WHERE slug = %s", (slug,))
        if not cur.fetchone():
            break
        slug = f"{base_slug}-{counter}"
        counter += 1

    cur.execute(
        """
        INSERT INTO entities (name, slug, entity_type, aliases, first_book_id, first_chapter_id, is_major)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (name) DO UPDATE SET
            aliases = array(SELECT DISTINCT unnest(entities.aliases || EXCLUDED.aliases)),
            is_major = entities.is_major OR EXCLUDED.is_major
        RETURNING id
        """,
        (name, slug, entity_type, aliases, first_book_id, first_chapter_id, is_major),
    )
    row = cur.fetchone()
    return row[0] if row else None


def insert_passages(cur, entity_id: int, chapter_id: int, passages: list[dict]):
    """Insert passages for an entity, deduplicating by text."""
    if not passages:
        return

    # Get existing passages for this entity+chapter to dedup
    cur.execute(
        "SELECT passage_text FROM passages WHERE entity_id = %s AND chapter_id = %s",
        (entity_id, chapter_id),
    )
    existing_texts = {row[0] for row in cur.fetchall()}

    rows = []
    for p in passages:
        text = p.get("passage_text", "").strip()
        if not text or text in existing_texts:
            continue

        ptype = p.get("passage_type", "other")
        if ptype not in VALID_PASSAGE_TYPES:
            ptype = "other"

        rows.append((
            entity_id,
            chapter_id,
            text,
            ptype,
            p.get("context_before", "")[:200],
            p.get("context_after", "")[:200],
        ))

    if rows:
        execute_values(
            cur,
            """
            INSERT INTO passages (entity_id, chapter_id, passage_text, passage_type, context_before, context_after)
            VALUES %s
            """,
            rows,
        )


def run_extractor(conn, gemini_api_key: str, batch_size: int = 10):
    """
    Main extraction loop. Processes unextracted chapters in batches.
    Skips chapters that have already had entities extracted.
    """
    client = genai.Client(api_key=gemini_api_key)

    cur = conn.cursor()

    # Find chapters not yet extracted (no passages link to them)
    cur.execute(
        """
        SELECT c.id, c.book_id, c.chapter_number, c.chapter_title, c.raw_text
        FROM chapters c
        WHERE NOT EXISTS (
            SELECT 1 FROM passages p WHERE p.chapter_id = c.id
        )
        ORDER BY c.book_id, c.chapter_number
        LIMIT %s
        """,
        (batch_size,),
    )
    chapters = cur.fetchall()

    if not chapters:
        logger.info("No unextracted chapters found.")
        cur.close()
        return 0

    logger.info(f"Extracting entities from {len(chapters)} chapters...")
    total_entities = 0
    total_passages = 0

    for chap_id, book_id, chap_num, chap_title, raw_text in chapters:
        logger.info(f"  Processing chapter {chap_num}: {chap_title}")
        result = extract_from_chapter(raw_text, client)

        if not result or "entities" not in result:
            logger.warning(f"  No entities extracted from chapter {chap_num}")
            continue

        chapter_entities = 0
        chapter_passages = 0

        for entity_data in result["entities"]:
            # Use a savepoint per entity so one failure doesn't abort the chapter's transaction
            try:
                cur.execute("SAVEPOINT sp_entity")
                entity_id = upsert_entity(cur, entity_data, book_id, chap_id)
                if entity_id is None:
                    cur.execute("RELEASE SAVEPOINT sp_entity")
                    continue
                passages = entity_data.get("passages", [])
                insert_passages(cur, entity_id, chap_id, passages)
                cur.execute("RELEASE SAVEPOINT sp_entity")
                total_entities += 1
                total_passages += len(passages)
                chapter_entities += 1
                chapter_passages += len(passages)
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT sp_entity")
                logger.warning(f"  Skipped entity '{entity_data.get('name', '?')}': {e}")

        conn.commit()
        logger.info(f"  Committed chapter {chap_num}: {chapter_entities} entities, {chapter_passages} passages")

    cur.close()
    logger.info(f"Extraction complete. {total_entities} entity refs, {total_passages} passages.")
    return total_entities


def run_extractor_forced(conn, gemini_api_key: str, chapter_ids: list[int]):
    """
    Force re-extraction on specific chapters regardless of whether they already
    have passages. Safe to run on already-processed chapters: insert_passages()
    dedupes by exact passage text per (entity, chapter), so this only ADDS
    passages the (possibly improved) prompt catches that the original pass
    missed -- it never deletes or duplicates existing rows.
    """
    client = genai.Client(api_key=gemini_api_key)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, book_id, chapter_number, chapter_title, raw_text FROM chapters WHERE id = ANY(%s) ORDER BY book_id, chapter_number",
        (chapter_ids,),
    )
    chapters = cur.fetchall()
    cur.close()

    if not chapters:
        logger.info("No matching chapters found for forced re-extraction.")
        return 0

    logger.info(f"Force re-extracting {len(chapters)} chapters...")
    total_new_passages = 0

    for chap_id, book_id, chap_num, chap_title, raw_text in chapters:
        logger.info(f"  Re-processing chapter {chap_num}: {chap_title}")
        result = extract_from_chapter(raw_text, client)
        if not result or "entities" not in result:
            logger.warning(f"  No entities extracted from chapter {chap_num}")
            continue

        cur = conn.cursor()
        chapter_new = 0
        for entity_data in result["entities"]:
            try:
                cur.execute("SAVEPOINT sp_entity")
                entity_id = upsert_entity(cur, entity_data, book_id, chap_id)
                if entity_id is None:
                    cur.execute("RELEASE SAVEPOINT sp_entity")
                    continue
                passages = entity_data.get("passages", [])
                cur.execute("SELECT COUNT(*) FROM passages WHERE entity_id=%s AND chapter_id=%s", (entity_id, chap_id))
                before = cur.fetchone()[0]
                insert_passages(cur, entity_id, chap_id, passages)
                cur.execute("SELECT COUNT(*) FROM passages WHERE entity_id=%s AND chapter_id=%s", (entity_id, chap_id))
                after = cur.fetchone()[0]
                chapter_new += (after - before)
                cur.execute("RELEASE SAVEPOINT sp_entity")
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT sp_entity")
                logger.warning(f"  Skipped entity '{entity_data.get('name', '?')}': {e}")
        conn.commit()
        cur.close()
        logger.info(f"  Chapter {chap_num}: {chapter_new} new passages")
        total_new_passages += chapter_new

    logger.info(f"Forced re-extraction complete. {total_new_passages} new passages added.")
    return total_new_passages
