"""
DCC Codex -- structured stat/skill extraction from existing 'ability'-typed passages.

This does NOT read raw book text directly. It re-parses passages that the extraction
pipeline already scoped and typed as passage_type='ability' (item enchantment effects,
spell/skill descriptions, numeric stat callouts like "an intelligence of 150") into
structured (entity, floor, stat_or_skill, value, reason) rows, matching the reference
structure used by the highest-engagement fan wiki for this exact series (Dungeon Crawler
Carl Wiki on Fandom): a Stats table (STR/INT/CON/DEX/CHA per floor/book with a cited
reason for each change) and a Skills table (skill name, level, origin, reference),
explicitly showing "?"/unknown rather than guessing when a level or value isn't stated.

Same "never trust unvalidated LLM output" discipline as every other seeder module
(floors.py, permanence.py, entity_resolution.py, persona.py): the model is only ever
asked to point at which of the ALREADY-EXTRACTED passages support a given stat/skill
fact and to state the value/level ONLY if explicitly given in that passage. Any output
that doesn't validate cleanly (fabricated passage_id, out-of-range stat name, non-integer
value) is dropped, never silently coerced.
"""

import json
import logging
import re
import time

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 1.0
CORE_STATS = {"STR", "INT", "CON", "DEX", "CHA"}

EXTRACT_SYSTEM = (
    "You extract structured game-stat and skill facts from Dungeon Crawler Carl book passages "
    "that have ALREADY been identified as describing an ability, enchantment, or stat effect. "
    "You will be given a numbered list of passages (each with a passage_id) for ONE entity. "
    "Your job: for each passage, decide whether it explicitly states (a) a numeric value for one "
    "of the five core Player Stats (STR, INT, CON, DEX, CHA -- Strength, Intelligence, Constitution, "
    "Dexterity, Charisma) -- e.g. 'an intelligence of 150', '+4 to Constitution', 'Strength 6' -- "
    "or (b) a named skill/spell and, if stated, its level -- e.g. 'adds a single level to the "
    "Determine Value skill', 'Iron Punch Skill'.\n\n"
    "STRICT RULES:\n"
    "- Only extract what is EXPLICITLY stated in the passage text given. Never infer, estimate, or "
    "guess a numeric value that isn't written down.\n"
    "- If a skill or ability is named but no level/value is given, still emit it with value=null -- "
    "do not omit it and do not invent a plausible-sounding number.\n"
    "- Every entry must cite the exact passage_id (from the list given) it came from. Never invent a "
    "passage_id.\n"
    "- A single passage may yield zero, one, or multiple entries (e.g. an item granting both +4 CON "
    "and a skill level).\n"
    "- 'reason' must be a short (<20 word) paraphrase of what in the passage caused this stat/skill "
    "(e.g. 'wearing the Enchanted Trollskin Shirt of Pummeling').\n\n"
    "Respond with ONLY a JSON array, no other text. Each element:\n"
    '{"passage_id": <int>, "kind": "stat"|"skill", "name": "<STR|INT|CON|DEX|CHA or free-text skill name>", '
    '"value": <int or null>, "reason": "<short phrase>"}\n'
    "If nothing in the given passages qualifies, return an empty array []."
)


def run_migrate(conn) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entity_stats (
            id SERIAL PRIMARY KEY,
            entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            floor_id INTEGER REFERENCES floors(id),
            stat_name VARCHAR(20) NOT NULL,
            value INTEGER,
            reason TEXT,
            source_passage_id INTEGER REFERENCES passages(id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_stats_entity ON entity_stats(entity_id)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entity_skills (
            id SERIAL PRIMARY KEY,
            entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            floor_id INTEGER REFERENCES floors(id),
            skill_name VARCHAR(255) NOT NULL,
            level INTEGER,
            origin VARCHAR(100),
            reason TEXT,
            source_passage_id INTEGER REFERENCES passages(id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_skills_entity ON entity_skills(entity_id)")
    cur.execute("ALTER TABLE entities ADD COLUMN IF NOT EXISTS stats_extracted_at TIMESTAMP")
    conn.commit()
    cur.close()
    logger.info("Migration: entity_stats, entity_skills, entities.stats_extracted_at ensured.")


def _fetch_ability_passages(conn, entity_id):
    """Return list of (passage_id, passage_text, floor_id, floor_number) for this entity's
    ability-typed passages, floor-resolved via chapter_floors (same view imager.py uses)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id, p.passage_text, cf.floor_id, cf.floor_number
        FROM passages p
        JOIN chapter_floors cf ON cf.chapter_id = p.chapter_id
        WHERE p.entity_id = %s AND p.passage_type = 'ability'
        ORDER BY cf.floor_number, p.id
    """, (entity_id,))
    rows = cur.fetchall()
    cur.close()
    return rows


def _extract_for_entity(client, entity_name, entity_type, passages):
    """passages: list of (passage_id, passage_text, floor_id, floor_number).
    Returns validated list of dicts: {passage_id, kind, name, value, reason, floor_id, floor_number}."""
    if not passages:
        return []

    numbered = "\n\n".join(
        f"[passage_id={pid}] (Floor {fn}): \"{text}\"" for pid, text, _, fn in passages
    )
    valid_ids = {pid for pid, _, _, _ in passages}
    floor_lookup = {pid: (fid, fn) for pid, _, fid, fn in passages}

    prompt = (
        EXTRACT_SYSTEM
        + f"\n\nEntity: {entity_name} ({entity_type})\n\nPASSAGES:\n{numbered}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = response.text.strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        raw_entries = json.loads(text)
    except Exception as e:
        logger.warning("  Extraction call failed for %s: %s", entity_name, e)
        return []

    validated = []
    for entry in raw_entries:
        try:
            pid = entry["passage_id"]
            kind = entry["kind"]
            name = entry["name"]
            value = entry.get("value")
            reason = entry.get("reason", "")

            if pid not in valid_ids:
                logger.warning("  REJECTED (fabricated passage_id %s) for %s", pid, entity_name)
                continue
            if kind not in ("stat", "skill"):
                logger.warning("  REJECTED (bad kind %r) for %s", kind, entity_name)
                continue
            if kind == "stat" and name not in CORE_STATS:
                logger.warning("  REJECTED (stat name %r not in core 5) for %s", name, entity_name)
                continue
            if value is not None and not isinstance(value, int):
                logger.warning("  REJECTED (non-integer value %r) for %s", value, entity_name)
                continue
            if not name or not str(name).strip():
                logger.warning("  REJECTED (empty name) for %s", entity_name)
                continue

            floor_id, floor_number = floor_lookup[pid]
            validated.append({
                "passage_id": pid, "kind": kind, "name": name, "value": value,
                "reason": reason, "floor_id": floor_id, "floor_number": floor_number,
            })
        except (KeyError, TypeError) as e:
            logger.warning("  REJECTED (malformed entry %r: %s) for %s", entry, e, entity_name)
            continue

    return validated


def run_stats_extraction(conn, gemini_api_key, entity_ids=None, batch_size=10):
    """Run structured stat/skill extraction.

    If entity_ids is given, process exactly those entities (pilot/targeted mode).
    Otherwise, pick the next `batch_size` unprocessed entities (stats_extracted_at IS NULL)
    that have at least one ability-typed passage, ordered by id -- same continuation
    pattern as imager.py's batch mode, so this is safely re-runnable in chunks.
    """
    run_migrate(conn)
    client = genai.Client(api_key=gemini_api_key)

    cur = conn.cursor()
    if entity_ids:
        cur.execute("SELECT id, name, entity_type::text FROM entities WHERE id = ANY(%s) ORDER BY id", (entity_ids,))
    else:
        cur.execute("""
            SELECT DISTINCT e.id, e.name, e.entity_type::text
            FROM entities e
            JOIN passages p ON p.entity_id = e.id AND p.passage_type = 'ability'
            WHERE e.stats_extracted_at IS NULL
            ORDER BY e.id
            LIMIT %s
        """, (batch_size,))
    targets = cur.fetchall()
    cur.close()

    logger.info("Running structured stat/skill extraction for %d entities...", len(targets))
    total_stats = 0
    total_skills = 0
    total_rejected_passages_all = 0

    for entity_id, name, entity_type in targets:
        passages = _fetch_ability_passages(conn, entity_id)
        if not passages:
            continue

        logger.info("  %s (%s): %d ability passages", name, entity_type, len(passages))
        entries = _extract_for_entity(client, name, entity_type, passages)
        time.sleep(RATE_LIMIT_SECONDS)

        cur = conn.cursor()
        n_stats = n_skills = 0
        for e in entries:
            if e["kind"] == "stat":
                cur.execute("""
                    INSERT INTO entity_stats (entity_id, floor_id, stat_name, value, reason, source_passage_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (entity_id, e["floor_id"], e["name"], e["value"], e["reason"], e["passage_id"]))
                n_stats += 1
            else:
                cur.execute("""
                    INSERT INTO entity_skills (entity_id, floor_id, skill_name, level, origin, reason, source_passage_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (entity_id, e["floor_id"], e["name"], e["value"], None, e["reason"], e["passage_id"]))
                n_skills += 1
        cur.execute("UPDATE entities SET stats_extracted_at = NOW() WHERE id = %s", (entity_id,))
        conn.commit()
        cur.close()

        logger.info("    -> %d stat entries, %d skill entries", n_stats, n_skills)
        total_stats += n_stats
        total_skills += n_skills

    logger.info("Done. %d entities processed, %d stat rows, %d skill rows.",
                len(targets), total_stats, total_skills)
    return total_stats, total_skills
