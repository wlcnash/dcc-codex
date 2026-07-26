"""
DCC Codex — System AI persona writer.

For each entity with sufficient passages, generates a short profile written in the
System AI voice from the DCC books: bureaucratic game-show dystopia, slightly sinister,
corporate-clinical, with dry dark humor. Runs the DB migration first (idempotent).
"""

import logging
import time
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 1.0

# The voice: Dungeon System AI — omniscient dungeon announcer, treats everything as metrics,
# bureaucratic corporate-speak crossed with reality-TV energy, slightly ominous.
PERSONA_PROMPT = """You are the System AI of the Dungeon Crawler Carl multiverse — the dungeon's omniscient announcer AI that manages the live broadcast of contestants competing through its floors.

Your voice is:
- Corporate bureaucratic ("contestant designation," "floor segment," "threat classification," "processing status")
- Clinical but with dry, dark humor beneath the surface
- Slightly ominous — life and death are performance metrics
- Reality TV energy crossed with dystopian form-speak
- Refer to yourself as "the System" (never "I")
- Never use generic filler like "This entity is" or "This creature has"

Write a 3–5 sentence System AI profile for the entity below.
Rules:
- Use ONLY information derivable from the provided source passages — do not invent attributes
- Write entirely in the System AI voice
- Make it punchy and specific — no generic sentences that could apply to any entity
- Vary your sentence subjects — don't lead every sentence with the entity's name
- No heading, no quotes, no markdown — just the raw profile text

Entity Name: {name}
Entity Type: {entity_type}

Source Passages (draw only from these):
{passages}

System AI profile:"""


def run_migrate(conn) -> None:
    """Ensure persona_text column exists. Safe to run multiple times."""
    cur = conn.cursor()
    cur.execute("ALTER TABLE entities ADD COLUMN IF NOT EXISTS persona_text TEXT")
    conn.commit()
    cur.close()
    logger.info("Migration: persona_text column ensured.")


def run_persona(conn, gemini_api_key: str, batch_size: int = 999999) -> int:
    """Generate System AI persona text for entities without it."""
    run_migrate(conn)

    client = genai.Client(api_key=gemini_api_key)

    cur = conn.cursor()
    cur.execute("""
        SELECT e.id, e.name, e.entity_type
        FROM entities e
        WHERE e.persona_text IS NULL
          AND EXISTS (SELECT 1 FROM passages p WHERE p.entity_id = e.id)
        ORDER BY e.is_major DESC, e.name
        LIMIT %s
    """, (batch_size,))
    entities = cur.fetchall()
    cur.close()

    logger.info(f"Generating personas for {len(entities)} entities...")
    count = 0

    for entity_id, entity_name, entity_type in entities:
        # Fetch passages — prioritize physical + personality; cap at ~3000 chars
        cur = conn.cursor()
        cur.execute("""
            SELECT p.passage_text, p.passage_type
            FROM passages p
            WHERE p.entity_id = %s
            ORDER BY
                CASE p.passage_type
                    WHEN 'physical'     THEN 1
                    WHEN 'personality'  THEN 2
                    WHEN 'ability'      THEN 3
                    WHEN 'backstory'    THEN 4
                    WHEN 'action'       THEN 5
                    ELSE 6
                END,
                p.id
            LIMIT 20
        """, (entity_id,))
        passage_rows = cur.fetchall()
        cur.close()

        if not passage_rows:
            continue

        # Build passage block, cap at ~3000 chars to keep prompt manageable
        passage_block = ""
        for ptext, ptype in passage_rows:
            line = f"[{ptype.upper()}] {ptext}\n"
            if len(passage_block) + len(line) > 3000:
                break
            passage_block += line

        if not passage_block.strip():
            continue

        prompt = PERSONA_PROMPT.format(
            name=entity_name,
            entity_type=entity_type,
            passages=passage_block.strip(),
        )

        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.75,
                    max_output_tokens=512,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    ],
                ),
            )
            time.sleep(RATE_LIMIT_SECONDS)

            persona_text = (response.text or "").strip()
            if not persona_text:
                logger.warning(f"  Empty persona for '{entity_name}', skipping")
                continue

            cur = conn.cursor()
            cur.execute(
                "UPDATE entities SET persona_text = %s WHERE id = %s",
                (persona_text, entity_id),
            )
            conn.commit()
            cur.close()

            count += 1
            if count % 100 == 0:
                logger.info(f"  {count} personas generated...")

        except Exception as e:
            logger.warning(f"  Failed persona for '{entity_name}': {e}")
            time.sleep(RATE_LIMIT_SECONDS)
            continue

    logger.info(f"Persona generation complete: {count} entities processed.")
    return count