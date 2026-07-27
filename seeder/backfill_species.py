"""
DCC Codex — species backfill.

`species` is now its own table (id, name, slug, description), separate from
entity_type, because species is orthogonal to crawler/npc/mob: a Kua-Tin can
be a crawler, an npc, or a dungeon-born mob. It describes what an individual
IS, not what narrative role it plays, so it doesn't belong in the same field
as entity_type.

This is a one-time backfill pass, same "never trust unvalidated LLM output"
philosophy as seeder/floors.py, seeder/permanence.py, and
seeder/reclassify_types.py: every answer is checked against the known species
list before being applied, results persist incrementally, and any entity
whose species isn't actually stated in the given text is left NULL and logged
rather than guessed at.
"""

import json
import logging
import re
import time

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

BATCH_SIZE = 20
RATE_LIMIT_SECONDS = 2.0

TARGET_ENTITY_TYPES = ("crawler", "npc", "mob")

BACKFILL_SYSTEM_PROMPT_TEMPLATE = (
    "You are identifying the species/race of characters and creatures from the LitRPG novel "
    "series Dungeon Crawler Carl. Here is the full list of known species in this setting:\n\n"
    "{species_list}\n\n"
    "You will be given a batch of entities, each with its current type (crawler/npc/mob), name, "
    "aliases, and a short persona/summary written from the book text.\n\n"
    "For each entity, if the text clearly states or strongly implies which species from the list "
    "above it belongs to, report that exact species name (matching the list exactly, case "
    "sensitive as shown). If the species is not stated or implied in the given text -- including "
    "the common case of an unremarked-upon human -- OMIT that entity from your response entirely. "
    "Do not guess, and do not default to 'Humans' just because no other species is mentioned; "
    "only report Humans if the text actually signals the entity is human (e.g. explicitly says so, "
    "or is a known real-world human character from Earth).\n\n"
    'Respond with ONLY a JSON array, no other text. Each element: {{"id": int, "species": "..."}}'
)


def _extract_json_array(text):
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def _call_batch(entities_batch, species_list_str, client):
    """entities_batch: list of (id, name, old_type, aliases, persona_text)."""
    lines = []
    for eid, name, etype, aliases, persona in entities_batch:
        alias_str = ", ".join(aliases) if aliases else "(none)"
        persona_trunc = (persona or "")[:600]
        lines.append(
            f"id={eid} | type={etype} | name={name!r} | aliases=[{alias_str}]\n"
            f"persona: {persona_trunc}"
        )
    system_prompt = BACKFILL_SYSTEM_PROMPT_TEMPLATE.format(species_list=species_list_str)
    prompt = system_prompt + "\n\n" + "\n\n".join(lines)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return _extract_json_array(response.text)


def _update_species_id(conn, entity_id, species_id):
    cur = conn.cursor()
    cur.execute(
        "UPDATE entities SET species_id = %s, updated_at = NOW() WHERE id = %s",
        (species_id, entity_id),
    )
    conn.commit()
    cur.close()


def run_backfill_species(conn, gemini_api_key: str, batch_size: int = BATCH_SIZE) -> dict:
    """One-time backfill pass. Returns a dict summary: count of entities matched,
    count skipped (no species stated in text), and total considered."""
    client = genai.Client(api_key=gemini_api_key)

    cur = conn.cursor()
    cur.execute("SELECT id, name FROM species ORDER BY name")
    species_rows = cur.fetchall()
    cur.close()
    name_to_id = {name.lower(): sid for sid, name in species_rows}
    species_list_str = "\n".join(f"- {name}" for _, name in species_rows)
    logger.info("Loaded %d known species.", len(species_rows))

    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, name, entity_type::text, aliases, persona_text
        FROM entities
        WHERE entity_type::text IN %s AND species_id IS NULL
        ORDER BY entity_type::text, id
        """,
        (TARGET_ENTITY_TYPES,),
    )
    rows = cur.fetchall()
    cur.close()
    logger.info("Loaded %d entities to backfill species for (types: %s).", len(rows), TARGET_ENTITY_TYPES)

    matched = 0
    skipped_ids = []
    all_ids = {r[0] for r in rows}

    i = 0
    while i < len(rows):
        batch = rows[i:i + batch_size]
        try:
            results = _call_batch(batch, species_list_str, client)
        except Exception as e:
            logger.warning("Backfill batch at index %d failed: %s", i, e)
            skipped_ids.extend(r[0] for r in batch)
            i += batch_size
            time.sleep(RATE_LIMIT_SECONDS)
            continue

        batch_ids = {r[0] for r in batch}
        answered_ids = set()
        for item in results:
            eid = item.get("id")
            species_name = item.get("species")
            if eid not in batch_ids:
                logger.warning("Backfill: model returned id=%s not in this batch, ignoring.", eid)
                continue
            sid = name_to_id.get((species_name or "").lower())
            if sid is None:
                logger.warning("Backfill: entity id=%s got unknown species %r, skipping.", eid, species_name)
                skipped_ids.append(eid)
                answered_ids.add(eid)
                continue
            _update_species_id(conn, eid, sid)
            matched += 1
            answered_ids.add(eid)
            logger.info("id=%s -> species %r (species_id=%s)", eid, species_name, sid)

        missed = batch_ids - answered_ids
        if missed:
            logger.info("Backfill: %d id(s) in this batch had no stated species, leaving NULL: %s",
                        len(missed), sorted(missed))
            skipped_ids.extend(missed)

        i += batch_size
        time.sleep(RATE_LIMIT_SECONDS)

    logger.info("Species backfill complete. Matched: %d. No species stated: %d. Total: %d.",
                matched, len(skipped_ids), len(all_ids))
    return {"matched": matched, "skipped_ids": skipped_ids, "total": len(all_ids)}
