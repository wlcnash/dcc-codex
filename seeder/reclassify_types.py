"""
DCC Codex — entity_type taxonomy migration.

Old taxonomy (8 types) was too coarse: `character` blended crawlers and NPCs,
`creature` blended monsters and species/races, `faction` blended organizations
and species/races, and `other` was a dumping ground for deities, in-universe
media, AI/system characters, and curses.

New taxonomy (11 types): crawler, npc, mob, species, item, ability, location,
floor, faction, deity, media. item/ability/location/floor are unchanged and
NOT touched by this script. character/creature/faction/other rows get
reclassified by an LLM pass grounded on the entity's name + persona_text +
aliases.

This is a one-time reclassification pass, same "never trust unvalidated LLM
output" philosophy as seeder/floors.py and seeder/permanence.py: every answer
is checked against the allowed type set before being applied, results persist
incrementally (per batch commit), and anything the model can't confidently
place is left alone and logged for manual follow-up rather than guessed at.
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

OLD_TYPES_TO_RECLASSIFY = ("character", "creature", "faction", "other")

ALLOWED_NEW_TYPES = {
    "crawler", "npc", "mob", "species", "faction", "deity", "media",
    "item", "ability", "location", "floor",
}

TYPE_DEFINITIONS = """
- crawler: an individual competitor entered into the dungeon crawl reality show. Includes
  humans and any alien species members who are themselves participating as a crawler
  (e.g. Carl, Donut, named alien crawlers). A crawler is always a single named individual.
- npc: a non-crawler character who is not a hostile monster -- dungeon-created NPCs, guild
  staff, sponsors, show hosts/announcers/producers, AI overseers (e.g. Mordecai, Odette,
  "the AI", "system AI"). A single named individual, not a group.
- mob: a hostile or neutral creature/monster that crawlers fight or encounter in the dungeon
  (goblins, ogres, generic monster types, dungeon-born beasts). NOT a named crawler or NPC,
  and NOT a species/race classification -- an individual monster or monster type encountered
  in combat.
- species: a race or species classification (e.g. Kua-Tin, gnoll, high elf, naga,
  dromedarian) -- the taxonomic category itself, describing what KIND of being something is,
  not an individual named member of it. If the entity is a named individual, it is NOT a
  species even if the species is mentioned in its description -- classify the individual as
  crawler/npc/mob instead.
- faction: an organized group -- corporation, guild, military unit, political power, or
  crawler team/party (e.g. Borant, the Blood Sultanate, a named battalion). NOT a species or
  race, even if the group is composed of one species.
- deity: a sponsoring god/Patron that grants power to crawlers in the Patron system (e.g.
  Donar, Taranis, The Dagda, Wakinyan, Zentix, T'Ghee). These are named after or modeled on
  real-world mythological deities.
- media: an in-universe broadcast segment, TV show, publicity event, or programming block
  within the galaxy-wide reality show that the dungeon crawl is broadcast as (e.g. Crawl Con,
  The Recital, Escape Velocity, Dungeon Crawler After Hours with Odette).
- item: a weapon, armor, or other object. (Existing type, only choose this if the entity was
  clearly misfiled and is actually an object, not a character/place/group.)
- ability: a skill, spell, feat, curse, or magical effect. (Existing type, only choose this if
  the entity was clearly misfiled.)
- location: a place. (Existing type, only choose this if the entity was clearly misfiled.)
- floor: a dungeon floor. (Existing type, only choose this if the entity was clearly misfiled.)
"""

RECLASSIFY_SYSTEM_PROMPT = (
    "You are reclassifying entities from the LitRPG novel series Dungeon Crawler Carl into a "
    "new, more precise taxonomy. You will be given a batch of entities, each with its current "
    "(too-coarse) type, name, aliases, and a short persona/summary written from the book text.\n\n"
    "The new type definitions are:\n" + TYPE_DEFINITIONS + "\n"
    "For each entity, choose exactly ONE new type from this exact list: "
    + ", ".join(sorted(ALLOWED_NEW_TYPES)) + ".\n\n"
    "Rules:\n"
    "- Base your answer only on the name/aliases/persona text given. Do not guess beyond what's shown.\n"
    "- An individual named being is crawler, npc, or mob -- never species.\n"
    "- A race/species classification (not a specific named individual) is species.\n"
    "- An organized group of beings (corporate, military, political, guild, team) is faction, "
    "even if every member is the same species.\n"
    "- If you are genuinely unsure and none of the definitions clearly fit, omit that entity "
    "from your response entirely rather than guessing -- an omitted entity will be reviewed "
    "manually instead of silently misclassified.\n\n"
    'Respond with ONLY a JSON array, no other text. Each element: {"id": int, "new_type": "..."}'
)


def _extract_json_array(text):
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def _call_batch(entities_batch, client):
    """entities_batch: list of (id, name, old_type, aliases, persona_text)."""
    lines = []
    for eid, name, old_type, aliases, persona in entities_batch:
        alias_str = ", ".join(aliases) if aliases else "(none)"
        persona_trunc = (persona or "")[:600]
        lines.append(
            f"id={eid} | current_type={old_type} | name={name!r} | aliases=[{alias_str}]\n"
            f"persona: {persona_trunc}"
        )
    prompt = RECLASSIFY_SYSTEM_PROMPT + "\n\n" + "\n\n".join(lines)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return _extract_json_array(response.text)


def _update_entity_type(conn, entity_id, new_type):
    cur = conn.cursor()
    cur.execute(
        "UPDATE entities SET entity_type = %s, updated_at = NOW() WHERE id = %s",
        (new_type, entity_id),
    )
    conn.commit()
    cur.close()


def run_reclassify_types(conn, gemini_api_key: str, batch_size: int = BATCH_SIZE) -> dict:
    """One-time reclassification pass. Returns a dict summary: counts per new_type,
    plus a list of entity ids that were skipped (either the model omitted them, or it
    returned an invalid type)."""
    client = genai.Client(api_key=gemini_api_key)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, name, entity_type::text, aliases, persona_text
        FROM entities
        WHERE entity_type::text IN %s
        ORDER BY entity_type::text, id
        """,
        (OLD_TYPES_TO_RECLASSIFY,),
    )
    rows = cur.fetchall()
    cur.close()
    logger.info("Loaded %d entities to reclassify (types: %s).", len(rows), OLD_TYPES_TO_RECLASSIFY)

    counts = {}
    skipped_ids = []
    all_ids = {r[0] for r in rows}

    i = 0
    while i < len(rows):
        batch = rows[i:i + batch_size]
        try:
            results = _call_batch(batch, client)
        except Exception as e:
            logger.warning("Reclassify batch at index %d failed: %s", i, e)
            skipped_ids.extend(r[0] for r in batch)
            i += batch_size
            time.sleep(RATE_LIMIT_SECONDS)
            continue

        batch_ids = {r[0] for r in batch}
        answered_ids = set()
        for item in results:
            eid = item.get("id")
            new_type = item.get("new_type")
            if eid not in batch_ids:
                logger.warning("Reclassify: model returned id=%s not in this batch, ignoring.", eid)
                continue
            if new_type not in ALLOWED_NEW_TYPES:
                logger.warning("Reclassify: entity id=%s got invalid type %r, skipping.", eid, new_type)
                skipped_ids.append(eid)
                answered_ids.add(eid)
                continue
            _update_entity_type(conn, eid, new_type)
            counts[new_type] = counts.get(new_type, 0) + 1
            answered_ids.add(eid)
            logger.info("id=%s -> %s", eid, new_type)

        missed = batch_ids - answered_ids
        if missed:
            logger.warning("Reclassify: model omitted %d id(s) from this batch: %s", len(missed), sorted(missed))
            skipped_ids.extend(missed)

        i += batch_size
        time.sleep(RATE_LIMIT_SECONDS)

    logger.info("Reclassification complete. Counts: %s. Skipped: %d of %d.",
                counts, len(skipped_ids), len(all_ids))
    return {"counts": counts, "skipped_ids": skipped_ids, "total": len(all_ids)}
