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

from persona import _is_valid_persona, MAX_OUTPUT_TOKENS

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


# 2026-07-29: species.description regeneration.
#
# species.description was built as a ONE-TIME snapshot during the 2026-07-27
# species-extraction migration (persona_text + concatenated distinct passage text,
# written directly via inline SQL, never routed through a reusable generation
# function). It predates both the 2026-07-28 persona-truncation fix and the
# 2026-07-29 persona-leak-pattern hardening in persona.py, so it inherited --
# and never had a chance to be fixed for -- the exact same bug classes: silent
# MAX_TOKENS truncation and drafting-scratchpad leakage. A live audit (2026-07-29)
# running persona.py's own _is_valid_persona() plus a truncation-punctuation check
# against all 39 species rows found 38/39 flagged. Examples: id=33 "the naga" cut
# off mid-word ("...designated as the n"); id=1 "Kua-Tin" had a leaked
# "...fact):*" outline fragment identical in shape to the entities.persona_text
# leak bug fixed earlier today; id=19 "bugbears" was almost entirely drafting-
# scratchpad text ("4.  **Drafting Sentences (iterative process):**").
#
# Since the original entities/passages that fed the 2026-07-27 migration are gone,
# the only surviving grounding text is each species' own "Source passages:" tail --
# still present verbatim in every corrupted row, since the bug only ever corrupted
# the generated summary written BEFORE that marker. This function re-splits each
# description on that marker, regenerates only the summary half using the same
# hardened generation pattern as persona.py (4096-token budget, BLOCK_NONE safety,
# explicit finish_reason MAX_TOKENS rejection before ever looking at the text, and
# persona.py's own _is_valid_persona() leak check reused as-is since it's the same
# voice/rule family), and reassembles "{new summary}\n\nSource passages: {unchanged
# source text}" so the source-passage attachment convention is preserved.

SOURCE_PASSAGES_MARKER_RE = re.compile(r"source passages\s*:\s*", re.IGNORECASE)

SPECIES_DESC_PROMPT = """You are the System AI of the Dungeon Crawler Carl multiverse — the dungeon's omniscient announcer AI that manages the live broadcast of crawlers competing through its floors.

Your voice is:
- Corporate bureaucratic ("crawler designation," "floor segment," "processing status")
- Clinical but with dry, dark humor beneath the surface
- Slightly ominous — life and death are performance metrics
- Reality TV energy crossed with dystopian form-speak
- Refer to yourself as "the System" (never "I")
- Never use generic filler like "This species is" or "This creature type has"

Write a 3-5 sentence System AI profile describing the SPECIES/RACE below as a whole
(not any single individual) -- general traits, biology, temperament, and dungeon
role shared across the species.

Rules:
- Use ONLY information derivable from the provided source passages -- do not invent attributes
- Write entirely in the System AI voice
- Make it punchy and specific -- no generic sentences that could apply to any species
- Do NOT open with a rigid "[Label]: [description]" declaration (e.g. "Species
  classification:", "Threat classification:", "Designation:") -- that exact templated
  shape is itself an AI-generated-text tell regardless of which words fill it in.
  Open with an actual sentence instead.
- Never use the word "contestant" for a dungeon participant -- the book's own term is
  "crawler". Use "crawler" instead.
- No heading, no quotes, no markdown -- just the raw profile text
- Describe general, durable species-wide traits only, not one individual's specific
  story or a single passing combat moment

Species Name: {name}

Source Passages (draw only from these):
{passages}

System AI profile:"""


def run_regen_species_descriptions(conn, gemini_api_key: str, species_ids=None) -> dict:
    """Regenerate the AI-written summary half of species.description, keeping the
    verbatim "Source passages:" tail unchanged. See module-level comment above for
    why this exists and why it's grounded on the embedded source-passage text rather
    than re-querying entities/passages (the originals no longer exist)."""
    client = genai.Client(api_key=gemini_api_key)

    cur = conn.cursor()
    if species_ids:
        cur.execute("SELECT id, name, description FROM species WHERE id = ANY(%s) ORDER BY id", (species_ids,))
    else:
        cur.execute("SELECT id, name, description FROM species ORDER BY id")
    rows = cur.fetchall()
    cur.close()

    updated = 0
    no_source = 0
    rejected_invalid = 0
    rejected_truncated = 0

    for species_id, name, description in rows:
        if not description:
            no_source += 1
            continue
        m = SOURCE_PASSAGES_MARKER_RE.search(description)
        if not m:
            logger.warning(
                f"  species id={species_id} '{name}': no 'Source passages:' marker found, "
                f"skipping (nothing to ground regeneration on)."
            )
            no_source += 1
            continue
        source_text = description[m.end():].strip()
        if not source_text:
            no_source += 1
            continue

        prompt = SPECIES_DESC_PROMPT.format(name=name, passages=source_text[:3000])

        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.75,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    ],
                ),
            )
            time.sleep(RATE_LIMIT_SECONDS)

            # Same discipline as persona.py: reject a silently-truncated MAX_TOKENS
            # response before ever looking at the text.
            finish_reason = None
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
            if finish_reason is not None and str(finish_reason).endswith("MAX_TOKENS"):
                rejected_truncated += 1
                logger.warning(
                    f"  REJECTED truncated (MAX_TOKENS) species description for '{name}' "
                    f"(id={species_id})"
                )
                continue

            new_summary = (response.text or "").strip()
            if not new_summary or not _is_valid_persona(new_summary):
                rejected_invalid += 1
                logger.warning(
                    f"  REJECTED malformed/scratchpad species description for '{name}' "
                    f"(id={species_id}): {new_summary[:80]!r}"
                )
                continue

            # Belt-and-suspenders truncation check: even a finish_reason=STOP response
            # should end on a real sentence boundary; if it doesn't, treat it as
            # truncated rather than trusting it (same failure mode the original
            # migration snapshot suffered from).
            if not re.search(r'[.!?]["\')]?\s*$', new_summary):
                rejected_truncated += 1
                logger.warning(
                    f"  REJECTED species description for '{name}' (id={species_id}) -- "
                    f"doesn't end in sentence punctuation, likely truncated: "
                    f"{new_summary[-80:]!r}"
                )
                continue

            new_description = f"{new_summary}\n\nSource passages: {source_text}"

            cur = conn.cursor()
            cur.execute("UPDATE species SET description = %s WHERE id = %s", (new_description, species_id))
            conn.commit()
            cur.close()

            updated += 1
            logger.info(f"  id={species_id} '{name}' -> regenerated ({len(new_summary)} chars)")

        except Exception as e:
            logger.warning(f"  Failed species description regen for '{name}' (id={species_id}): {e}")
            time.sleep(RATE_LIMIT_SECONDS)
            continue

    logger.info(
        f"Species description regen complete: {updated} updated, {no_source} skipped "
        f"(no source text), {rejected_invalid} rejected as malformed, "
        f"{rejected_truncated} rejected as truncated."
    )
    return {
        "updated": updated,
        "no_source": no_source,
        "rejected_invalid": rejected_invalid,
        "rejected_truncated": rejected_truncated,
        "total": len(rows),
    }
