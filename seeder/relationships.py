"""
DCC Codex — entity relationship extraction.

`entity_relationships` (entity_a_id, entity_b_id, relation_type, evidence, chapter_id)
has existed in the schema since early on but was never populated. This is the first
pipeline step that fills it in.

Grounded in real text before building anything (2026-07-30): confirmed e.g. "Bopca
Protectors are magical, gnome-like creatures who exist solely to watch over Safe
Rooms," with individually-named ones (Tally, Sebastian, Gordo, Wendita, Qwist)
already tagged as npc with "Bopca Protector" in their aliases -- a clean, real
guard/inhabits-location relationship, not a guessed genre-convention category.

Design decisions:

1. **Only scan passages belonging to "agent" entity types** (crawler, npc, mob,
   faction, deity) as the entity_a / subject side. Locations, items, floors,
   abilities, and media can be the entity_b / object of a relationship (a crawler
   guards a location, worships a deity, owns a pet-item) but are never scanned as
   the subject themselves. This is a narrative-realism constraint (the story never
   narrates "the Safe Room's relationship to Tally"), and it has a load-bearing
   side effect: it prevents the same real-world fact from ever being extracted
   twice from both directions (once as "Tally guards Safe Rooms" while scanning
   Tally, again as "Safe Rooms guarded_by Tally" while scanning Safe Rooms) which
   would otherwise slip past the (entity_a, entity_b, relation_type) unique
   constraint as two distinct rows and double up on the target's page.

2. **relation_type is always stored in the FORWARD direction** (entity_a's relation
   TO entity_b), never pre-flipped for storage. Reverse-direction display labels
   (e.g. "guards" -> "guarded by" when viewed from entity_b's page) are a pure
   display-layer concern -- see RELATION_REVERSE_LABELS in app/main.py, which is a
   deliberately duplicated small dict (not a shared import) since app/ and seeder/
   are independently built/deployed images that don't import from each other today.
   Keep the two lists in sync by hand if the vocabulary changes.

3. **relation_type is free text, not a rigid enum**, unlike boss_tier. Real
   relationships are far more varied and compositional than boss_tier's clean
   6-rung ladder was -- forcing a fixed enum would either be too narrow (omitting
   real relationships) or balloon into an unmaintainable list. Instead: the prompt
   is biased toward a curated common-verb vocabulary (kept in sync with
   RELATION_REVERSE_LABELS on the display side), normalized to lowercase
   snake_case, and validated for evidence groundedness rather than restricted to
   an allowlist. Anything outside the curated set still gets stored and still
   displays correctly -- it just falls back to a symmetric label (same word shown
   on both entities' pages) until someone adds it to the curated reverse-label map,
   which is a strict improvement over today's un-implemented state either way.

4. **Every relationship must cite a verbatim, verifiable quote.** Same "never trust
   unvalidated LLM output" discipline as floors.py/reclassify_types.py/
   backfill_species.py: evidence_quote must be an exact substring of the specific
   passage (by id) the model claims to be citing, and that passage must actually
   belong to the entity being scanned. The other party (other_name) must resolve
   to an existing entities.name or alias by exact case-insensitive match -- no
   fuzzy matching, same conservative-linking philosophy as main.py's
   linkify_text(). Ambiguous or unresolvable names are dropped, not guessed.
   See audit_relationship_evidence.py for the standing rerunnable check.
"""

import json
import logging
import re
import time

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 1.5

# Entities of these types are scanned as the subject/entity_a side. See design
# note #1 above for why this list intentionally excludes location/item/floor/
# ability/media/other -- those types are only ever the object (entity_b) of a
# relationship in this pipeline, never the subject.
SUBJECT_ENTITY_TYPES = ("crawler", "npc", "mob", "faction", "deity")

# Passage budget per entity, by total character count rather than passage count.
# Deliberately larger than other generation steps' caps (e.g. species backfill's
# source_text[:3000], item_descriptions' ~900-char cap) because relationships are
# sparse events scattered across a large volume of narrative for well-covered
# entities (Carl alone has 200+ passages) -- a small cap would systematically miss
# real relationships for exactly the entities most likely to have them.
MAX_PASSAGE_CHARS = 20000

# Curated vocabulary bias for the prompt. Not an enforced allowlist -- the model
# may use a different short snake_case verb if none of these fit, per design note
# #3. Kept in sync (by hand) with RELATION_REVERSE_LABELS in app/main.py.
SUGGESTED_RELATION_TYPES = (
    "guards", "member_of", "leads", "mentor_of", "owns_pet", "parent_of",
    "sibling_of", "spouse_of", "ally_of", "rival_of", "enemy_of", "friend_of",
    "serves", "worships", "works_at", "created",
)

RELATIONSHIP_SYSTEM_PROMPT = (
    "You are extracting explicit relationships between named characters/groups from the "
    "LitRPG novel series Dungeon Crawler Carl.\n\n"
    "You will be given one entity (the SUBJECT) and a numbered list of its own source "
    "passages from the book. Find every OTHER named entity that these passages show has an "
    "explicit relationship to the SUBJECT -- family, romantic, professional/employment, "
    "membership in a group, guarding/inhabiting a place, mentorship, pet ownership, "
    "worship of a deity, leadership, alliance, rivalry, or friendship.\n\n"
    "Common relation_type verbs (use one of these if it fits, otherwise invent a similarly "
    "short snake_case verb phrase describing the SUBJECT's relation TO the other entity):\n"
    + ", ".join(SUGGESTED_RELATION_TYPES) + "\n\n"
    "Rules:\n"
    "- Only report a relationship if it is EXPLICITLY stated or unambiguously shown in the "
    "given passages -- do not infer from genre convention or guess.\n"
    "- The other entity's name must appear in the passages substantively enough to identify "
    "who/what it is -- a passing pronoun reference with no name is not enough.\n"
    "- relation_type must describe the SUBJECT's relation TO the other entity (forward "
    "direction only), as a short lowercase snake_case verb phrase (2-3 words max).\n"
    "- evidence_quote must be copied EXACTLY, character-for-character, from the passage you "
    "cite -- no paraphrasing, no combining text from two passages into one quote.\n"
    "- evidence_passage_id must be the exact numeric id shown before the passage you're "
    "citing.\n"
    "- If you are not confident, omit that relationship entirely rather than guessing.\n\n"
    'Respond with ONLY a JSON array, no other text. Each element: '
    '{"other_name": "...", "relation_type": "...", "evidence_passage_id": int, "evidence_quote": "..."}'
)


def _extract_json_array(text):
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def _normalize_relation_type(raw):
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip().lower().replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or len(s) > 60:
        return None
    return s


def _build_name_index(conn):
    """Map every lowercased name/alias to the set of entity ids that claim it.
    A name claimed by more than one entity is ambiguous and must never be
    auto-resolved -- see design note #4."""
    cur = conn.cursor()
    cur.execute("SELECT id, name, aliases FROM entities")
    rows = cur.fetchall()
    cur.close()

    index = {}
    for eid, name, aliases in rows:
        for candidate in [name] + (aliases or []):
            key = candidate.strip().lower()
            if not key:
                continue
            index.setdefault(key, set()).add(eid)
    return index


def _resolve_entity(name, name_index):
    if not name or not isinstance(name, str):
        return None
    ids = name_index.get(name.strip().lower())
    if not ids or len(ids) != 1:
        return None
    return next(iter(ids))


def _fetch_subject_entities(conn, entity_types, only_major, limit):
    cur = conn.cursor()
    query = """
        SELECT id, name, entity_type::text
        FROM entities
        WHERE entity_type::text IN %s
        AND id IN (SELECT DISTINCT entity_id FROM passages)
    """
    params = [entity_types]
    if only_major:
        query += " AND is_major = TRUE"
    query += " ORDER BY id"
    if limit:
        query += " LIMIT %s"
        params.append(limit)
    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    cur.close()
    return rows


def _fetch_passages(conn, entity_id, max_chars):
    cur = conn.cursor()
    cur.execute(
        "SELECT id, passage_text FROM passages WHERE entity_id = %s ORDER BY id",
        (entity_id,),
    )
    rows = cur.fetchall()
    cur.close()

    selected = []
    total = 0
    for pid, text in rows:
        if not text:
            continue
        total += len(text)
        selected.append((pid, text))
        if total >= max_chars:
            break
    return selected, {pid: text for pid, text in selected}


def _call_extraction(subject_name, passages, client):
    lines = [f"[{pid}] {text}" for pid, text in passages]
    prompt = (
        RELATIONSHIP_SYSTEM_PROMPT
        + f"\n\nSUBJECT: {subject_name}\n\nPassages:\n"
        + "\n\n".join(lines)
    )
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return _extract_json_array(response.text)


def _insert_relationship(conn, entity_a_id, entity_b_id, relation_type, evidence, chapter_id):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO entity_relationships (entity_a_id, entity_b_id, relation_type, evidence, chapter_id)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (entity_a_id, entity_b_id, relation_type) DO NOTHING
        """,
        (entity_a_id, entity_b_id, relation_type, evidence, chapter_id),
    )
    conn.commit()
    inserted = cur.rowcount > 0
    cur.close()
    return inserted


def _chapter_for_passage(conn, passage_id):
    cur = conn.cursor()
    cur.execute("SELECT chapter_id FROM passages WHERE id = %s", (passage_id,))
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def run_relationships(
    conn,
    gemini_api_key: str,
    entity_types=SUBJECT_ENTITY_TYPES,
    only_major: bool = False,
    batch_size: int = None,
) -> dict:
    """Extract and persist entity relationships. See module docstring for the
    subject/object asymmetry, grounding, and dedup-by-construction rationale.

    Returns a summary dict: inserted count, and per-rejection-reason counts so a
    run can be sanity-checked without re-reading logs line by line."""
    client = genai.Client(api_key=gemini_api_key)
    name_index = _build_name_index(conn)
    logger.info("Built name/alias index: %d unique keys.", len(name_index))

    subjects = _fetch_subject_entities(conn, entity_types, only_major, batch_size)
    logger.info(
        "Loaded %d subject entities to scan (types: %s, only_major=%s).",
        len(subjects), entity_types, only_major,
    )

    inserted = 0
    no_passages = 0
    llm_errors = 0
    rejected_unresolved_name = 0
    rejected_self = 0
    rejected_bad_evidence = 0
    rejected_bad_relation_type = 0
    entities_with_hits = 0

    for entity_id, name, etype in subjects:
        passages, passage_by_id = _fetch_passages(conn, entity_id, MAX_PASSAGE_CHARS)
        if not passages:
            no_passages += 1
            continue

        try:
            results = _call_extraction(name, passages, client)
        except Exception as e:
            logger.warning("Relationship extraction failed for id=%s (%s): %s", entity_id, name, e)
            llm_errors += 1
            time.sleep(RATE_LIMIT_SECONDS)
            continue

        entity_had_hit = False
        for item in results:
            other_name = item.get("other_name")
            raw_relation_type = item.get("relation_type")
            evidence_passage_id = item.get("evidence_passage_id")
            evidence_quote = item.get("evidence_quote")

            other_id = _resolve_entity(other_name, name_index)
            if other_id is None:
                logger.info(
                    "  id=%s (%s): couldn't uniquely resolve other_name=%r, skipping.",
                    entity_id, name, other_name,
                )
                rejected_unresolved_name += 1
                continue

            if other_id == entity_id:
                rejected_self += 1
                continue

            relation_type = _normalize_relation_type(raw_relation_type)
            if relation_type is None:
                logger.warning(
                    "  id=%s (%s): bad relation_type %r, skipping.",
                    entity_id, name, raw_relation_type,
                )
                rejected_bad_relation_type += 1
                continue

            source_text = passage_by_id.get(evidence_passage_id)
            if (
                source_text is None
                or not isinstance(evidence_quote, str)
                or not evidence_quote.strip()
                or evidence_quote.strip() not in source_text
            ):
                logger.warning(
                    "  id=%s (%s): evidence_quote not found verbatim in passage %r, skipping. quote=%r",
                    entity_id, name, evidence_passage_id, (evidence_quote or "")[:120],
                )
                rejected_bad_evidence += 1
                continue

            chapter_id = _chapter_for_passage(conn, evidence_passage_id)
            was_inserted = _insert_relationship(
                conn, entity_id, other_id, relation_type, evidence_quote.strip(), chapter_id
            )
            if was_inserted:
                inserted += 1
                entity_had_hit = True
                logger.info(
                    "  id=%s (%s) --%s--> id=%s (%s)",
                    entity_id, name, relation_type, other_id, other_name,
                )

        if entity_had_hit:
            entities_with_hits += 1

        time.sleep(RATE_LIMIT_SECONDS)

    summary = {
        "subjects_scanned": len(subjects),
        "entities_with_hits": entities_with_hits,
        "inserted": inserted,
        "no_passages": no_passages,
        "llm_errors": llm_errors,
        "rejected_unresolved_name": rejected_unresolved_name,
        "rejected_self": rejected_self,
        "rejected_bad_evidence": rejected_bad_evidence,
        "rejected_bad_relation_type": rejected_bad_relation_type,
    }
    logger.info("Relationship extraction complete: %s", summary)
    return summary
