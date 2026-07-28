"""
seeder/entity_resolution.py

Extraction-time entity resolution -- goes beyond exact name/alias string
matching to catch duplicates that arise when the extraction LLM refers to the
same in-story entity under different wording across chapters (e.g. "Katia" in
one chapter, "Katia Grimmsdottir" in another; "AI", "System AI", "dungeon AI"
all being the same dungeon-overseer NPC).

WHY THIS EXISTS -- READ BEFORE TOUCHING:
extractor.py's original upsert_entity() only ever matched on EXACT string
equality (`name = %s OR %s = ANY(aliases)`). Every duplicate-entity bug found
during the 2026-07-27/28 DCC Codex sessions (the 64-group casing/article
dedup, the 4-way AI cluster, ~17 inconsistently-named floor entities, the
Katia cluster, the Donut duplicate) traces back to that single gap: nothing
between extraction and insertion ever asked "does this already exist under a
different name?"

pg_trgm trigram string similarity ALONE is not a reliable fix for this.
Calibration run against real, confirmed duplicate/non-duplicate pairs from
this project showed no threshold cleanly separates the two classes:
  - Known TRUE duplicates scored as LOW as 0.233-0.316 trigram similarity
    (e.g. "Katia" vs "Katia Grimmsdottir").
  - Known genuinely DIFFERENT entities scored as HIGH as 0.454
    (short/generic names collide on trigrams incidentally).
A fixed similarity cutoff would either merge unrelated entities or miss real
duplicates -- often both at different thresholds. See
feedback_never_guess_read_source.md and ENTITY_RESOLUTION_POLICY.md (repo
root of this concern) for the full rationale.

DESIGN: two-stage resolution, mirroring the blocking-then-matching pattern
used in production entity-resolution / GraphRAG systems:

  STAGE 1 -- find_candidates() (blocking / recall filter):
    A deliberately LOOSE pg_trgm similarity search against existing entities
    of the SAME entity_type. This never decides same/different by itself --
    it only bounds how many expensive content-judgment calls stage 2 makes.
    False candidates just cost one wasted LLM call; missed candidates are
    permanent silent duplicates, so the filter is tuned toward recall.

  STAGE 2 -- judge_same_entity() (content judgment):
    For each candidate, an LLM call is given BOTH entities' actual
    descriptive content (persona_text, or raw passage text if no persona
    exists yet) -- never just the names -- and asked for a grounded
    same/different/uncertain verdict with a one-to-two-sentence citation of
    what drove the answer. This follows the same "never trust unvalidated
    LLM output" rule used throughout this codebase (floors.py, permanence.py,
    reclassify_types.py): the verdict is checked against an allowed set, any
    malformed/failed/invalid response is forced to "uncertain" (never
    silently "different", which would let a real duplicate through; never
    silently "same", which would risk an incorrect auto-merge), and
    "uncertain" is a first-class, expected outcome -- not an error.

RESOLUTION OUTCOMES for a new entity mention, enforced by resolve_entity():
  - verdict "same" (confident)      -> merge into the existing candidate.
                                        No new row created; new name/aliases
                                        are folded onto the candidate, exactly
                                        like the pre-existing exact-match path
                                        in upsert_entity().
  - verdict "uncertain"             -> create the entity normally (data is
                                        NEVER silently dropped or force-merged
                                        on a guess) AND write a row to
                                        entity_resolution_log with
                                        reviewed=false so a human can resolve
                                        it later. Never auto-merged, never
                                        silently ignored.
  - verdict "different" / no        -> create the entity normally, no review
    candidates found                   needed.

EVERY resolution attempt (same, different, or uncertain) is logged to
entity_resolution_log for full audit -- this is what makes every merge or
non-merge decision independently verifiable after the fact, per
feedback_never_guess_read_source.md's rule to always show the receipt, not
just the conclusion.

This module is imported by extractor.py's upsert_entity(). Per
ENTITY_RESOLUTION_POLICY.md, no future extraction code path may create an
entity row without going through resolve_entity() first -- there is no
"skip resolution for speed/cost" escape hatch. If cost needs to be reduced,
reduce it here (fewer candidates, cheaper grounding), not by bypassing the
module.
"""

import json
import logging
import re

from google.genai import types

logger = logging.getLogger(__name__)

# Deliberately loose -- see module docstring. This is a recall filter, not a
# same/different decision. Calibration showed true duplicates scoring as low
# as 0.233, so 0.15 is set well below that with margin, accepting some noise
# candidates in exchange for not missing real ones.
CANDIDATE_SIMILARITY_THRESHOLD = 0.15

# Bounds LLM calls per new entity mention. Cost gate, not an accuracy knob --
# if this is ever raised, no other value in this module needs to change.
MAX_CANDIDATES_CHECKED = 3

MAX_PASSAGES_FOR_GROUNDING = 5
GROUNDING_CHAR_CAP = 1500

VALID_VERDICTS = {"same", "different", "uncertain"}

JUDGE_SYSTEM_PROMPT = (
    "You are resolving whether two entity records from the LitRPG novel series "
    "Dungeon Crawler Carl refer to the SAME in-story entity or to two DIFFERENT "
    "entities that merely have similar names.\n\n"
    "You will be given each entity's name, aliases, entity_type, and grounding "
    "content (either a persona summary or verbatim passage text describing them).\n\n"
    "Base your verdict ONLY on the content given -- names alone are not sufficient "
    "evidence in either direction. Two different entities can share a name fragment "
    "or a generic title (e.g. two different floors, two different unnamed guards). "
    "The same entity can also be referred to very differently across chapters (e.g. "
    "'Katia' vs 'Katia Grimmsdottir', or 'the AI' vs 'the dungeon's system AI').\n\n"
    "Respond with ONLY a JSON object, no other text:\n"
    '{"verdict": "same"|"different"|"uncertain", "reasoning": "one or two sentences '
    'citing the specific detail(s) that drove your answer"}\n\n'
    "Use \"uncertain\" whenever the grounding content given is too thin, generic, or "
    "ambiguous to support a confident same/different call. Do not guess to force a "
    "same or different answer -- an incorrect \"uncertain\" costs a human a few "
    "minutes of review; an incorrect \"same\" or \"different\" corrupts the record "
    "permanently until someone happens to notice."
)


def find_candidates(cur, name, aliases, entity_type, limit=MAX_CANDIDATES_CHECKED):
    """
    Stage 1: loose pg_trgm similarity search against existing entities of the
    SAME entity_type. Compares every combination of the new mention's
    name+aliases against every existing candidate's name+aliases and takes
    the best (max) trigram similarity score. Returns up to `limit` candidates
    scoring at or above CANDIDATE_SIMILARITY_THRESHOLD, ordered best-first.

    Returns a list of dicts: {id, name, aliases, persona_text, score}.
    Empty list means "no plausibly-related existing entity" -- the caller
    should create the new entity normally with no LLM judgment call at all.
    """
    names_to_check = sorted({name, *(aliases or [])})
    cur.execute(
        """
        SELECT e.id, e.name, e.aliases, e.persona_text,
               MAX(similarity(nn.n, cand.a)) AS score
        FROM entities e
        CROSS JOIN unnest(%(names)s::text[]) AS nn(n)
        CROSS JOIN LATERAL unnest(e.aliases || ARRAY[e.name]) AS cand(a)
        WHERE e.entity_type::text = %(entity_type)s
        GROUP BY e.id, e.name, e.aliases, e.persona_text
        HAVING MAX(similarity(nn.n, cand.a)) >= %(threshold)s
        ORDER BY score DESC
        LIMIT %(limit)s
        """,
        {
            "names": names_to_check,
            "entity_type": entity_type,
            "threshold": CANDIDATE_SIMILARITY_THRESHOLD,
            "limit": limit,
        },
    )
    return [
        {
            "id": r[0],
            "name": r[1],
            "aliases": r[2] or [],
            "persona_text": r[3],
            "score": float(r[4]),
        }
        for r in cur.fetchall()
    ]


def _candidate_grounding(cur, candidate):
    """Persona_text if it exists (cheaper, curated); otherwise fall back to
    raw passage text for that entity. Returns None if neither exists -- an
    entity with zero descriptive content cannot be judged against, only
    flagged as uncertain (see judge_same_entity)."""
    if candidate.get("persona_text"):
        return candidate["persona_text"][:GROUNDING_CHAR_CAP]
    cur.execute(
        "SELECT passage_text FROM passages WHERE entity_id = %s ORDER BY id LIMIT %s",
        (candidate["id"], MAX_PASSAGES_FOR_GROUNDING),
    )
    rows = cur.fetchall()
    if not rows:
        return None
    return "\n".join(r[0] for r in rows)[:GROUNDING_CHAR_CAP]


def _new_entity_grounding(entity_data):
    """Grounding text for the entity currently being extracted, built from
    the passages the extractor just pulled for it this chapter (no extra DB
    round-trip needed -- it's already in entity_data)."""
    passages = entity_data.get("passages", [])
    texts = [p.get("passage_text", "") for p in passages[:MAX_PASSAGES_FOR_GROUNDING] if p.get("passage_text")]
    if not texts:
        return None
    return "\n".join(texts)[:GROUNDING_CHAR_CAP]


def judge_same_entity(client, new_name, new_type, new_aliases, new_grounding, candidate, candidate_grounding):
    """
    Stage 2: content-grounded same/different/uncertain judgment via Gemini.

    Any failure mode -- missing grounding text on either side, an API error,
    malformed JSON, or a verdict outside VALID_VERDICTS -- is forced to
    "uncertain" with an explanatory reasoning string. This function must
    never return "same" or "different" on anything less than a clean,
    validated model response grounded in real content on both sides.

    Returns (verdict: str, reasoning: str).
    """
    if not new_grounding or not candidate_grounding:
        return (
            "uncertain",
            "Insufficient grounding text to compare (no persona_text or passages "
            "available for the new mention and/or candidate id={}) -- never guessing "
            "an identity match without source content.".format(candidate["id"]),
        )

    prompt = (
        f"ENTITY A (new mention, being extracted now):\n"
        f"name: {new_name!r}\n"
        f"aliases: {new_aliases}\n"
        f"entity_type: {new_type}\n"
        f"content: {new_grounding}\n\n"
        f"ENTITY B (existing entity, id={candidate['id']}):\n"
        f"name: {candidate['name']!r}\n"
        f"aliases: {candidate['aliases']}\n"
        f"entity_type: {new_type}\n"
        f"content: {candidate_grounding}\n\n"
        "Are ENTITY A and ENTITY B the same in-story entity?"
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=JUDGE_SYSTEM_PROMPT + "\n\n" + prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        text = response.text.strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        data = json.loads(text)
        verdict = data.get("verdict")
        reasoning = data.get("reasoning", "")
        if verdict not in VALID_VERDICTS:
            return (
                "uncertain",
                f"Model returned invalid verdict {verdict!r}; forcing uncertain rather "
                f"than guessing which way it meant. Raw reasoning given: {reasoning}",
            )
        return verdict, reasoning
    except Exception as e:
        return (
            "uncertain",
            f"Resolution LLM call failed or returned unparseable output ({e}); "
            f"forcing uncertain rather than guessing same or different.",
        )


def _log_attempt(cur, new_name, new_type, candidate, verdict, reasoning, action_taken, resulting_entity_id):
    """Returns the new log row's id, so callers can backfill resulting_entity_id
    once a brand-new entity row's real id is known (it doesn't exist yet at the
    moment a 'different' or 'uncertain' verdict is logged)."""
    cur.execute(
        """
        INSERT INTO entity_resolution_log
            (new_entity_name, new_entity_type, candidate_entity_id, candidate_name_at_time,
             similarity_score, verdict, reasoning, action_taken, resulting_entity_id,
             reviewed)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            new_name, new_type, candidate["id"], candidate["name"],
            candidate.get("score"), verdict, reasoning, action_taken, resulting_entity_id,
            False if verdict == "uncertain" else True,
        ),
    )
    return cur.fetchone()[0]


def backfill_resulting_entity_id(cur, log_ids, entity_id):
    """Call once the caller has created the new entity row and knows its real
    id, for any log rows returned as pending from resolve_entity(). Keeps
    entity_resolution_log fully joinable to entities even for the
    'different'/'uncertain' branches where the row didn't exist yet at log
    time."""
    if not log_ids:
        return
    cur.execute(
        "UPDATE entity_resolution_log SET resulting_entity_id = %s WHERE id = ANY(%s)",
        (entity_id, log_ids),
    )


def resolve_entity(cur, client, entity_data, entity_type):
    """
    Full resolution, called by upsert_entity() ONLY after its own exact-match
    lookup has already failed (resolve_entity does not repeat that check).

    Returns (merge_into_id, pending_log_ids):
      - merge_into_id is the id of an EXISTING entity to merge this mention
        into, or None if the caller should create a brand-new entity row.
      - pending_log_ids is a list of entity_resolution_log row ids that need
        their resulting_entity_id backfilled via backfill_resulting_entity_id()
        once the caller knows the new row's real id (only non-empty when
        merge_into_id is None -- a merge already has a real id to log against).

    Checks up to MAX_CANDIDATES_CHECKED candidates, stopping at the first
    confident "same". Uncertain verdicts do not stop the loop (a later
    candidate might still be a confident match) but every uncertain verdict
    is logged for human review regardless of what happens with later
    candidates.
    """
    name = entity_data["name"].strip()
    aliases = entity_data.get("aliases", [])

    candidates = find_candidates(cur, name, aliases, entity_type)
    if not candidates:
        return None, []

    new_grounding = _new_entity_grounding(entity_data)
    pending_log_ids = []

    for candidate in candidates:
        candidate_grounding = _candidate_grounding(cur, candidate)
        verdict, reasoning = judge_same_entity(
            client, name, entity_type, aliases, new_grounding, candidate, candidate_grounding
        )

        if verdict == "same":
            _log_attempt(cur, name, entity_type, candidate, verdict, reasoning,
                         action_taken="auto_merged", resulting_entity_id=candidate["id"])
            logger.info(
                "entity_resolution: merged new mention %r into existing entity id=%s (%r) -- %s",
                name, candidate["id"], candidate["name"], reasoning,
            )
            if aliases or name.lower() != candidate["name"].lower():
                merged_aliases = list(aliases) + [name]
                cur.execute(
                    "UPDATE entities SET aliases = array(SELECT DISTINCT unnest(aliases || %s::text[])) WHERE id = %s",
                    (merged_aliases, candidate["id"]),
                )
            return candidate["id"], []

        elif verdict == "uncertain":
            log_id = _log_attempt(cur, name, entity_type, candidate, verdict, reasoning,
                                   action_taken="flagged_for_review", resulting_entity_id=None)
            pending_log_ids.append(log_id)
            logger.info(
                "entity_resolution: flagged new mention %r vs existing entity id=%s (%r) for human review -- %s",
                name, candidate["id"], candidate["name"], reasoning,
            )
            # keep checking remaining candidates -- do not stop the loop

        else:  # "different"
            log_id = _log_attempt(cur, name, entity_type, candidate, verdict, reasoning,
                                   action_taken="created_separately", resulting_entity_id=None)
            pending_log_ids.append(log_id)

    return None, pending_log_ids