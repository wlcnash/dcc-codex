"""
DCC Codex — System AI persona writer.

For each entity with sufficient passages, generates a short profile written in the
System AI voice from the DCC books: bureaucratic game-show dystopia, slightly sinister,
corporate-clinical, with dry dark humor. Runs the DB migration first (idempotent).

2026-07-28 fix (post live audit showing 1848/1948 -- ~95% -- of ALL persona_text rows
sitewide cut off mid-sentence, caught by manually reading Carl's persona_text and
finding it truncated at 81 characters): gemini-3.6-flash spends a large, variable
number of tokens on internal "thinking" before writing any visible output (487-907+
tokens observed in direct reproduction). The original max_output_tokens=512 budget
covers thinking alone in many cases, leaving almost nothing for the actual profile
text, so the response silently hits MAX_TOKENS after only a few words. The API still
returns a non-empty response.text in this case, so the old code (which only checked
for an empty string and a set of scratchpad-leakage patterns) accepted these truncated
fragments as valid and wrote them straight to the DB.

Fixed by (a) raising max_output_tokens to a much more generous budget (thinking can
vary a lot, so this leaves real headroom instead of just nudging the old cap), and
(b) explicitly checking finish_reason and rejecting (leaving NULL for retry) any
response that still hit MAX_TOKENS, exactly like the existing scratchpad-leakage
rejection path -- never trust a response just because response.text is non-empty.
Confirmed via direct reproduction: the same prompt that produced a 64-char fragment
under the old config produced a clean, complete, finish_reason=STOP profile at
max_output_tokens=2048.

2026-07-28 fix #2 (Wes's feedback after reading Carl's regenerated persona live):
the persona was describing acute, floor-specific combat state ("covered in blood, bug
chitin, and white goo... deep scalp wounds, phantom limb pain, and severe arm burns")
as if it were the character's permanent description. Root cause: the passage query
below pulled ANY physical/action passage regardless of the existing `is_durable`
classification (see seeder/permanence.py) -- so a one-off injury-mid-fight passage was
just as likely to get sampled into the "who is this person" blurb as a genuinely
permanent trait. Since persona_text is ONE non-floor-keyed field per entity (there is
no per-floor persona regeneration, and there should not be one -- per Wes: "shouldn't
we keep the status specifics out of the summary since they change floor to floor...
he should look like the same person," meaning only the PORTRAIT should vary by floor,
not the identity blurb), any transient detail baked into persona_text would read as
permanently, perpetually true no matter which floor a visitor is looking at, which is
wrong on its face. Fixed by filtering physical/action passages to `is_durable = TRUE
OR is_durable IS NULL` (excluding only passages positively classified transient) and
adding an explicit prompt instruction to describe durable identity/personality, not
current injury/combat/gear state.
"""

import logging
import re
import time
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 1.0
MAX_OUTPUT_TOKENS = 4096

# The voice: Dungeon System AI — omniscient dungeon announcer, treats everything as metrics,
# bureaucratic corporate-speak crossed with reality-TV energy, slightly ominous.
#
# 2026-07-29 fix: Wes flagged a live entity page ("Princess Donut") where the persona opened
# with "Contestant classification: quadrupedal tortoiseshell Persian designated Princess Donut
# the Queen Anne Chonk," -- read as AI slop. Root-caused via direct DB query: 9 of 1948
# persona_text rows opened with a near-identical "[X] classification[:| indicates| update]"
# templated declaration. The cause was this exact prompt: the voice example below literally
# said "contestant designation" and "threat classification" as sample bureaucratic phrases,
# and the model converged on reusing that exact "X classification:" shape verbatim as a
# stock sentence-opener across multiple entities -- a formulaic, rigid label-and-colon
# declaration is itself a classic AI-slop tell independent of word choice. Two separate fixes:
# (1) "contestant" was never the book's term for dungeon competitors (confirmed 2026-07-29,
# round 7: "crawler" appears 228 times in the corpus vs. "contestant" once, in unrelated
# dialogue) -- swapped the example phrase to "crawler designation." (2) added an explicit rule
# banning the rigid "[Label]: [description]" opening format the model kept reaching for,
# separate from the terminology fix, since a differently-worded version of the same formulaic
# structure would still read as AI-generated.
PERSONA_PROMPT = """You are the System AI of the Dungeon Crawler Carl multiverse — the dungeon's omniscient announcer AI that manages the live broadcast of crawlers competing through its floors.

Your voice is:
- Corporate bureaucratic ("crawler designation," "floor segment," "processing status")
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
- Do NOT open with a rigid "[Label]: [description]" declaration (e.g. "Contestant
  classification:", "Threat classification:", "Designation:") — that exact templated shape is
  itself an AI-generated-text tell regardless of which words fill it in. Open with an actual
  sentence instead.
- Never use the word "contestant" for a dungeon participant — the book's own term is "crawler"
  (confirmed directly against the source text: "crawler" appears 228 times in the corpus,
  "contestant" appears exactly once, in unrelated dialogue). Use "crawler" instead.
- No heading, no quotes, no markdown — just the raw profile text
- IMPORTANT: this profile is shown to visitors regardless of which floor/point in the story they're
  reading about, so it must describe PERMANENT, DURABLE traits only — build, personality, standing
  role, established backstory. Do NOT describe a specific in-progress injury, a specific fight's
  blood/gore, or transient current-moment gear/condition as if it were a permanent description, even
  if the source passages mention it. If a passage only shows a passing combat state, either omit that
  detail or, if it's illustrative of personality/resilience, phrase it as a general pattern of behavior
  rather than "he is currently covered in X."

Entity Name: {name}
Entity Type: {entity_type}

Source Passages (draw only from these):
{passages}

System AI profile:"""


# Patterns that indicate the model leaked drafting scratchpad / meta-commentary
# instead of a clean finished profile (e.g. "*   *Drafting Attempt 1:*",
# "Length: 3-5 sentences.", "Let's stay strictly factual..."). Found via a live
# audit (2026-07-28) showing 137/2029 entities with persona_text corrupted this
# way, because the original code only checked for an empty string. This never
# trusts unvalidated LLM output — same discipline already used in floors.py,
# permanence.py, reclassify_types.py, and entity_resolution.py.
_INVALID_PERSONA_PATTERNS = [
    r"\*\*",                                   # markdown bold
    r"(?im)^\s*[-*]\s*\*",                      # bullet-prefixed meta lines ("*   *Drafting...")
    r"(?i)\bdrafting\b",
    r"(?i)\battempt\s*\d",
    r"(?im)^\s*length\s*:",
    r"(?im)^\s*voice\s*:",
    r"(?im)^\s*fact[- ]check",
    r"(?im)^\s*sentence\s*\d",
    r"(?i)\blet'?s\s+(stay|make sure|double[- ]check|verify|keep)\b",
    r"(?i)\bstrictly factual\b",
    r"(?i)\bno heading,?\s*no quotes\b",
    r"(?im)^\s{0,3}#{1,6}\s",                   # markdown headings
    # 2026-07-29: a second, previously-undetected leak shape found via direct DB audit --
    # 8 entities had persona_text that was actually a fragment of the model's own outline/
    # planning scratchpad, e.g. "(Comparison):*", "(Physical/Visuals/Auditory):*",
    # "(Optional 4th):* A dry observation...", "Sentence 3: Focus on...". These don't contain
    # "drafting" or "attempt N" so the original patterns above missed them entirely, and
    # would have kept being stored as broken persona text on every future run forever.
    r"^\s*\(",                                  # response starts with a parenthetical, never legitimate prose
    r"\([\w\s/]{2,40}\):\s*\*",                 # "(Label/Label):*" outline-section markers
    # 2026-07-29 (round 10, live-corpus miss): the 2026-07-29 prompt fix added a RULE telling
    # the model not to open with a rigid "[Label]: [description]" declaration, and the
    # existing 1948-row audit was re-run afterward -- but that audit only checked the leak
    # patterns above, which say nothing about this shape. The rule was a soft instruction with
    # no hard gate behind it, and it only got RETROACTIVELY applied to entities that happened
    # to also contain the word "contestant". A broader sweep (prompted by Wes catching that
    # the site still had this problem after the "0/1948 fail" report) found 151 more entities
    # opening with the exact same rigid shape using other label words entirely -- "Item
    # designation:", "Ability Classification:", "Threat classification:", "Crawler
    # designation:", "Faction designation:", "System AI style parameters:", etc. -- proving the
    # instruction alone doesn't reliably stop the model, and that "we told it not to" is not
    # the same as "it can't get through validation." This pattern makes the rule a real gate:
    # reject any response that opens with a short (<=5 word) label phrase immediately followed
    # by a colon, regardless of which words fill it in, exactly the same shape/not-vocabulary
    # principle already established for the "contestant" fix.
    r"^\s*[A-Za-z][\w'/-]*(?:\s+[A-Za-z][\w'/-]*){0,4}\s*:\s",
    # 2026-07-29: broadened while investigating the above -- two more leak shapes found live
    # in the same sweep that the existing patterns should have caught but didn't: id=87 "Apple
    # Core" had a trailing "-> Let's make it more System-esque." (old pattern only matched
    # "let's stay/make sure/double-check/verify/keep", not "let's make it"), and id=1190
    # "Monk Seals" had literally leaked the PERSONA_PROMPT's own voice-description text
    # ("System AI style parameters: Corporate form-speak, ...") instead of writing a profile.
    # The voice never has legitimate reason to say "let's" (it's a third-person System AI, not
    # a collaborative first-person writer), so reject the phrase outright rather than trying to
    # enumerate every possible continuation.
    r"(?i)\blet'?s\b",
    r"(?i)\bstyle parameters\b",
]


def _is_valid_persona(text: str) -> bool:
    """Reject LLM drafting-scratchpad / meta-commentary leakage before it ever
    reaches the database. A malformed response must never be stored — leave
    persona_text NULL instead so the next run retries it cleanly."""
    if not text:
        return False
    for pattern in _INVALID_PERSONA_PATTERNS:
        if re.search(pattern, text):
            return False
    return True


def run_migrate(conn) -> None:
    """Ensure persona_text column exists. Safe to run multiple times."""
    cur = conn.cursor()
    cur.execute("ALTER TABLE entities ADD COLUMN IF NOT EXISTS persona_text TEXT")
    conn.commit()
    cur.close()
    logger.info("Migration: persona_text column ensured.")


def run_persona(conn, gemini_api_key: str, batch_size: int = 999999, entity_ids=None) -> int:
    """Generate System AI persona text for entities without it.

    If entity_ids is given, (re)generate exactly those entities regardless of whether
    persona_text is already set -- targeted mode, used for regenerating known-bad rows.
    Otherwise, only fills entities where persona_text IS NULL (original batch mode).
    """
    run_migrate(conn)

    client = genai.Client(api_key=gemini_api_key)

    cur = conn.cursor()
    if entity_ids:
        cur.execute("""
            SELECT e.id, e.name, e.entity_type
            FROM entities e
            WHERE e.id = ANY(%s) AND EXISTS (SELECT 1 FROM passages p WHERE p.entity_id = e.id)
            ORDER BY e.id
        """, (entity_ids,))
    else:
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
    rejected = 0
    truncated = 0

    for entity_id, entity_name, entity_type in entities:
        # Fetch passages — prioritize physical + personality; cap at ~3000 chars.
        # 2026-07-28: physical/action passages are filtered to durable-or-unclassified
        # only (is_durable IS NOT FALSE) so a one-off injury/combat-state passage never
        # gets sampled into this entity's permanent, floor-agnostic identity blurb.
        # personality/backstory/ability passages are inherently non-transient, so they're
        # left unfiltered.
        cur = conn.cursor()
        cur.execute("""
            SELECT p.passage_text, p.passage_type
            FROM passages p
            WHERE p.entity_id = %s
              AND (p.passage_type NOT IN ('physical', 'action') OR p.is_durable IS NOT FALSE)
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

            # Reject silently-truncated responses BEFORE looking at the text at all --
            # a MAX_TOKENS finish means the model was cut off mid-thought, regardless
            # of whether response.text happens to look plausible at a glance.
            finish_reason = None
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
            if finish_reason is not None and str(finish_reason).endswith("MAX_TOKENS"):
                truncated += 1
                logger.warning(
                    f"  REJECTED truncated (MAX_TOKENS) persona for '{entity_name}' "
                    f"(id={entity_id}), leaving NULL for retry: {(response.text or '')[:80]!r}"
                )
                continue

            persona_text = (response.text or "").strip()
            if not persona_text:
                logger.warning(f"  Empty persona for '{entity_name}', skipping")
                continue

            if not _is_valid_persona(persona_text):
                rejected += 1
                logger.warning(
                    f"  Rejected malformed/scratchpad persona for '{entity_name}' "
                    f"(id={entity_id}), leaving NULL for retry: {persona_text[:80]!r}"
                )
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

    logger.info(
        f"Persona generation complete: {count} entities processed, "
        f"{rejected} rejected as malformed, {truncated} rejected as truncated (MAX_TOKENS)."
    )
    return count
