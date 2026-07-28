"""
DCC Codex -- double-pass verification for structured stat/skill extraction.

Runs AFTER seeder/stats_extraction.py has already populated entity_stats/entity_skills
for an entity. For every not-yet-verified row, re-fetches the exact source passage text
and asks the model, independently, TWICE: "does this passage actually support this
specific claim (name, value, value_type)?" A row only counts as verified-clean if BOTH
independent passes say valid. Any row where either pass says invalid, or the two passes
disagree with each other, is left unverified and flagged in verify_notes for manual
review -- it is never auto-accepted just because the original extraction validated
mechanically.

This is deliberately a SEPARATE step from extraction, not folded into it, so that the
model doing the verifying is never the same call that produced the claim, and so a
future re-verification pass can be re-run without re-extracting anything.

Batches by entity, same continuation pattern as imager.py/stats_extraction.py:
entities.stats_verified_at IS NULL gates which entities are still due for a check.
"""

import json
import logging
import re
import time

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 1.0

VERIFY_SYSTEM = (
    "You are fact-checking structured data that was extracted from Dungeon Crawler Carl book "
    "passages. You will be given a list of CLAIMS, each with a claim_id, the exact passage text "
    "it was supposedly extracted from, and the extracted fact itself (a stat or skill name, a "
    "value, and a value_type of either 'absolute' -- meaning the passage states this is the "
    "CURRENT TOTAL -- or 'delta' -- meaning the passage states this is a CHANGE/BONUS/POINTS "
    "ADDED, not a total).\n\n"
    "For each claim, decide if it is VALID: does the passage text, read plainly, actually support "
    "this exact name + value + value_type combination? Reject as INVALID if: the value isn't "
    "stated in the passage at all, the value_type is backwards (e.g. a '+N to X' bonus phrase was "
    "labeled absolute, or a 'X was now N' total phrase was labeled delta), the stat/skill name "
    "doesn't match what's described, or the claim reads more into the passage than it actually "
    "says.\n\n"
    "Be strict and literal. If you are unsure, mark it INVALID rather than valid -- the cost of "
    "wrongly rejecting a correct claim is much lower than the cost of confirming a wrong one.\n\n"
    "Respond with ONLY a JSON array, one element per claim, in the same order given:\n"
    '{"claim_id": <int>, "valid": true|false, "note": "<short (<15 word) reason, especially if invalid>"}'
)


def run_verify_migrate(conn) -> None:
    cur = conn.cursor()
    cur.execute("ALTER TABLE entity_stats ADD COLUMN IF NOT EXISTS verify_pass1 BOOLEAN")
    cur.execute("ALTER TABLE entity_stats ADD COLUMN IF NOT EXISTS verify_pass2 BOOLEAN")
    cur.execute("ALTER TABLE entity_stats ADD COLUMN IF NOT EXISTS verify_notes TEXT")
    cur.execute("ALTER TABLE entity_stats ADD COLUMN IF NOT EXISTS verified_at TIMESTAMP")
    cur.execute("ALTER TABLE entity_skills ADD COLUMN IF NOT EXISTS verify_pass1 BOOLEAN")
    cur.execute("ALTER TABLE entity_skills ADD COLUMN IF NOT EXISTS verify_pass2 BOOLEAN")
    cur.execute("ALTER TABLE entity_skills ADD COLUMN IF NOT EXISTS verify_notes TEXT")
    cur.execute("ALTER TABLE entity_skills ADD COLUMN IF NOT EXISTS verified_at TIMESTAMP")
    cur.execute("ALTER TABLE entities ADD COLUMN IF NOT EXISTS stats_verified_at TIMESTAMP")
    conn.commit()
    cur.close()
    logger.info("Migration: verify_pass1/2, verify_notes, verified_at on entity_stats/entity_skills; "
                "entities.stats_verified_at ensured.")


def _fetch_claims_for_entity(conn, entity_id):
    """Returns list of dicts: {table, row_id, kind_name, value, value_type, reason, passage_text}."""
    cur = conn.cursor()
    cur.execute("""
        SELECT es.id, es.stat_name, es.value, es.value_type, es.reason, p.passage_text
        FROM entity_stats es
        JOIN passages p ON p.id = es.source_passage_id
        WHERE es.entity_id = %s AND es.verified_at IS NULL
        ORDER BY es.id
    """, (entity_id,))
    stat_rows = cur.fetchall()

    cur.execute("""
        SELECT sk.id, sk.skill_name, sk.level, sk.value_type, sk.reason, p.passage_text
        FROM entity_skills sk
        JOIN passages p ON p.id = sk.source_passage_id
        WHERE sk.entity_id = %s AND sk.verified_at IS NULL
        ORDER BY sk.id
    """, (entity_id,))
    skill_rows = cur.fetchall()
    cur.close()

    claims = []
    for row_id, name, value, value_type, reason, passage_text in stat_rows:
        claims.append({
            "table": "entity_stats", "row_id": row_id, "name": name, "value": value,
            "value_type": value_type, "reason": reason, "passage_text": passage_text,
        })
    for row_id, name, value, value_type, reason, passage_text in skill_rows:
        claims.append({
            "table": "entity_skills", "row_id": row_id, "name": name, "value": value,
            "value_type": value_type, "reason": reason, "passage_text": passage_text,
        })
    return claims


def _run_one_verify_pass(client, entity_name, claims):
    """Returns dict: claim_index -> (valid: bool, note: str). claim_index is position in `claims`."""
    if not claims:
        return {}

    numbered = "\n\n".join(
        f"[claim_id={i}] Passage: \"{c['passage_text']}\"\n"
        f"Extracted: name={c['name']!r}, value={c['value']!r}, value_type={c['value_type']!r}, "
        f"reason={c['reason']!r}"
        for i, c in enumerate(claims)
    )
    prompt = VERIFY_SYSTEM + f"\n\nEntity: {entity_name}\n\nCLAIMS:\n{numbered}"

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        text = response.text.strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        raw = json.loads(text)
    except Exception as e:
        logger.warning("  Verification call failed for %s: %s", entity_name, e)
        return {}

    result = {}
    for entry in raw:
        try:
            cid = entry["claim_id"]
            valid = entry["valid"]
            note = entry.get("note", "")
            if not isinstance(valid, bool) or not isinstance(cid, int) or cid not in range(len(claims)):
                continue
            result[cid] = (valid, note)
        except (KeyError, TypeError):
            continue
    return result


def run_verification(conn, gemini_api_key, entity_ids=None, batch_size=50):
    """Double-pass-verify extracted stat/skill claims.

    If entity_ids is given, process exactly those entities. Otherwise, pick the next
    `batch_size` entities with unverified entity_stats/entity_skills rows, ordered by id.
    Marks entities.stats_verified_at once ALL of that entity's rows have been checked
    (verified_at set on every row, whether passed or flagged) -- so this is safely
    re-runnable and won't reprocess an entity twice.
    """
    run_verify_migrate(conn)
    client = genai.Client(api_key=gemini_api_key)

    cur = conn.cursor()
    if entity_ids:
        cur.execute("SELECT id, name FROM entities WHERE id = ANY(%s) ORDER BY id", (entity_ids,))
    else:
        cur.execute("""
            SELECT DISTINCT e.id, e.name
            FROM entities e
            WHERE e.stats_verified_at IS NULL
              AND (
                EXISTS (SELECT 1 FROM entity_stats es WHERE es.entity_id = e.id AND es.verified_at IS NULL)
                OR EXISTS (SELECT 1 FROM entity_skills sk WHERE sk.entity_id = e.id AND sk.verified_at IS NULL)
              )
            ORDER BY e.id
            LIMIT %s
        """, (batch_size,))
    targets = cur.fetchall()
    cur.close()

    logger.info("Running double-pass verification for %d entities...", len(targets))
    total_checked = 0
    total_passed_both = 0
    total_flagged = 0

    for entity_id, name in targets:
        claims = _fetch_claims_for_entity(conn, entity_id)
        if not claims:
            cur = conn.cursor()
            cur.execute("UPDATE entities SET stats_verified_at = NOW() WHERE id = %s", (entity_id,))
            conn.commit()
            cur.close()
            continue

        logger.info("  %s: %d unverified claims", name, len(claims))
        pass1 = _run_one_verify_pass(client, name, claims)
        time.sleep(RATE_LIMIT_SECONDS)
        pass2 = _run_one_verify_pass(client, name, claims)
        time.sleep(RATE_LIMIT_SECONDS)

        cur = conn.cursor()
        n_passed = n_flagged = 0
        for i, c in enumerate(claims):
            v1, note1 = pass1.get(i, (False, "pass1 missing/malformed response"))
            v2, note2 = pass2.get(i, (False, "pass2 missing/malformed response"))
            both_valid = bool(v1) and bool(v2)
            note = note1 if not v1 else (note2 if not v2 else "both passes confirmed valid")
            if not both_valid:
                note = f"pass1={v1} ({note1}) | pass2={v2} ({note2})"

            cur.execute(f"""
                UPDATE {c['table']}
                SET verify_pass1 = %s, verify_pass2 = %s, verify_notes = %s, verified_at = NOW()
                WHERE id = %s
            """, (bool(v1), bool(v2), note, c["row_id"]))

            if both_valid:
                n_passed += 1
            else:
                n_flagged += 1
                logger.warning("    FLAGGED %s.id=%s (%s=%r): %s",
                                c["table"], c["row_id"], c["name"], c["value"], note)

        cur.execute("UPDATE entities SET stats_verified_at = NOW() WHERE id = %s", (entity_id,))
        conn.commit()
        cur.close()

        logger.info("    -> %d passed both passes, %d flagged", n_passed, n_flagged)
        total_checked += len(claims)
        total_passed_both += n_passed
        total_flagged += n_flagged

    logger.info("Done. %d entities, %d claims checked, %d passed both passes, %d flagged for review.",
                len(targets), total_checked, total_passed_both, total_flagged)
    return total_checked, total_passed_both, total_flagged
