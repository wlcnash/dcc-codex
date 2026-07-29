"""DCC Codex — verbatim item/gear descriptions.

2026-07-29: Wes's explicit rule after spot-checking a random item ("Def Leppard cart"):
its persona_text was an AI paraphrase ("provides massive rapid-response utility... Tactical
logs highlight...") that invented flavor text found nowhere in the book, when the actual
source passage was already sitting right there: "Each rapid-response cart was about the
width and height of a cargo van, and maybe one and a half times longer." Every item-type
entity goes through the exact same persona.py LLM-paraphrase pipeline as crawlers/npcs/mobs,
which makes sense for a character (a "voice" adds value) but not for an inanimate object --
there is no upside to paraphrasing a physical description when the real text is already
stored, and every paraphrase is a chance to fabricate a detail that isn't in the source.

Rule going forward: entity_type='item' entities get their persona_text assembled from
VERBATIM source passage text only. No Gemini call, no paraphrase, ever, for this entity
type. Enforced two ways: (1) the assembly function below only ever copies passage_text
substrings directly, never generates new text; (2) _is_verbatim_item_text() is a standing,
rerunnable check (see audit_item_verbatim.py) that any item's stored description is
provably built entirely from its own source passages -- run it any time to confirm this
rule still holds, including after this file's own writes.
"""

import logging

logger = logging.getLogger(__name__)

# Prefer passages that actually describe/explain the object itself; a physical-description
# passage is the most useful "what does this look like" text, ability/action passages cover
# "what does it do", backstory/personality are rare for inanimate objects but not impossible
# (an heirloom item with a story) and better than nothing if that's all there is.
ITEM_PASSAGE_TYPE_PRIORITY = ["physical", "ability", "action", "backstory", "personality", "other"]

MAX_CHARS = 900
JOIN_SEP = "\n\n"


def _is_verbatim_item_text(description: str, source_passage_texts: list) -> bool:
    """The standing rule check: every paragraph of `description` (split on the same
    separator the assembly function joins with) must appear character-for-character
    inside at least one of the entity's own source passages. This is deliberately NOT
    fuzzy -- an item description that isn't an exact quote fails, full stop. Use this to
    audit the whole item corpus at any time (see audit_item_verbatim.py), not just to
    gate new writes."""
    if not description or not description.strip():
        return False
    segments = [s.strip() for s in description.split(JOIN_SEP) if s.strip()]
    if not segments:
        return False
    for seg in segments:
        if not any(seg in pt for pt in source_passage_texts):
            return False
    return True


def run_item_descriptions(conn, entity_ids=None) -> dict:
    """Rebuild persona_text for item-type entities from verbatim source passages only.
    If entity_ids is given, only those entities are processed (targeted re-run mode,
    e.g. to fix a specific flagged item); otherwise every entity_type='item' row is
    (re)built. Safe to re-run -- always overwrites with a freshly assembled, freshly
    verified verbatim description."""
    cur = conn.cursor()
    if entity_ids:
        cur.execute("SELECT id, name FROM entities WHERE id = ANY(%s) AND entity_type = 'item'", (entity_ids,))
    else:
        cur.execute("SELECT id, name FROM entities WHERE entity_type = 'item'")
    items = cur.fetchall()
    cur.close()

    updated = 0
    skipped_no_passages = 0
    rejected_invalid = 0

    for entity_id, name in items:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT passage_text, passage_type FROM passages
            WHERE entity_id = %s
            ORDER BY CASE passage_type
                WHEN 'physical' THEN 1 WHEN 'ability' THEN 2 WHEN 'action' THEN 3
                WHEN 'backstory' THEN 4 WHEN 'personality' THEN 5 ELSE 6 END, id
            """,
            (entity_id,),
        )
        rows = cur.fetchall()
        cur.close()

        if not rows:
            skipped_no_passages += 1
            logger.info(f"  '{name}' (id={entity_id}): no passages, skipping.")
            continue

        all_texts = [r[0] for r in rows]
        segments, seen, total_len = [], set(), 0
        for text, _ptype in rows:
            t = (text or "").strip()
            if not t or t in seen:
                continue
            if segments and total_len + len(t) > MAX_CHARS:
                break
            segments.append(t)
            seen.add(t)
            total_len += len(t)

        description = JOIN_SEP.join(segments)

        if not _is_verbatim_item_text(description, all_texts):
            # Should never actually trigger, since segments are copied directly from
            # all_texts above -- this is the defensive gate in case a future edit to the
            # assembly logic (e.g. adding any transformation/truncation mid-word) breaks
            # the verbatim guarantee without anyone noticing.
            rejected_invalid += 1
            logger.warning(
                f"  REJECTED non-verbatim assembly for '{name}' (id={entity_id}) -- "
                f"this should be impossible with the current assembly logic; investigate."
            )
            continue

        cur = conn.cursor()
        cur.execute("UPDATE entities SET persona_text = %s WHERE id = %s", (description, entity_id))
        conn.commit()
        cur.close()

        updated += 1
        logger.info(f"  '{name}' (id={entity_id}) -> {len(segments)} verbatim passage(s), {len(description)} chars")

    logger.info(
        f"Item description rebuild complete: {updated} updated, {skipped_no_passages} skipped "
        f"(no passages), {rejected_invalid} rejected (should be 0)."
    )
    return {
        "updated": updated,
        "skipped_no_passages": skipped_no_passages,
        "rejected_invalid": rejected_invalid,
        "total": len(items),
    }
