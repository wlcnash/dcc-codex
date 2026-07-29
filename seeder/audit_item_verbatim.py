"""Verbatim item-description rule check.

2026-07-29: Wes's rule -- item/gear entity descriptions must be verbatim book text, never
an LLM paraphrase (see item_descriptions.py for the full rationale and the assembly
function this audits). This is the standing, rerunnable enforcement mechanism for that
rule: for every entity_type='item' row with a persona_text, confirms it is provably built
entirely out of that entity's own source passages (each paragraph must appear
character-for-character in at least one passage_text row). Anything that fails this is
either a stale pre-rule row that hasn't been rebuilt yet, or a future regression where
something wrote a paraphrase back into an item's description -- both should be
investigated and re-run through item_descriptions.run_item_descriptions().

Read-only, safe to run any time.
Usage: python3 audit_item_verbatim.py
Exit 0 if every item passes, 1 if anything is flagged.
"""
import psycopg2, os, sys, json

from item_descriptions import _is_verbatim_item_text


def get_conn():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'], dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'], password=os.environ['POSTGRES_PASSWORD'])


def main():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, persona_text FROM entities WHERE entity_type = 'item'")
    items = cur.fetchall()

    flagged = []
    no_description = []
    checked = 0

    for entity_id, name, description in items:
        if not description:
            no_description.append({'id': entity_id, 'name': name})
            continue
        cur.execute("SELECT passage_text FROM passages WHERE entity_id = %s", (entity_id,))
        source_texts = [r[0] for r in cur.fetchall()]
        checked += 1
        if not _is_verbatim_item_text(description, source_texts):
            flagged.append({'id': entity_id, 'name': name, 'description_preview': (description or '')[:150]})

    cur.close(); conn.close()

    result = {
        'check': 'audit_item_verbatim',
        'total_items': len(items),
        'checked': checked,
        'no_description': no_description,
        'flagged_non_verbatim': flagged,
        'flagged_count': len(flagged),
    }
    print(json.dumps(result, indent=1, ensure_ascii=False))
    sys.exit(1 if flagged else 0)


if __name__ == '__main__':
    main()
