"""Casing/spacing/hyphenation duplicate-entity scan.

2026-08-01: found via a live report (bopca/Bopca, the same unnamed shopkeeper split
into entity_type=mob and entity_type=npc purely by name casing) that this corpus has
had this exact bug class sitting around, unfixed, since at least the 2026-07-27
taxonomy migration (which explicitly flagged but deferred Borant/Borant Corporation
variants and gnolls/Gnolls). A broader scan the same day, normalizing away case AND
punctuation/spacing (not just case), turned up 6 more real candidates in one pass:
Loop-de-Loop/Loop-De-Loop, Monk seals/Monk Seals, the Sledge/The Sledge, All Tree/
all-tree, Crawl Con/CrawlCon (all confirmed genuine dupes by reading their passages --
merged via merge_entities.py) and WAR CRIME/War Crime (confirmed NOT a dupe -- a
spellbook item and the spell it teaches, a normal item-grants-ability pairing, left
alone).

This is a DETECTION-ONLY tool, deliberately not an auto-merge. Every prior fix in this
class required reading the actual passages to tell a real duplicate (same thing,
split by name normalization) from a coincidental name collision (two different
things that happen to share a name, e.g. an item and its associated spell, which is
common and expected in this universe's naming). Never merge just because two entities
normalize to the same string -- confirm the entities represent the same thing by
reading their passages first, exactly as documented in project memory for every merge
this script would have flagged historically.

Read-only, safe to run any time.
Usage: python3 audit_casing_duplicates.py
Exit 0 if no candidate groups found, 1 if any exist (informational flag for a human
to review with merge_entities.py, not raised because the code did anything wrong).
"""
import psycopg2, os, re, sys, json


def get_conn():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'], dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'], password=os.environ['POSTGRES_PASSWORD'])


def normalize(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


def main():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, entity_type FROM entities")
    rows = cur.fetchall()

    groups = {}
    for entity_id, name, entity_type in rows:
        key = normalize(name)
        if not key:
            continue
        groups.setdefault(key, []).append({'id': entity_id, 'name': name, 'entity_type': entity_type})

    candidates = {k: v for k, v in groups.items() if len(v) > 1}

    # Pull one sample passage per candidate entity to make manual review faster --
    # this is exactly the step that has caught every real vs. coincidental case so far.
    for key, members in candidates.items():
        for m in members:
            cur.execute("SELECT passage_type, passage_text FROM passages WHERE entity_id = %s LIMIT 1", (m['id'],))
            row = cur.fetchone()
            m['sample_passage'] = {'type': row[0], 'text': row[1][:200]} if row else None

    cur.close()
    conn.close()

    result = {
        'check': 'audit_casing_duplicates',
        'total_entities': len(rows),
        'candidate_groups': len(candidates),
        'groups': candidates,
        'note': 'Detection only -- read each group\'s sample_passage (and full passages if '
                'ambiguous) before merging. Not every normalized-name collision is a real '
                'duplicate (e.g. an item and the ability it teaches can legitimately share a name).',
    }
    print(json.dumps(result, indent=1, ensure_ascii=False))
    sys.exit(1 if candidates else 0)


if __name__ == '__main__':
    main()
