"""Alias-collision check.

Nothing in the schema stops two different entities from claiming the same
alias string. That's a latent risk for any future text-matching/linking
feature, and also another lens on the duplicate-entity problem fixed
2026-07-27 -- two entities sharing an alias is sometimes a sign they're
actually the same thing (e.g. this check's first real run found "Katia" /
"Katia Grim" / "Katia Grimmsdottir" as 3 separate crawler entities all
claiming the alias "katia" -- a likely duplicate the casing-fold dedup script
couldn't catch since the names differ by more than casing/articles).

IMPORTANT -- unlike audit_duplicates.py, a nonzero result here is NOT itself
a bug to drive to zero. In a corpus this size, lots of different characters
genuinely get called the same generic thing in narration ("the orc", "dad",
"castle", "potion") without being the same entity. This script's job is to
surface candidates for a human to read and judge, same as the extraction
coverage audit -- not to assert they're all bugs.

Also dedupes each entity's OWN alias array case-insensitively before
comparing across entities -- first version double-counted entities that had
the same alias twice in their own array with different casing (a smaller,
separate data-quality nit worth knowing about but not what this check is for).

Output is capped (default 40 groups, sorted by claimant count) to avoid
blowing past tool output limits -- the first real run against this corpus
had 163 groups and crashed the caller before this cap was added.

Only checks within-entity_type collisions by default (cross-type homonyms are
common and expected -- e.g. "The Scavenger's Daughter" is both an item and a
crawler nickname, confirmed as real homonymy during the 2026-07-27 dedup pass,
not a bug). Pass --cross-type to also compute those (much noisier).

Usage: python3 audit_alias_collisions.py [--cross-type] [--limit 40]
Exit 0 if no same-type collisions, 1 if any found (informational, see above).
"""
import psycopg2, os, sys, json, argparse
from collections import defaultdict

def get_conn():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'], dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'], password=os.environ['POSTGRES_PASSWORD'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cross-type', action='store_true')
    ap.add_argument('--limit', type=int, default=40)
    args = ap.parse_args()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, entity_type, aliases FROM entities")
    rows = cur.fetchall()
    cur.close(); conn.close()

    claims = defaultdict(list)
    for (id_, name, etype, aliases) in rows:
        seen_lower = set()
        for a in (aliases or []):
            if a.lower() in seen_lower:
                continue  # dedupe this entity's own case-variant self-collision
            seen_lower.add(a.lower())
            claims[a.lower()].append({'id': id_, 'name': name, 'entity_type': etype})

    same_type, cross_type = [], []
    for alias_lower, claimants in claims.items():
        if len(claimants) < 2:
            continue
        types_present = set(c['entity_type'] for c in claimants)
        (same_type if len(types_present) == 1 else cross_type).append(
            {'alias': alias_lower, 'claimants': claimants})

    same_type.sort(key=lambda g: -len(g['claimants']))

    result = {
        'check': 'audit_alias_collisions',
        'note': 'nonzero is expected in a corpus this size -- read entries, do not assume all are bugs',
        'same_type_collisions_total': len(same_type),
        'same_type_shown': min(len(same_type), args.limit),
        'same_type_details': same_type[:args.limit],
    }
    if args.cross_type:
        result['cross_type_collisions_total'] = len(cross_type)
        result['cross_type_shown'] = min(len(cross_type), args.limit)
        result['cross_type_details'] = cross_type[:args.limit]

    print(json.dumps(result, indent=1, ensure_ascii=False))
    sys.exit(1 if same_type else 0)

if __name__ == '__main__':
    main()
