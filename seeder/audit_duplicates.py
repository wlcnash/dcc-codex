"""Duplicate-entity health check.

Finds entities that likely refer to the same thing but exist as separate rows,
based on casing/leading-article-normalized name collisions within the same
entity_type. This is the mechanical, narrow check: pure string-normalization
duplicates (e.g. "Borant Corporation" vs "borant corporation" vs
"The Borant Corporation"). It will NOT catch semantically-identical entities
that use genuinely different wording (e.g. the old AI/System AI/dungeon AI
cluster) -- that class of duplicate requires reading passages, not string
matching (see feedback_never_guess_read_source memory note). Run
audit_entity_types.py and manual review for that class.

Safe to run any time; read-only, no writes.

Usage: python3 audit_duplicates.py
Exit code 0 if no duplicates found, 1 if duplicates found (for CI/cron use).
"""
import psycopg2, os, re, sys, json

def get_conn():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'],
    )

def norm(name):
    return re.sub(r'^(the|The) ', '', name).lower()

def find_duplicate_groups(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, name, entity_type FROM entities ORDER BY entity_type, name")
    rows = cur.fetchall()
    cur.close()
    groups = {}
    for (id_, name, etype) in rows:
        key = (norm(name), etype)
        groups.setdefault(key, []).append({'id': id_, 'name': name})
    return {k: v for k, v in groups.items() if len(v) > 1}

def main():
    conn = get_conn()
    dup_groups = find_duplicate_groups(conn)
    conn.close()
    total_rows = sum(len(v) for v in dup_groups.values())
    result = {
        'check': 'audit_duplicates',
        'duplicate_groups': len(dup_groups),
        'duplicate_rows': total_rows,
        'groups': [
            {'norm': k[0], 'entity_type': k[1], 'entities': v}
            for k, v in dup_groups.items()
        ],
    }
    print(json.dumps(result, indent=1, ensure_ascii=False))
    sys.exit(1 if dup_groups else 0)

if __name__ == '__main__':
    main()
