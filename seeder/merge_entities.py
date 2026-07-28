"""Entity merge utility -- the formalized version of the ad hoc merge scripts
written repeatedly during the 2026-07-27 dedup sessions (AI cluster, floor
cleanup, Katia, Donut, and the 64-group casing pass before that).

DRY-RUN IS THE DEFAULT. Nothing is written unless --confirm is passed
explicitly. This is deliberate -- every prior merge this session was run as a
hand-written rollback-then-commit script; this utility bakes that discipline
in so it can't be skipped by accident in a future session.

Every merge is logged to `entity_merge_log` BEFORE the old rows are deleted
-- previously, merges just DELETEd the losing rows with no permanent record,
making any bad merge unrecoverable and unauditable. Now every merge leaves a
row: old id/name/type/aliases, which canonical it merged into, why, and when.

What it does, in order, only when --confirm is passed:
  1. Merges aliases (canonical's own + every other row's name + aliases,
     case-insensitively deduped) onto the canonical entity.
  2. Writes one entity_merge_log row per merged-away entity BEFORE deleting.
  3. Repoints passages.entity_id, entity_appearances.entity_id (guarding the
     (entity_id, floor_id) unique constraint), and entity_relationships'
     entity_a_id/entity_b_id (guarding the unique constraint and dropping any
     resulting self-relationships) from the old ids to the canonical id.
  4. Deletes the old entity rows (safe now that nothing references them).

Usage:
  python3 merge_entities.py --canonical 406 --others 563 1648 --reason "..."
      (dry run -- prints the plan, writes nothing)
  python3 merge_entities.py --canonical 406 --others 563 1648 --reason "..." --confirm
      (actually performs the merge)
  python3 merge_entities.py --recent 20
      (show the last N entity_merge_log rows, for auditing)
"""
import psycopg2, os, sys, json, argparse

def get_conn():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'], dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'], password=os.environ['POSTGRES_PASSWORD'])

def fetch(cur, ids):
    cur.execute("SELECT id, name, entity_type, aliases FROM entities WHERE id = ANY(%s)", (ids,))
    return {r[0]: {'name': r[1], 'entity_type': r[2], 'aliases': r[3] or []} for r in cur.fetchall()}

def build_plan(rows, canonical_id, other_ids):
    canonical = rows[canonical_id]
    alias_set = list(canonical['aliases'])
    seen = set(a.lower() for a in alias_set) | {canonical['name'].lower()}
    for oid in other_ids:
        o = rows[oid]
        if o['name'].lower() not in seen:
            alias_set.append(o['name']); seen.add(o['name'].lower())
        for a in o['aliases']:
            if a.lower() not in seen:
                alias_set.append(a); seen.add(a.lower())
    return alias_set

def do_merge(conn, canonical_id, other_ids, reason, confirm):
    cur = conn.cursor()
    rows = fetch(cur, [canonical_id] + other_ids)
    missing = [i for i in [canonical_id] + other_ids if i not in rows]
    if missing:
        print(json.dumps({'error': 'entity id(s) not found', 'missing': missing}))
        sys.exit(2)

    canonical = rows[canonical_id]
    alias_set = build_plan(rows, canonical_id, other_ids)

    plan = {
        'mode': 'CONFIRMED - WRITING' if confirm else 'DRY RUN - no changes made',
        'canonical': {'id': canonical_id, 'name': canonical['name'], 'entity_type': canonical['entity_type']},
        'merging_away': [{'id': i, 'name': rows[i]['name'], 'entity_type': rows[i]['entity_type']} for i in other_ids],
        'reason': reason,
        'final_alias_count': len(alias_set),
    }

    if not confirm:
        print(json.dumps(plan, indent=1, ensure_ascii=False))
        cur.close()
        return

    for oid in other_ids:
        o = rows[oid]
        cur.execute("""
            INSERT INTO entity_merge_log (old_entity_id, old_name, old_entity_type, old_aliases, canonical_entity_id, canonical_name_at_merge, reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (oid, o['name'], o['entity_type'], o['aliases'], canonical_id, canonical['name'], reason))

    cur.execute("UPDATE entities SET aliases = %s WHERE id = %s", (alias_set, canonical_id))
    cur.execute("UPDATE passages SET entity_id = %s WHERE entity_id = ANY(%s)", (canonical_id, other_ids))
    cur.execute("""UPDATE entity_appearances ea SET entity_id = %s WHERE ea.entity_id = ANY(%s)
        AND NOT EXISTS (SELECT 1 FROM entity_appearances ea2 WHERE ea2.entity_id=%s AND ea2.floor_id=ea.floor_id)""", (canonical_id, other_ids, canonical_id))
    cur.execute("DELETE FROM entity_appearances WHERE entity_id = ANY(%s)", (other_ids,))
    cur.execute("""UPDATE entity_relationships er SET entity_a_id = %s WHERE er.entity_a_id = ANY(%s) AND er.entity_b_id != %s
        AND NOT EXISTS (SELECT 1 FROM entity_relationships er2 WHERE er2.entity_a_id=%s AND er2.entity_b_id=er.entity_b_id AND er2.relation_type=er.relation_type)""", (canonical_id, other_ids, canonical_id, canonical_id))
    cur.execute("""UPDATE entity_relationships er SET entity_b_id = %s WHERE er.entity_b_id = ANY(%s) AND er.entity_a_id != %s
        AND NOT EXISTS (SELECT 1 FROM entity_relationships er2 WHERE er2.entity_b_id=%s AND er2.entity_a_id=er.entity_a_id AND er2.relation_type=er.relation_type)""", (canonical_id, other_ids, canonical_id, canonical_id))
    cur.execute("DELETE FROM entity_relationships WHERE entity_a_id = ANY(%s) OR entity_b_id = ANY(%s)", (other_ids, other_ids))
    cur.execute("DELETE FROM entities WHERE id = ANY(%s)", (other_ids,))

    conn.commit()
    print(json.dumps(plan, indent=1, ensure_ascii=False))
    cur.close()

def show_recent(conn, n):
    cur = conn.cursor()
    cur.execute("""
        SELECT old_entity_id, old_name, old_entity_type, canonical_entity_id, canonical_name_at_merge, reason, merged_at
        FROM entity_merge_log ORDER BY merged_at DESC LIMIT %s
    """, (n,))
    rows = cur.fetchall()
    cur.close()
    out = [{'old_entity_id': r[0], 'old_name': r[1], 'old_entity_type': r[2],
            'canonical_entity_id': r[3], 'canonical_name_at_merge': r[4],
            'reason': r[5], 'merged_at': r[6].isoformat()} for r in rows]
    print(json.dumps(out, indent=1, ensure_ascii=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical', type=int)
    ap.add_argument('--others', type=int, nargs='+')
    ap.add_argument('--reason', type=str)
    ap.add_argument('--confirm', action='store_true')
    ap.add_argument('--recent', type=int, help='show last N merge log entries instead of merging')
    args = ap.parse_args()

    conn = get_conn()
    if args.recent is not None:
        show_recent(conn, args.recent)
    else:
        if not args.canonical or not args.others or not args.reason:
            print(json.dumps({'error': '--canonical, --others, and --reason are all required for a merge (or use --recent N to just view the log)'}))
            sys.exit(2)
        do_merge(conn, args.canonical, args.others, args.reason, args.confirm)
    conn.close()

if __name__ == '__main__':
    main()
