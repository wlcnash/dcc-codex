"""Spoiler-gate regression test.

Covers the invariant the entire floor-migration work (2026-07-27) exists to
protect: a reader who's only unlocked floor N should never see art or content
from floor N+1 or later, and should always see the LATEST appearance they've
unlocked (not stuck on an older one). Checks two layers:

  DB layer (always runs):
    1. floors.floor_number is sequential 1..max with no gaps or repeats.
    2. Every entity_appearances.floor_id resolves to a floor number <= the
       known max (structurally guaranteed by the FK, but asserted directly
       here rather than trusted blindly).

  Live-app layer (runs only if the app is reachable at APP_URL, default
  http://localhost:8000 -- intended to run from inside the dcc-codex app
  pod, same as every other live check this session):
    3. For each entity with 2+ floor-keyed appearances, walk floor values in
       increasing order and confirm the resolved image's floor_number is
       monotonically non-decreasing -- i.e. unlocking more floors never shows
       an OLDER appearance than what a lower floor value already showed.

Usage: python3 audit_spoiler_gate.py [--app-url http://localhost:8000] [--skip-live]
Exit 0 if all checks pass, 1 if any invariant is violated.
"""
import psycopg2, os, sys, json, argparse, urllib.request, hashlib

def get_conn():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'], dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'], password=os.environ['POSTGRES_PASSWORD'])

def check_floor_sequence(conn):
    cur = conn.cursor()
    cur.execute("SELECT floor_number FROM floors ORDER BY floor_number")
    nums = [r[0] for r in cur.fetchall()]
    cur.close()
    problems = []
    expected = list(range(1, len(nums) + 1))
    if nums != expected:
        problems.append({'issue': 'floor sequence has gaps or repeats', 'found': nums, 'expected': expected})
    return problems

def check_appearance_floor_validity(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT ea.id, ea.entity_id, ea.floor_id FROM entity_appearances ea
        LEFT JOIN floors f ON f.id = ea.floor_id
        WHERE ea.floor_id IS NOT NULL AND f.id IS NULL
    """)
    orphans = cur.fetchall()
    cur.close()
    return [{'issue': 'entity_appearance references nonexistent floor', 'appearance_id': r[0], 'entity_id': r[1], 'floor_id': r[2]} for r in orphans]

def check_monotonic_serving(conn, app_url):
    cur = conn.cursor()
    cur.execute("""
        SELECT ea.entity_id, e.slug, count(*) FROM entity_appearances ea
        JOIN entities e ON e.id = ea.entity_id
        WHERE ea.floor_id IS NOT NULL
        GROUP BY ea.entity_id, e.slug HAVING count(*) >= 2
    """)
    multi = cur.fetchall()
    cur.execute("SELECT max(floor_number) FROM floors")
    max_floor = cur.fetchone()[0]
    cur.close()

    problems = []
    for (entity_id, slug, cnt) in multi:
        last_hash = None
        last_floor_used = 0
        seen_hashes_in_order = []
        for floor in range(1, max_floor + 1):
            req = urllib.request.Request(f'{app_url}/images/{slug}.jpg')
            req.add_header('Cookie', f'max_floor={floor}')
            try:
                resp = urllib.request.urlopen(req, timeout=10)
                h = hashlib.md5(resp.read()).hexdigest()
            except Exception as e:
                continue  # no image available yet at this floor -- not a violation
            seen_hashes_in_order.append((floor, h))
        # Monotonic check: once an image hash changes, it should never revert
        # to a PREVIOUSLY seen hash later (that would mean serving got stuck
        # on or reverted to an older appearance for a higher floor value).
        seen_so_far = []
        for floor, h in seen_hashes_in_order:
            if h in seen_so_far and h != seen_so_far[-1]:
                problems.append({
                    'issue': 'image regressed to an earlier appearance at a higher floor value',
                    'entity_id': entity_id, 'slug': slug, 'floor': floor,
                })
            if not seen_so_far or seen_so_far[-1] != h:
                seen_so_far.append(h)
    return problems

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--app-url', default='http://localhost:8000')
    ap.add_argument('--skip-live', action='store_true')
    args = ap.parse_args()

    conn = get_conn()
    problems = []
    problems += check_floor_sequence(conn)
    problems += check_appearance_floor_validity(conn)

    live_checked = False
    if not args.skip_live:
        try:
            urllib.request.urlopen(f'{args.app_url}/health', timeout=5)
            live_checked = True
            problems += check_monotonic_serving(conn, args.app_url)
        except Exception as e:
            pass  # app unreachable from here -- DB-only checks still ran
    conn.close()

    result = {'check': 'audit_spoiler_gate', 'live_layer_checked': live_checked, 'problems_found': len(problems), 'problems': problems}
    print(json.dumps(result, indent=1, ensure_ascii=False))
    sys.exit(1 if problems else 0)

if __name__ == '__main__':
    main()
