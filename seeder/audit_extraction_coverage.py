"""Extraction coverage audit.

Generalizes the manual gap-finding done 2026-07-27 (Donut Book 7 at chapter
1/87, Carl Book 2 at 7/25, etc.) into a rerunnable check.

Heuristic: an entity that (a) has passages in a LATER book, meaning they're
still part of the story past an earlier book, and (b) already has a
substantial number of physical passages WITHIN that earlier book (proving
they were an active presence there, not a one-off background mention) --
if their last physical passage in that earlier book stops well short of the
book's last chapter, that's a likely real extraction gap (the original bug:
Carl ch36-47 of Book 1 had zero physical passages despite clearly-relevant
content). Deliberately excludes an entity's LAST book of appearance (dying or
exiting there is normal) and excludes thin/background mentions (min_passages
filter) -- first version of this script had no min_passages filter and
flagged 181 "gaps", almost all just minor characters mentioned once early and
again many books later, nothing like a real extraction miss. Caught in
testing before this script was trusted.

Usage: python3 audit_extraction_coverage.py [--threshold 0.85] [--min-passages 5]
Exit 0 if nothing below threshold, 1 if gaps found.
"""
import psycopg2, os, sys, json, argparse

def get_conn():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'], dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'], password=os.environ['POSTGRES_PASSWORD'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.85)
    ap.add_argument('--min-passages', type=int, default=5)
    args = ap.parse_args()

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id, (SELECT count(*) FROM chapters c WHERE c.book_id=b.id) FROM books b ORDER BY id")
    book_chapter_counts = dict(cur.fetchall())

    cur.execute("""
        SELECT p.entity_id, c.book_id, max(c.chapter_number), count(*)
        FROM passages p JOIN chapters c ON c.id = p.chapter_id
        WHERE p.passage_type = 'physical'
        GROUP BY p.entity_id, c.book_id
    """)
    rows = cur.fetchall()

    entity_names = {}
    cur.execute("SELECT id, name FROM entities")
    for (eid, name) in cur.fetchall():
        entity_names[eid] = name
    cur.close(); conn.close()

    by_entity = {}
    for (entity_id, book_id, last_chapter, pcount) in rows:
        by_entity.setdefault(entity_id, {})[book_id] = (last_chapter, pcount)

    gaps = []
    for entity_id, book_map in by_entity.items():
        books_present = sorted(book_map.keys())
        if len(books_present) < 2:
            continue
        last_book = books_present[-1]
        for b in books_present:
            if b == last_book:
                continue
            total = book_chapter_counts.get(b)
            covered, pcount = book_map[b]
            if not total or pcount < args.min_passages:
                continue
            ratio = covered / total
            if ratio < args.threshold:
                gaps.append({
                    'entity_id': entity_id, 'entity_name': entity_names.get(entity_id),
                    'book_id': b, 'last_covered_chapter': covered, 'total_chapters': total,
                    'coverage_ratio': round(ratio, 3), 'passages_in_book': pcount,
                })

    gaps.sort(key=lambda g: g['coverage_ratio'])

    result = {'check': 'audit_extraction_coverage', 'threshold': args.threshold,
              'min_passages': args.min_passages, 'gaps_found': len(gaps), 'gaps': gaps}
    print(json.dumps(result, indent=1, ensure_ascii=False))
    sys.exit(1 if gaps else 0)

if __name__ == '__main__':
    main()
