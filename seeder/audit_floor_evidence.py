"""Floor-boundary drift check.

The 11 canonical `floors` table rows are treated as permanent, trusted ground
truth (per project convention -- never re-derived once validated). This script
guards against that trust ever silently becoming wrong: it re-verifies each
floor's evidence_quote still appears (whitespace-normalized) in its
start_chapter's raw_text. Should only ever fail if someone hand-edits the
floors table incorrectly, or a chapter's raw_text gets re-scraped/altered.

Whitespace is normalized before comparing because the scraped raw_text has
genuine mid-sentence line-break artifacts from the source ebook (e.g.
"Welcome,\n Crawler to the third floor." -- confirmed 2026-07-27 this is in
the real source, not a bug), while evidence_quote was recorded with clean
spacing. A naive substring match false-positives on this for floors 3/10/11
-- caught in testing before this script was trusted.

Floor 1 is exempt -- its evidence_quote is a synthetic note ("seeded, not
model-detected"), not a real quote, by design.

Read-only, safe to run any time.
Usage: python3 audit_floor_evidence.py
Exit 0 if all quotes verified, 1 if any mismatch found.
"""
import psycopg2, os, re, sys, json

def get_conn():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'], dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'], password=os.environ['POSTGRES_PASSWORD'])

def norm_ws(s):
    return re.sub(r'\s+', ' ', s).strip()

def main():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT f.floor_number, f.evidence_quote, c.raw_text, b.title, c.chapter_number
        FROM floors f
        JOIN chapters c ON c.id = f.start_chapter_id
        JOIN books b ON b.id = c.book_id
        ORDER BY f.floor_number
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()

    mismatches = []
    verified = []
    for (floor_number, evidence_quote, raw_text, book_title, chapter_number) in rows:
        if floor_number == 1:
            continue  # synthetic seed row, not a real quote
        if evidence_quote and norm_ws(evidence_quote) in norm_ws(raw_text):
            verified.append(floor_number)
        else:
            mismatches.append({
                'floor_number': floor_number, 'evidence_quote': evidence_quote,
                'book': book_title, 'chapter_number': chapter_number,
            })

    result = {
        'check': 'audit_floor_evidence',
        'verified_floor_numbers': verified,
        'mismatches': mismatches,
    }
    print(json.dumps(result, indent=1, ensure_ascii=False))
    sys.exit(1 if mismatches else 0)

if __name__ == '__main__':
    main()
