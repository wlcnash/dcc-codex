"""Entity_type sanity check.

Currently checks entity_type='floor' rows specifically (the class of bug found
and fixed 2026-07-27: misclassified rows like "A Parade of Horribles" and
"Over City" that were tagged floor but aren't, plus rows lacking floor-number
evidence entirely).

Flags two problem classes:
  1. A 'floor' entity with NO numeric/ordinal floor reference in its own name
     or aliases (checked first), and none in its passages either (fallback --
     passages are noisier since ordinal words show up in unrelated sentences,
     so this is only used to avoid missing a real name like "The Bubbles"
     when name/aliases alone come up empty, same as the manual process used
     2026-07-27). Still flagged as WEAK evidence for a human to check.
  2. Two or more DIFFERENT 'floor' entities whose NAME/ALIASES resolve to the
     SAME floor number -- a real duplicate audit_duplicates.py's exact-string
     match wouldn't catch (the same class of bug as the old AI/System AI
     cluster, applied to floors). Deliberately does NOT use passage text for
     this check -- passages mention other floors in dialogue constantly, which
     would flood this with false positives (caught in testing 2026-07-27:
     first version used passage text here and produced garbage collisions).

Read-only, safe to run any time.
Usage: python3 audit_entity_types.py
Exit 0 if clean, 1 if anything flagged.
"""
import psycopg2, os, re, sys, json

WORDS = {'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'sixth':6,'seventh':7,
         'eighth':8,'ninth':9,'tenth':10,'eleventh':11,'twelfth':12,'thirteenth':13,
         'fourteenth':14,'fifteenth':15,'sixteenth':16,'seventeenth':17,'eighteenth':18,
         'nineteenth':19,'twentieth':20}

def get_conn():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'], dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'], password=os.environ['POSTGRES_PASSWORD'])

def extract_numbers(text):
    t = text.lower()
    nums = set()
    for m in re.finditer(r'(\d+)\s*(?:st|nd|rd|th)\b', t):
        nums.add(int(m.group(1)))
    for m in re.finditer(r'\b(?:floor|level)\s+(\d+)\b', t):
        nums.add(int(m.group(1)))
    for w, n in WORDS.items():
        if re.search(r'\b' + w + r'\b', t):
            nums.add(n)
    return nums

def main():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, aliases FROM entities WHERE entity_type='floor' ORDER BY id")
    floors = cur.fetchall()

    no_number = []
    weak_evidence = []
    number_map = {}  # number -> set of (id, name), from name/aliases ONLY
    for (id_, name, aliases) in floors:
        name_alias_text = ' | '.join([name] + (aliases or []))
        name_nums = extract_numbers(name_alias_text)
        if name_nums:
            for n in name_nums:
                number_map.setdefault(n, set()).add((id_, name))
            continue
        # fall back to passages only when name/aliases gave nothing
        cur.execute("SELECT passage_text FROM passages WHERE entity_id = %s", (id_,))
        passage_text = ' | '.join(r[0] for r in cur.fetchall())
        passage_nums = extract_numbers(passage_text)
        if passage_nums:
            weak_evidence.append({'id': id_, 'name': name, 'note': 'no number in name/aliases; passages mention floor numbers but not confirmed as THIS floor\'s identity - verify by reading before trusting', 'candidate_numbers': sorted(passage_nums)})
        else:
            no_number.append({'id': id_, 'name': name})

    cur.close(); conn.close()

    collisions = {n: sorted(str(x) for x in ids) for n, ids in number_map.items() if len(ids) > 1}

    result = {
        'check': 'audit_entity_types (floor)',
        'no_number_evidence_anywhere': no_number,
        'weak_evidence_passages_only': weak_evidence,
        'floor_number_collisions_name_alias_only': collisions,
    }
    print(json.dumps(result, indent=1, ensure_ascii=False))
    sys.exit(1 if (no_number or weak_evidence or collisions) else 0)

if __name__ == '__main__':
    main()
