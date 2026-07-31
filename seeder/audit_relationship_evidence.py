"""Verbatim evidence rule check for entity_relationships.

2026-07-30: every entity_relationships row must be traceable back to a real, exact quote
from the entity_a side's own passages -- see relationships.py's module docstring for the
full rationale (entity_a is always the "subject" that was scanned; entity_b is the "object"
of the relationship and never scanned itself, so evidence always lives on entity_a's side).
This is the standing, rerunnable enforcement mechanism for that rule: for every row, confirms
(1) both entity_a_id and entity_b_id still exist, (2) entity_a_id and entity_b_id are not the
same entity, (3) relation_type is non-empty lowercase snake_case, and (4) evidence is an
exact substring of at least one of entity_a's own passages. Anything that fails this either
predates the rule (shouldn't happen -- relationships.py enforces this at insert time) or
points at a downstream data change (e.g. a passage was edited/removed after the relationship
was extracted) that broke a previously-valid citation.

Read-only, safe to run any time.
Usage: python3 audit_relationship_evidence.py
Exit 0 if every row passes, 1 if anything is flagged.
"""
import psycopg2, os, sys, re, json


def get_conn():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'], dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'], password=os.environ['POSTGRES_PASSWORD'])


def main():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, entity_a_id, entity_b_id, relation_type, evidence FROM entity_relationships")
    rows = cur.fetchall()

    entity_ids = set()
    for _, a, b, _, _ in rows:
        entity_ids.add(a); entity_ids.add(b)
    cur.execute("SELECT id, name FROM entities WHERE id = ANY(%s)", (list(entity_ids),))
    names = dict(cur.fetchall())

    flagged_missing_entity = []
    flagged_self_relation = []
    flagged_bad_relation_type = []
    flagged_no_evidence = []
    flagged_evidence_not_verbatim = []

    relation_type_re = re.compile(r"^[a-z][a-z_]*[a-z]$|^[a-z]$")
    passage_cache = {}

    for rel_id, a, b, relation_type, evidence in rows:
        if a not in names or b not in names:
            flagged_missing_entity.append({'id': rel_id, 'entity_a_id': a, 'entity_b_id': b})
            continue
        if a == b:
            flagged_self_relation.append({'id': rel_id, 'entity_id': a, 'name': names[a]})
            continue
        if not relation_type or not relation_type_re.match(relation_type):
            flagged_bad_relation_type.append({'id': rel_id, 'relation_type': relation_type})
            continue
        if not evidence or not evidence.strip():
            flagged_no_evidence.append({'id': rel_id, 'entity_a': names[a], 'entity_b': names[b], 'relation_type': relation_type})
            continue

        if a not in passage_cache:
            cur.execute("SELECT passage_text FROM passages WHERE entity_id = %s", (a,))
            passage_cache[a] = [r[0] for r in cur.fetchall() if r[0]]

        if not any(evidence.strip() in text for text in passage_cache[a]):
            flagged_evidence_not_verbatim.append({
                'id': rel_id, 'entity_a': names[a], 'entity_b': names[b],
                'relation_type': relation_type, 'evidence_preview': evidence[:150],
            })

    cur.close(); conn.close()

    flagged_count = (len(flagged_missing_entity) + len(flagged_self_relation)
                      + len(flagged_bad_relation_type) + len(flagged_no_evidence)
                      + len(flagged_evidence_not_verbatim))

    result = {
        'check': 'audit_relationship_evidence',
        'total_relationships': len(rows),
        'flagged_missing_entity': flagged_missing_entity,
        'flagged_self_relation': flagged_self_relation,
        'flagged_bad_relation_type': flagged_bad_relation_type,
        'flagged_no_evidence': flagged_no_evidence,
        'flagged_evidence_not_verbatim': flagged_evidence_not_verbatim,
        'flagged_count': flagged_count,
    }
    print(json.dumps(result, indent=1, ensure_ascii=False))
    sys.exit(1 if flagged_count else 0)


if __name__ == '__main__':
    main()
