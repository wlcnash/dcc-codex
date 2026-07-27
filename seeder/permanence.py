"""
DCC Codex — physical-passage permanence classifier.

Splits 'physical' passages into DURABLE (body build, hair, skin, tattoos, scars,
long-kept gear — stays true going forward) vs TRANSIENT (fresh wounds, blood, dirt,
healing injuries, one-scene outfit details — true only in the moment). Lets the
imager build a cumulative, end-of-book-accurate appearance instead of randomly
blending permanent traits with one-off injuries.
"""

import logging
import time
import re

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
RATE_LIMIT_SECONDS = 1.0
CLASSIFY_BATCH = 20

CLASSIFY_SYSTEM = (
    "You are classifying physical-description passages about a character from a novel, "
    "for the purpose of building an accurate cumulative physical profile.\n\n"
    "For each numbered passage, decide:\n"
    "DURABLE = a trait that stays true going forward once established: body build, height, "
    "weight, hair color/style, skin tone, permanent tattoos, scars that don't heal, gear or "
    "clothing the character keeps wearing/carrying for an extended period.\n"
    "TRANSIENT = a state true only in that moment: fresh wounds, blood, dirt, an injury that "
    "will heal, a temporary debuff or magical effect, a passing outfit detail specific to one "
    "scene, damage/gore from a fight that just happened.\n\n"
    "Respond with ONLY a JSON array of \"DURABLE\" or \"TRANSIENT\" strings, one per passage, "
    "in the same order given. No other text."
)


def run_migrate(conn) -> None:
    cur = conn.cursor()
    cur.execute("ALTER TABLE passages ADD COLUMN IF NOT EXISTS is_durable BOOLEAN")
    conn.commit()
    cur.close()
    logger.info("Migration: passages.is_durable column ensured.")


def _classify_batch(passages_batch, client):
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(passages_batch))
    prompt = CLASSIFY_SYSTEM + "\n\nPassages:\n" + numbered
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    text = response.text.strip()
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    import json
    labels = json.loads(text)
    return [str(l).strip().upper() == "DURABLE" for l in labels]


def run_classify(conn, gemini_api_key: str, entity_id=None, batch_size: int = 999999) -> int:
    """Classify unclassified 'physical' passages as durable/transient. Returns count classified."""
    run_migrate(conn)
    client = genai.Client(api_key=gemini_api_key)

    cur = conn.cursor()
    if entity_id is not None:
        cur.execute(
            "SELECT id, passage_text FROM passages WHERE entity_id=%s AND passage_type='physical' AND is_durable IS NULL ORDER BY id LIMIT %s",
            (entity_id, batch_size),
        )
    else:
        cur.execute(
            "SELECT id, passage_text FROM passages WHERE passage_type='physical' AND is_durable IS NULL ORDER BY id LIMIT %s",
            (batch_size,),
        )
    rows = cur.fetchall()
    cur.close()

    if not rows:
        logger.info("No unclassified physical passages found.")
        return 0

    logger.info("Classifying %d physical passages as durable/transient...", len(rows))
    classified = 0
    for i in range(0, len(rows), CLASSIFY_BATCH):
        chunk = rows[i:i + CLASSIFY_BATCH]
        ids = [r[0] for r in chunk]
        texts = [r[1] for r in chunk]
        try:
            labels = _classify_batch(texts, client)
            if len(labels) != len(chunk):
                logger.warning("Label count mismatch (%d vs %d), skipping chunk", len(labels), len(chunk))
                continue
        except Exception as e:
            logger.warning("Classification batch failed: %s", e)
            continue
        cur = conn.cursor()
        for pid, is_durable in zip(ids, labels):
            cur.execute("UPDATE passages SET is_durable=%s WHERE id=%s", (is_durable, pid))
            classified += 1
        conn.commit()
        cur.close()
        time.sleep(RATE_LIMIT_SECONDS)

    logger.info("Classified %d passages.", classified)
    return classified
