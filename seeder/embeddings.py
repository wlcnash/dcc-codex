"""
DCC Codex -- content-based entity embeddings for dedup candidate generation.

This is deliberately NOT a permanent site feature. It exists to close a
specific, real blind spot in the name/alias-based entity resolution methods
(seeder/entity_resolution.py's trigram stage, plus the retroactive
alias-collision and manual-name-similarity dedup passes): every one of those
methods keys off the entity NAME. Two entities that are the same underlying
thing but were extracted under completely different names, with zero alias
overlap, are structurally invisible to name-based candidate generation --
no amount of better trigram tuning or LLM judgment helps if the candidate
pair is never even proposed.

This module embeds each entity's actual descriptive CONTENT (persona_text,
falling back to concatenated distinct passage text) and finds nearest
neighbors by cosine distance within the same entity_type, regardless of how
dissimilar the names are. This is the RAG piece Wes asked for -- kept live
as part of the seeding toolkit for the remainder of the build phase (not
torn down after one run), so it can be re-run periodically as new entities
get extracted, but it is still scoped narrowly to *this* problem, not wired
into any live user-facing query path.
"""

import logging
import time
from typing import Optional

from google import genai

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 1.0
EMBED_MODEL = "gemini-embedding-001"
MAX_CONTENT_CHARS = 4000


def run_migrate(conn) -> None:
    """Ensure content_embedding/embedding_source columns exist. Safe to run multiple times."""
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("ALTER TABLE entities ADD COLUMN IF NOT EXISTS content_embedding vector(3072)")
    cur.execute("ALTER TABLE entities ADD COLUMN IF NOT EXISTS embedding_source TEXT")
    conn.commit()
    cur.close()
    logger.info("Migration: content_embedding/embedding_source ensured, pgvector extension ensured.")


def _build_content(conn, entity_id: int) -> Optional[str]:
    """Prefer persona_text (already a clean, validated summary post-persona.py fix).
    Fall back to concatenated distinct passage text if no persona exists yet."""
    cur = conn.cursor()
    cur.execute("SELECT persona_text FROM entities WHERE id = %s", (entity_id,))
    row = cur.fetchone()
    persona_text = row[0] if row else None
    if persona_text and persona_text.strip():
        cur.close()
        return persona_text.strip()[:MAX_CONTENT_CHARS]

    cur.execute(
        "SELECT DISTINCT passage_text FROM passages WHERE entity_id = %s ORDER BY passage_text LIMIT 20",
        (entity_id,),
    )
    rows = cur.fetchall()
    cur.close()
    if not rows:
        return None
    joined = " ".join(r[0] for r in rows if r[0])
    return joined.strip()[:MAX_CONTENT_CHARS] if joined.strip() else None


def run_embeddings(conn, gemini_api_key: str, batch_size: int = 999999) -> int:
    """Generate content embeddings for entities that don't have one yet
    (or whose persona_text/passages changed since the last embedding -- callers
    that want to force a refresh should NULL content_embedding first, same
    pattern as persona.py)."""
    run_migrate(conn)
    client = genai.Client(api_key=gemini_api_key)

    cur = conn.cursor()
    cur.execute("""
        SELECT e.id FROM entities e
        WHERE e.content_embedding IS NULL
        ORDER BY e.id
        LIMIT %s
    """, (batch_size,))
    ids = [r[0] for r in cur.fetchall()]
    cur.close()

    logger.info(f"Generating embeddings for {len(ids)} entities...")
    count = 0
    skipped = 0

    for entity_id in ids:
        content = _build_content(conn, entity_id)
        if not content:
            skipped += 1
            continue
        try:
            resp = client.models.embed_content(model=EMBED_MODEL, contents=content)
            vec = resp.embeddings[0].values
            time.sleep(RATE_LIMIT_SECONDS)

            cur = conn.cursor()
            cur.execute(
                "UPDATE entities SET content_embedding = %s, embedding_source = %s WHERE id = %s",
                (str(vec), content[:200], entity_id),
            )
            conn.commit()
            cur.close()
            count += 1
            if count % 100 == 0:
                logger.info(f"  {count} embeddings generated...")
        except Exception as e:
            logger.warning(f"  Failed embedding for entity {entity_id}: {e}")
            time.sleep(RATE_LIMIT_SECONDS)
            continue

    logger.info(f"Embedding generation complete: {count} generated, {skipped} skipped (no content).")
    return count


def find_content_neighbors(conn, similarity_threshold: float = 0.90, limit_per_entity: int = 3):
    """Within each entity_type, find pairs whose content embeddings are highly
    similar (cosine similarity >= threshold) regardless of name similarity.
    Returns a list of (id_a, name_a, id_b, name_b, entity_type, similarity)
    tuples, deduped so each pair appears once (id_a < id_b)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT a.id, a.name, b.id, b.name, a.entity_type,
               1 - (a.content_embedding <=> b.content_embedding) AS cosine_sim
        FROM entities a
        JOIN entities b
          ON a.entity_type = b.entity_type
         AND a.id < b.id
         AND a.content_embedding IS NOT NULL
         AND b.content_embedding IS NOT NULL
        WHERE 1 - (a.content_embedding <=> b.content_embedding) >= %s
        ORDER BY cosine_sim DESC
    """, (similarity_threshold,))
    rows = cur.fetchall()
    cur.close()
    return rows
