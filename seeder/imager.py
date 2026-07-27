import base64, io, json, logging, time
import psycopg2, boto3
from botocore.config import Config
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
IMAGE_BUCKET = "dcc-codex"
RATE_LIMIT_SECONDS = 3.0
TRANSIENT_RECENT_LIMIT = 3

PROMPT_BUILDER_SYSTEM = (
    "You are creating image generation prompts for a Dungeon Crawler Carl compendium. "
    "Base prompts ONLY on the author exact descriptions provided. "
    "Do not add details not present in the source text. Keep the dungeon aesthetic: gritty, alien, dangerous. "
    "The passages below represent the character's CUMULATIVE appearance as of the END of the specified book: "
    "DURABLE TRAITS are established, ongoing features (build, hair, gear kept over time) that still apply. "
    "CURRENT STATE are the most recent transient details (wounds, dirt, temporary effects) at that point in the story. "
    "Combine both into one coherent, internally-consistent appearance. If CURRENT STATE contradicts an earlier "
    "DURABLE detail (e.g. a wound has healed), prefer CURRENT STATE. "
    "Return ONLY the image prompt text, nothing else."
)

PROMPT_BUILDER_USER = (
    'Based on these exact passages describing "{entity_name}" ({entity_type}) as of the end of '
    '"{book_title}" (Book {book_number}):\n\n'
    "DURABLE TRAITS (established, still apply):\n{durable_text}\n\n"
    "CURRENT STATE (most recent, near end of this book):\n{transient_text}\n\n"
    "Create a single detailed image prompt capturing his cumulative appearance as of the end of this book. "
    "Include: visual style (dark fantasy illustration), dungeon torchlight lighting, "
    "and all specific details mentioned: colors, sizes, materials, anatomy, gear."
)


def build_image_prompt(entity_name, entity_type, durable_passages, transient_passages, book_title, book_number, client):
    sep = "\n\n---\n\n"
    durable_text = sep.join('"' + p + '"' for p in durable_passages) if durable_passages else "(none established yet)"
    transient_text = sep.join('"' + p + '"' for p in transient_passages) if transient_passages else "(no notable current condition)"
    prompt = PROMPT_BUILDER_SYSTEM + "\n\n" + PROMPT_BUILDER_USER.format(
        entity_name=entity_name, entity_type=entity_type, durable_text=durable_text,
        transient_text=transient_text, book_title=book_title, book_number=book_number,
    )
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text.strip()


def generate_image(prompt, client):
    try:
        interaction = client.interactions.create(
            model="gemini-3.1-flash-image",
            input=prompt,
            response_format={"type": "image", "mime_type": "image/jpeg", "aspect_ratio": "1:1", "image_size": "1K"},
        )
        time.sleep(RATE_LIMIT_SECONDS)
        if interaction.output_image and interaction.output_image.data:
            return base64.b64decode(interaction.output_image.data)
        return None
    except Exception as e:
        logger.error("Nano Banana failed: %s", e)
        return None


def upload_to_minio(minio_client, slug, book_id, image_bytes):
    key = "entities/{}/book_{}.jpg".format(slug, book_id)
    minio_client.put_object(Bucket=IMAGE_BUCKET, Key=key, Body=io.BytesIO(image_bytes), ContentType="image/jpeg")
    return "/images/{}.jpg".format(slug)


def ensure_bucket(minio_client):
    try:
        minio_client.head_bucket(Bucket=IMAGE_BUCKET)
    except Exception:
        minio_client.create_bucket(Bucket=IMAGE_BUCKET)


def log_run_start(conn, step, meta=None):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO pipeline_runs (step, status, meta) VALUES (%s, 'running', %s) RETURNING id",
        (step, json.dumps(meta) if meta else None)
    )
    run_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return run_id


def log_run_finish(conn, run_id, processed, failed, error=None):
    status = "error" if error else "success"
    cur = conn.cursor()
    cur.execute(
        "UPDATE pipeline_runs SET status=%s, items_processed=%s, items_failed=%s, error_detail=%s, finished_at=NOW() WHERE id=%s",
        (status, processed, failed, error, run_id)
    )
    conn.commit()
    cur.close()


def _fetch_passages(conn, entity_id, book_id, book_number):
    """Return (durable_passages, transient_passages) for an entity as of the end of book_number.

    Durable: ALL classified-durable physical passages from book 1 through book_number (cumulative,
    uncapped -- this is finite, static source material, no reason to truncate it).
    Transient: the most recent classified-transient physical passages from THIS book only, representing
    the character's current condition near the end of the book (capped small on purpose -- this is meant
    to be a snapshot of 'right now', not an accumulation of every scrape and bruise across the whole book).
    If this entity's physical passages haven't been classified yet (is_durable IS NULL for all of them),
    fall back to the old behavior: every physical passage in this book, uncapped.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM passages WHERE entity_id=%s AND passage_type='physical' AND is_durable IS NOT NULL",
        (entity_id,),
    )
    is_classified = cur.fetchone()[0] > 0
    cur.close()

    if not is_classified:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.passage_text FROM passages p
            JOIN chapters c ON c.id = p.chapter_id
            WHERE p.entity_id = %s AND c.book_id = %s AND p.passage_type = 'physical'
            ORDER BY p.id
        """, (entity_id, book_id))
        all_text = [row[0] for row in cur.fetchall()]
        cur.close()
        return all_text, []

    cur = conn.cursor()
    cur.execute("""
        SELECT p.passage_text FROM passages p
        JOIN chapters c ON c.id = p.chapter_id
        JOIN books b2 ON b2.id = c.book_id
        WHERE p.entity_id = %s AND p.passage_type = 'physical' AND p.is_durable = TRUE
          AND b2.book_number <= %s
        ORDER BY b2.book_number, c.chapter_number, p.id
    """, (entity_id, book_number))
    durable = [row[0] for row in cur.fetchall()]
    cur.close()

    cur = conn.cursor()
    cur.execute("""
        SELECT p.passage_text FROM passages p
        JOIN chapters c ON c.id = p.chapter_id
        WHERE p.entity_id = %s AND c.book_id = %s AND p.passage_type = 'physical' AND p.is_durable = FALSE
        ORDER BY c.chapter_number DESC, p.id DESC
        LIMIT %s
    """, (entity_id, book_id, TRANSIENT_RECENT_LIMIT))
    transient = [row[0] for row in cur.fetchall()]
    transient.reverse()
    cur.close()

    return durable, transient


def run_imager(conn, gemini_api_key, minio_endpoint, minio_access_key, minio_secret_key, batch_size=20):
    """
    Generate per-book images for entities.
    One image per (entity, book) pair where physical passages exist.
    Stores in entity_appearances; also keeps entities.image_url updated for backward compat.
    """
    run_id = log_run_start(conn, "images", {"batch_size": batch_size})
    client = genai.Client(api_key=gemini_api_key)
    minio_client = boto3.client(
        "s3", endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key, aws_secret_access_key=minio_secret_key,
        config=Config(signature_version="s3v4")
    )
    ensure_bucket(minio_client)

    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (e.id, b.id)
               e.id, e.name, e.slug, e.entity_type::text, e.is_major,
               b.id AS book_id, b.book_number, b.title AS book_title
        FROM entities e
        JOIN passages p  ON p.entity_id  = e.id
        JOIN chapters c  ON c.id         = p.chapter_id
        JOIN books b     ON b.id         = c.book_id
        WHERE p.passage_type = 'physical'
          AND NOT EXISTS (
              SELECT 1 FROM entity_appearances ea
              WHERE ea.entity_id = e.id AND ea.book_id = b.id
          )
        ORDER BY e.id, b.id, e.is_major DESC
        LIMIT %s
    """, (batch_size,))
    targets = cur.fetchall()
    cur.close()

    if not targets:
        logger.info("No (entity, book) pairs need images.")
        log_run_finish(conn, run_id, 0, 0)
        return 0

    logger.info("Generating per-book images for %d (entity, book) pairs...", len(targets))
    generated = 0
    failed = 0

    for entity_id, name, slug, entity_type, is_major, book_id, book_number, book_title in targets:
        durable, transient = _fetch_passages(conn, entity_id, book_id, book_number)

        if not durable and not transient:
            continue

        logger.info("  [Book %d] %s (%d durable, %d current-state passages)", book_number, name, len(durable), len(transient))

        try:
            image_prompt = build_image_prompt(name, entity_type, durable, transient, book_title, book_number, client)
        except Exception as e:
            logger.warning("  Prompt failed for %s book %d: %s", name, book_number, e)
            failed += 1
            continue

        image_bytes = generate_image(image_prompt, client)
        if image_bytes is None:
            logger.warning("  No image for %s book %d", name, book_number)
            failed += 1
            continue

        try:
            image_url = upload_to_minio(minio_client, slug, book_id, image_bytes)
        except Exception as e:
            logger.warning("  MinIO upload failed for %s book %d: %s", name, book_number, e)
            failed += 1
            continue

        all_passages = durable + transient
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO entity_appearances (entity_id, book_id, image_url, image_prompt, image_source_passages)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (entity_id, book_id) DO UPDATE SET
                image_url = EXCLUDED.image_url,
                image_prompt = EXCLUDED.image_prompt,
                image_source_passages = EXCLUDED.image_source_passages
        """, (entity_id, book_id, image_url, image_prompt, all_passages))
        cur.execute("""
            UPDATE entities SET image_url=%s, image_prompt=%s, image_source_passages=%s
            WHERE id=%s AND (image_url IS NULL OR NOT EXISTS (
                SELECT 1 FROM entity_appearances ea2
                JOIN books b2 ON b2.id=ea2.book_id
                WHERE ea2.entity_id=%s AND b2.book_number>%s
            ))
        """, (image_url, image_prompt, all_passages, entity_id, entity_id, book_number))
        conn.commit()
        cur.close()
        generated += 1
        logger.info("  OK %s (book %d) -> %s", name, book_number, image_url)

    log_run_finish(conn, run_id, generated, failed)
    logger.info("Done. %d generated, %d failed.", generated, failed)
    return generated
