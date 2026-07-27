import base64, io, json, logging, time
import psycopg2, boto3
from botocore.config import Config
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
IMAGE_BUCKET = "dcc-codex"
RATE_LIMIT_SECONDS = 3.0

PROMPT_BUILDER_SYSTEM = (
    "You are creating image generation prompts for a Dungeon Crawler Carl compendium. "
    "Base prompts ONLY on the author exact descriptions from the specified book. "
    "Do not add details not present in the source text. Keep the dungeon aesthetic: gritty, alien, dangerous. "
    "Return ONLY the image prompt text, nothing else."
)

PROMPT_BUILDER_USER = (
    'Based on these exact passages from "{book_title}" (Book {book_number}), '
    "create an image generation prompt for: {entity_name} ({entity_type})\n\n"
    "PASSAGES FROM THIS BOOK:\n{passages}\n\n"
    "Create a single detailed image prompt capturing the appearance described in THIS book specifically. "
    "The entity may look different in other books -- use only what is above. "
    "Include: visual style (dark fantasy illustration), dungeon torchlight lighting, "
    "and all specific details mentioned: colors, sizes, materials, anatomy. Max 400 words."
)


def build_image_prompt(entity_name, entity_type, passages, book_title, book_number, client):
    sep = "\n\n---\n\n"
    passages_text = sep.join('"' + p + '"' for p in passages[:10])
    prompt = PROMPT_BUILDER_SYSTEM + "\n\n" + PROMPT_BUILDER_USER.format(
        entity_name=entity_name, entity_type=entity_type, passages=passages_text,
        book_title=book_title, book_number=book_number,
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
        cur = conn.cursor()
        cur.execute("""
            SELECT p.passage_text FROM passages p
            JOIN chapters c ON c.id = p.chapter_id
            WHERE p.entity_id = %s AND c.book_id = %s AND p.passage_type = 'physical'
            ORDER BY p.id
        """, (entity_id, book_id))
        passages = [row[0] for row in cur.fetchall()]
        cur.close()

        if not passages:
            continue

        logger.info("  [Book %d] %s (%d passages)", book_number, name, len(passages))

        try:
            image_prompt = build_image_prompt(name, entity_type, passages, book_title, book_number, client)
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

        cur = conn.cursor()
        cur.execute("""
            INSERT INTO entity_appearances (entity_id, book_id, image_url, image_prompt, image_source_passages)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (entity_id, book_id) DO UPDATE SET
                image_url = EXCLUDED.image_url,
                image_prompt = EXCLUDED.image_prompt,
                image_source_passages = EXCLUDED.image_source_passages
        """, (entity_id, book_id, image_url, image_prompt, passages[:5]))
        cur.execute("""
            UPDATE entities SET image_url=%s, image_prompt=%s, image_source_passages=%s
            WHERE id=%s AND (image_url IS NULL OR NOT EXISTS (
                SELECT 1 FROM entity_appearances ea2
                JOIN books b2 ON b2.id=ea2.book_id
                WHERE ea2.entity_id=%s AND b2.book_number>%s
            ))
        """, (image_url, image_prompt, passages[:5], entity_id, entity_id, book_number))
        conn.commit()
        cur.close()
        generated += 1
        logger.info("  OK %s (book %d) -> %s", name, book_number, image_url)

    log_run_finish(conn, run_id, generated, failed)
    logger.info("Done. %d generated, %d failed.", generated, failed)
    return generated
