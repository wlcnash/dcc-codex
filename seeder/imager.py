"""
Gemini Nano Banana image generator for DCC Codex.

For each entity with physical description passages but no image,
builds a detailed prompt from those passages and calls Nano Banana 2
(gemini-3.1-flash-image) via the Interactions API to generate a
visualization. Stores the result in MinIO.
"""

import base64
import io
import logging
import time
import psycopg2
import boto3
from botocore.config import Config
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

IMAGE_BUCKET = "dcc-codex"
RATE_LIMIT_SECONDS = 3.0  # Image gen has stricter rate limits

PROMPT_BUILDER_SYSTEM = """You are creating image generation prompts for a Dungeon Crawler Carl compendium.
Your goal is to create vivid, accurate visualizations based ONLY on the author's exact descriptions.
Do not add details not present in the source text. Keep the dungeon aesthetic — gritty, alien, dangerous.
Return ONLY the image prompt text, nothing else."""

PROMPT_BUILDER_USER = """Based on these exact passages from the book, create an image generation prompt for: {entity_name} ({entity_type})

PASSAGES:
{passages}

Create a single detailed image prompt that captures the physical appearance described.
Include: visual style (dark fantasy illustration), lighting (dungeon torchlight),
and all specific details mentioned. Be precise about colors, sizes, materials, anatomy.
Max 400 words."""


def build_image_prompt(entity_name: str, entity_type: str, passages: list[str], client) -> str:
    """Use Gemini Flash to synthesize a detailed image prompt from passages."""
    passages_text = "\n\n---\n\n".join(f'"{p}"' for p in passages[:10])

    prompt = f"{PROMPT_BUILDER_SYSTEM}\n\n{PROMPT_BUILDER_USER.format(entity_name=entity_name, entity_type=entity_type, passages=passages_text)}"

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3),
    )
    return response.text.strip()


def generate_image(prompt: str, client) -> bytes | None:
    """
    Call Nano Banana 2 (gemini-3.1-flash-image) via Interactions API.
    Returns PNG bytes or None.
    """
    try:
        interaction = client.interactions.create(
            model="gemini-3.1-flash-image",
            input=prompt,
            response_format={
                "type": "image",
                "mime_type": "image/png",
                "aspect_ratio": "1:1",
                "image_size": "1K",
            },
        )
        time.sleep(RATE_LIMIT_SECONDS)

        if interaction.output_image and interaction.output_image.data:
            return base64.b64decode(interaction.output_image.data)
        return None
    except Exception as e:
        logger.error(f"Nano Banana image generation failed: {e}")
        return None


def upload_to_minio(minio_client, entity_slug: str, image_bytes: bytes) -> str:
    """Upload image to MinIO and return the public URL."""
    key = f"entities/{entity_slug}.png"
    minio_client.put_object(
        Bucket=IMAGE_BUCKET,
        Key=key,
        Body=io.BytesIO(image_bytes),
        ContentType="image/png",
    )
    return f"/images/{entity_slug}.png"  # served via app proxy


def ensure_bucket(minio_client):
    """Create the dcc-codex bucket if it doesn't exist."""
    try:
        minio_client.head_bucket(Bucket=IMAGE_BUCKET)
    except Exception:
        minio_client.create_bucket(Bucket=IMAGE_BUCKET)
        logger.info(f"Created MinIO bucket: {IMAGE_BUCKET}")


def run_imager(
    conn,
    gemini_api_key: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    batch_size: int = 20,
):
    """
    Main image generation loop.
    Finds entities with physical passages but no image, generates and uploads.
    """
    client = genai.Client(api_key=gemini_api_key)

    minio_client = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        config=Config(signature_version="s3v4"),
    )
    ensure_bucket(minio_client)

    cur = conn.cursor()

    # Find entities with physical passages but no image, major entities first
    cur.execute(
        """
        SELECT DISTINCT e.id, e.name, e.slug, e.entity_type::text
        FROM entities e
        JOIN passages p ON p.entity_id = e.id
        WHERE p.passage_type = 'physical'
          AND (e.image_url IS NULL OR e.image_url = '')
        ORDER BY e.is_major DESC, e.id
        LIMIT %s
        """,
        (batch_size,),
    )
    entities = cur.fetchall()

    if not entities:
        logger.info("No entities need images.")
        cur.close()
        return 0

    logger.info(f"Generating images for {len(entities)} entities (Nano Banana 2)...")
    generated = 0

    for entity_id, name, slug, entity_type in entities:
        # Fetch physical description passages
        cur.execute(
            """
            SELECT passage_text FROM passages
            WHERE entity_id = %s AND passage_type = 'physical'
            ORDER BY id
            """,
            (entity_id,),
        )
        passages = [row[0] for row in cur.fetchall()]

        if not passages:
            continue

        logger.info(f"  Generating image for: {name}")

        # Build prompt from passages
        try:
            image_prompt = build_image_prompt(name, entity_type, passages, client)
        except Exception as e:
            logger.warning(f"  Prompt build failed for {name}: {e}")
            continue

        # Generate image
        image_bytes = generate_image(image_prompt, client)
        if image_bytes is None:
            logger.warning(f"  Skipping {name} — Nano Banana returned no image")
            continue

        # Upload to MinIO
        try:
            image_url = upload_to_minio(minio_client, slug, image_bytes)
        except Exception as e:
            logger.warning(f"  MinIO upload failed for {name}: {e}")
            continue

        # Update entity record
        cur.execute(
            """
            UPDATE entities
            SET image_url = %s, image_prompt = %s, image_source_passages = %s
            WHERE id = %s
            """,
            (image_url, image_prompt, passages[:5], entity_id),
        )
        conn.commit()
        generated += 1
        logger.info(f"  ✓ {name} → {image_url}")

    cur.close()
    logger.info(f"Image generation complete. {generated} images created.")
    return generated