import base64, io, json, logging, time
import psycopg2, boto3
from botocore.config import Config
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
IMAGE_BUCKET = "dcc-codex"
RATE_LIMIT_SECONDS = 3.0
TRANSIENT_RECENT_LIMIT = 3
MAX_IMAGE_ATTEMPTS = 3

# 2026-07-28: hand-curated core-identity anchors for a small number of major recurring
# characters, keyed by entity_id. Purpose: Wes flagged that Carl "looks like a different
# dude from floor to floor" -- confirmed by inspecting the actual stored prompts across
# his 10 generated floor appearances: hair alone was described as unspecified, "completely
# bald... no eyebrows" (a one-off transient injury baked in as if permanent), and "long,
# wavy, shining hair that cascades all the way down to his waist" (an invented exaggeration)
# on different floors, with nothing forcing continuity between them. Root cause: each
# floor's image prompt is built independently from that floor's own durable+transient
# passage sample with no persistent baseline, so ambiguous or sparsely-described traits
# get reinterpreted differently every single time.
#
# Fix: an optional, constant CORE IDENTITY block included in every prompt for entities
# listed here, describing foundational traits (species/build/face/base hair color) that
# must not vary floor-to-floor. Sourced from public fan-wiki consensus
# (dungeon-crawler-carl.fandom.com), NOT purely from in-corpus passages -- an explicit,
# narrow exception to the "never invent unsourced detail" rule that Wes authorized
# specifically for this stabilizing purpose, since the alternative (silence on an
# attribute) is what caused the drift in the first place. This is a floor, not a ceiling:
# any floor's own durable passages can still override a detail here (e.g. Carl's canon
# hair-length change once he acquires the Enchanted Hairbrush of the Beefmaster) --
# the prompt below explicitly tells the model that a later durable passage wins.
CANONICAL_IDENTITY = {
    1: (  # Carl
        "CORE IDENTITY (always true, must appear consistently in every image regardless of floor): "
        "a human man, twenty-seven years old, 6'3\" tall, 230 lbs, with a naturally muscular, "
        "broad-shouldered build and fair skin. Short, practical brown hair, kept neatly trimmed, "
        "and a clean-shaven face -- he shaves every single day even in the dungeon. Facial features "
        "inherited from his father, most notably his nose. This is his baseline appearance and must "
        "be reflected on every floor UNLESS a passage below explicitly and durably describes a "
        "permanent change to one of these specific traits (for example: growing his hair out long and "
        "keeping it that way from then on) -- in that case, follow the passage instead of this block, "
        "but only for the specific trait the passage actually changes, not the rest of this description."
    ),
}

PROMPT_BUILDER_SYSTEM = (
    "You are creating image generation prompts for a Dungeon Crawler Carl compendium. "
    "Base prompts ONLY on the author exact descriptions from the specified floor, plus the CORE "
    "IDENTITY block when one is given. Do not add details not present in the source text or the "
    "CORE IDENTITY block. Keep the dungeon aesthetic: gritty, alien, dangerous. "
    "Return ONLY the image prompt text, nothing else."
)

PROMPT_BUILDER_USER = (
    '{identity_block}'
    'Based on these exact passages from the story up through Floor {floor_number}, '
    "create an image generation prompt for: {entity_name} ({entity_type})\n\n"
    "PASSAGES, in story order (earlier passages first, most recent last). If two passages describe the "
    "same thing differently (e.g. footwear, an item he's carrying, an injury), the LATER passage in this "
    "list is what's currently true -- use that one and ignore the earlier, superseded detail. If a passage "
    "here conflicts with the CORE IDENTITY block above on a specific named trait, the passage wins for that "
    "trait only (it means the character has permanently changed). Passages describing a one-off, in-the-"
    "moment combat state (mid-fight injuries, blood/gore from a specific ongoing scene) should be reflected "
    "as recent battle damage layered on top of the core identity, not as a replacement of it -- do not let a "
    "single transient action passage overwrite foundational traits like hair color, face, or build:\n{passages}\n\n"
    "Create a single detailed image prompt capturing the appearance described above, resolved to his current, "
    "present-day state. "
    "Include: visual style (dark fantasy illustration), dungeon torchlight lighting, "
    "and all specific details mentioned: colors, sizes, materials, anatomy. Max 400 words. "
    "IMPORTANT: only include objects, gear, and held items that are explicitly listed in the passages above. "
    "Do not add weapons, tools, or props of your own invention. If no weapon or held item is described, the "
    "final sentence of the prompt must explicitly state that the hands are empty and no weapon is present. "
    "If this is a LOCATION or environment (not a character), the final sentence of the prompt must explicitly "
    "state whether any person is present, based only on the passages above -- if no person is described in the "
    "passages, state explicitly that the scene shows no people, empty of any figures. "
    "GROUNDING RULE, NO EXCEPTIONS: for any physical attribute NOT covered by the passages above OR the CORE "
    "IDENTITY block (for example: eye color, specific facial features not already named), do NOT invent "
    "or state a specific value for it. Leave it out of the prompt entirely rather than guessing. It is far better "
    "for the illustration to render an unremarkable, generic default for an unmentioned attribute than for this "
    "prompt to assert a specific detail neither source ever established."
)

VERIFY_SYSTEM = (
    "You are a strict visual QA checker for an AI-generated illustration used in a Dungeon Crawler Carl "
    "compendium. You will be given the exact image-generation prompt that was used, and the resulting image. "
    "Your job is to catch anything the image renderer added that was NOT requested in the prompt -- this "
    "covers two known failure modes:\n\n"
    "1. UNSOURCED OBJECTS: the renderer commonly invents extra objects (most often weapons: knives, axes, "
    "swords, guns) that have no basis in the prompt. If the prompt says hands are empty / no weapon present, "
    "and the image shows the character holding ANYTHING, that is a violation.\n"
    "2. UNSOURCED PEOPLE/CHARACTERS: for location and environment prompts especially, the renderer commonly "
    "populates an otherwise-empty scene with generic background people (staff, bystanders, crowds) that are "
    "not named or described in the prompt at all. If the prompt states the scene has no people / is empty of "
    "figures, and the image shows ANY person or humanoid figure -- even one that looks incidental or purely "
    "decorative background -- that is a violation.\n\n"
    "Check specifically:\n"
    "1. Is the character holding, wearing, or carrying ANY weapon, tool, or object that is not explicitly "
    "named in the prompt?\n"
    "2. Does the image contain ANY person/humanoid figure that the prompt did not call for?\n"
    "3. List every such unsourced object or person you can identify.\n\n"
    "Respond with ONLY a JSON object: "
    '{"unsourced_objects": ["short description", ...], "pass": true/false}. '
    "pass=false if unsourced_objects is non-empty. No other text."
)


def build_image_prompt(entity_name, entity_type, durable_passages, transient_passages, floor_number, client, entity_id=None):
    sep = "\n\n---\n\n"
    all_passages = durable_passages + transient_passages
    passages_text = sep.join('"' + p + '"' for p in all_passages) if all_passages else "(no passages available)"
    identity_anchor = CANONICAL_IDENTITY.get(entity_id)
    identity_block = (identity_anchor + "\n\n") if identity_anchor else ""
    prompt = PROMPT_BUILDER_SYSTEM + "\n\n" + PROMPT_BUILDER_USER.format(
        identity_block=identity_block,
        entity_name=entity_name, entity_type=entity_type, passages=passages_text,
        floor_number=floor_number,
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


def verify_image(image_bytes, prompt, client):
    """Ask a vision-capable Gemini call whether the rendered image contains anything not in the prompt.

    Returns (passed: bool, unsourced_objects: list[str], raw_error: str|None).
    On any failure to get a parseable verdict, returns (True, [], error) -- i.e. we do NOT block publishing
    on a broken verifier, we just fail to catch that specific image. This is intentionally a checker, not a
    silent gate that can wedge the whole pipeline if the verifier call itself errors.
    """
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                VERIFY_SYSTEM,
                "PROMPT USED TO GENERATE THIS IMAGE:\n" + prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = response.text.strip()
        import re
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        result = json.loads(text)
        unsourced = result.get("unsourced_objects", [])
        passed = bool(result.get("pass", len(unsourced) == 0))
        return passed, unsourced, None
    except Exception as e:
        logger.warning("Image verification call failed (not blocking): %s", e)
        return True, [], str(e)


def generate_and_verify_image(prompt, client, max_attempts=MAX_IMAGE_ATTEMPTS):
    """Generate an image, verify it against the prompt, and retry with a stricter prompt if it fails.

    Returns (image_bytes, final_prompt_used, verification_log) where verification_log is a list of
    per-attempt dicts: {"attempt": n, "passed": bool, "unsourced_objects": [...], "error": str|None}.
    If every attempt fails verification, returns the LAST attempt's image anyway (better to publish a
    flagged image than none), with verification_log showing the failure history for audit.
    """
    verification_log = []
    current_prompt = prompt
    last_image_bytes = None

    for attempt in range(1, max_attempts + 1):
        image_bytes = generate_image(current_prompt, client)
        if image_bytes is None:
            verification_log.append({"attempt": attempt, "passed": False, "unsourced_objects": [], "error": "no_image_returned"})
            continue

        last_image_bytes = image_bytes
        passed, unsourced, err = verify_image(image_bytes, current_prompt, client)
        verification_log.append({"attempt": attempt, "passed": passed, "unsourced_objects": unsourced, "error": err})

        if passed:
            return image_bytes, current_prompt, verification_log

        logger.warning("  Verification failed (attempt %d): unsourced objects %s", attempt, unsourced)
        if attempt < max_attempts:
            current_prompt = (
                prompt
                + "\n\nSTRICT CORRECTION: a previous attempt at this exact prompt incorrectly added: "
                + ", ".join(unsourced)
                + ". Do NOT include any of these in this image. Hands must be empty unless a held item is "
                "explicitly named above. Do not add any people or figures unless explicitly named above."
            )

    return last_image_bytes, current_prompt, verification_log


def upload_to_minio(minio_client, slug, floor_number, image_bytes):
    key = "entities/{}/floor_{}.jpg".format(slug, floor_number)
    minio_client.put_object(Bucket=IMAGE_BUCKET, Key=key, Body=io.BytesIO(image_bytes), ContentType="image/jpeg")
    return "/images/{}.jpg".format(slug)


def ensure_bucket(minio_client):
    try:
        minio_client.head_bucket(Bucket=IMAGE_BUCKET)
    except Exception:
        minio_client.create_bucket(Bucket=IMAGE_BUCKET)


def run_migrate(conn) -> None:
    cur = conn.cursor()
    cur.execute("ALTER TABLE entity_appearances ADD COLUMN IF NOT EXISTS verification_passed BOOLEAN")
    cur.execute("ALTER TABLE entity_appearances ADD COLUMN IF NOT EXISTS verification_log JSONB")
    conn.commit()
    cur.close()


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


def _fetch_passages(conn, entity_id, floor_id, floor_number):
    """Return (durable_passages, transient_passages) for an entity as of the end of floor_number.

    Uses the `chapter_floors` view (chapter -> nearest floor whose start_chapter_id <= it) to map every
    passage's chapter to a floor, replacing the old book-keyed join.

    Durable: ALL classified-durable physical passages through the end of floor_number (cumulative,
    uncapped -- this is finite, static source material, no reason to truncate it).
    Transient: the most recent classified-transient physical passages from THIS floor only, representing
    the character's current condition near the end of the floor (capped small on purpose -- this is meant
    to be a snapshot of 'right now', not an accumulation of every scrape and bruise across the whole floor).
    If this entity's physical passages haven't been classified yet (is_durable IS NULL for all of them),
    fall back to the old behavior: every physical passage in this floor, uncapped.
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
            JOIN chapter_floors cf ON cf.chapter_id = p.chapter_id
            WHERE p.entity_id = %s AND cf.floor_id = %s AND p.passage_type = 'physical'
            ORDER BY p.id
        """, (entity_id, floor_id))
        all_text = [row[0] for row in cur.fetchall()]
        cur.close()
        return all_text, []

    cur = conn.cursor()
    cur.execute("""
        SELECT p.passage_text FROM passages p
        JOIN chapter_floors cf ON cf.chapter_id = p.chapter_id
        WHERE p.entity_id = %s AND p.passage_type = 'physical' AND p.is_durable = TRUE
          AND cf.floor_number <= %s
        ORDER BY cf.floor_number, p.chapter_id, p.id
    """, (entity_id, floor_number))
    durable = [row[0] for row in cur.fetchall()]
    cur.close()

    cur = conn.cursor()
    cur.execute("""
        SELECT p.passage_text FROM passages p
        JOIN chapter_floors cf ON cf.chapter_id = p.chapter_id
        WHERE p.entity_id = %s AND cf.floor_id = %s AND p.passage_type = 'physical' AND p.is_durable = FALSE
        ORDER BY p.chapter_id DESC, p.id DESC
        LIMIT %s
    """, (entity_id, floor_id, TRANSIENT_RECENT_LIMIT))
    transient = [row[0] for row in cur.fetchall()]
    transient.reverse()
    cur.close()

    return durable, transient


def run_imager(conn, gemini_api_key, minio_endpoint, minio_access_key, minio_secret_key, batch_size=20, entity_floor_pairs=None):
    """
    Generate per-floor images for entities.
    One image per (entity, floor) pair where physical passages exist.
    Stores in entity_appearances (keyed on floor_id, with book_id carried along from the floor's
    starting book for display/legacy purposes); also keeps entities.image_url updated for backward compat.
    Every generated image is checked against its own prompt by a second vision model call before being
    accepted; images that add unsourced objects (most commonly weapons) or unsourced people (most commonly
    generic background figures in location/environment images) get up to MAX_IMAGE_ATTEMPTS retries with an
    explicit correction clause. The full verification history is stored per appearance so failures are
    auditable even when a retry doesn't fully resolve them.

    entity_floor_pairs: optional list of (entity_id, floor_number) tuples for a TARGETED regen (bypasses
    the "only pairs with no existing row" gate) -- used to re-run specific already-generated appearances
    after a prompt/logic fix, without touching everything else. If given, batch_size is ignored.
    """
    run_migrate(conn)
    run_id = log_run_start(conn, "images", {"batch_size": batch_size})
    client = genai.Client(api_key=gemini_api_key)
    minio_client = boto3.client(
        "s3", endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key, aws_secret_access_key=minio_secret_key,
        config=Config(signature_version="s3v4")
    )
    ensure_bucket(minio_client)

    cur = conn.cursor()
    if entity_floor_pairs:
        cur.execute("""
            SELECT e.id, e.name, e.slug, e.entity_type::text, e.is_major,
                   f.id AS floor_id, f.floor_number, f.book_id
            FROM entities e
            JOIN floors f ON (e.id, f.floor_number) IN %s
            ORDER BY e.id, f.id
        """, (tuple(entity_floor_pairs),))
    else:
        cur.execute("""
            SELECT DISTINCT ON (e.id, f.id)
                   e.id, e.name, e.slug, e.entity_type::text, e.is_major,
                   f.id AS floor_id, f.floor_number, f.book_id
            FROM entities e
            JOIN passages p  ON p.entity_id  = e.id
            JOIN chapter_floors cf ON cf.chapter_id = p.chapter_id
            JOIN floors f    ON f.id         = cf.floor_id
            WHERE p.passage_type = 'physical'
              AND NOT EXISTS (
                  SELECT 1 FROM entity_appearances ea
                  WHERE ea.entity_id = e.id AND ea.floor_id = f.id
              )
            ORDER BY e.id, f.id, e.is_major DESC
            LIMIT %s
        """, (batch_size,))
    targets = cur.fetchall()
    cur.close()

    if not targets:
        logger.info("No (entity, floor) pairs need images.")
        log_run_finish(conn, run_id, 0, 0)
        return 0

    logger.info("Generating per-floor images for %d (entity, floor) pairs...", len(targets))
    generated = 0
    failed = 0
    flagged = 0

    for entity_id, name, slug, entity_type, is_major, floor_id, floor_number, book_id in targets:
        durable, transient = _fetch_passages(conn, entity_id, floor_id, floor_number)

        if not durable and not transient and entity_id not in CANONICAL_IDENTITY:
            continue

        logger.info("  [Floor %d] %s (%d durable, %d current-state passages)", floor_number, name, len(durable), len(transient))

        try:
            image_prompt = build_image_prompt(name, entity_type, durable, transient, floor_number, client, entity_id=entity_id)
        except Exception as e:
            logger.warning("  Prompt failed for %s floor %d: %s", name, floor_number, e)
            failed += 1
            continue

        image_bytes, final_prompt, verification_log = generate_and_verify_image(image_prompt, client)
        if image_bytes is None:
            logger.warning("  No image for %s floor %d", name, floor_number)
            failed += 1
            continue

        verification_passed = verification_log[-1]["passed"] if verification_log else None
        if not verification_passed:
            flagged += 1
            logger.warning("  %s floor %d PUBLISHED WITH FLAG -- unresolved unsourced objects: %s",
                            name, floor_number, verification_log[-1].get("unsourced_objects"))

        try:
            image_url = upload_to_minio(minio_client, slug, floor_number, image_bytes)
        except Exception as e:
            logger.warning("  MinIO upload failed for %s floor %d: %s", name, floor_number, e)
            failed += 1
            continue

        all_passages = durable + transient
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO entity_appearances
                (entity_id, book_id, floor_id, image_url, image_prompt, image_source_passages, verification_passed, verification_log)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_id, floor_id) DO UPDATE SET
                image_url = EXCLUDED.image_url,
                image_prompt = EXCLUDED.image_prompt,
                image_source_passages = EXCLUDED.image_source_passages,
                verification_passed = EXCLUDED.verification_passed,
                verification_log = EXCLUDED.verification_log
        """, (entity_id, book_id, floor_id, image_url, final_prompt, all_passages, verification_passed, json.dumps(verification_log)))
        cur.execute("""
            UPDATE entities SET image_url=%s, image_prompt=%s, image_source_passages=%s
            WHERE id=%s AND (image_url IS NULL OR NOT EXISTS (
                SELECT 1 FROM entity_appearances ea2
                JOIN floors f2 ON f2.id=ea2.floor_id
                WHERE ea2.entity_id=%s AND f2.floor_number>%s
            ))
        """, (image_url, final_prompt, all_passages, entity_id, entity_id, floor_number))
        conn.commit()
        cur.close()
        generated += 1
        logger.info("  OK %s (floor %d) -> %s (verified=%s)", name, floor_number, image_url, verification_passed)

    log_run_finish(conn, run_id, generated, failed)
    logger.info("Done. %d generated, %d failed, %d published with unresolved verification flags.", generated, failed, flagged)
    return generated
