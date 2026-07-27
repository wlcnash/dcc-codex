"""
DCC Codex seeder — main orchestrator.

Usage:
  python main.py --step all           # full pipeline (scrape → extract → images → persona)
  python main.py --step scrape        # scrape Royal Road chapters
  python main.py --step extract       # entity/passage extraction (Gemini)
  python main.py --step images        # Imagen 3 image generation
  python main.py --step persona       # System AI persona text generation
  python main.py --step migrate       # DB migrations only (idempotent)
  python main.py --step extract --batch 50  # process 50 chapters
"""

import argparse
import logging
import os
import sys
import psycopg2

from scraper  import run_scraper
from extractor import run_extractor
from imager   import run_imager
from persona  import run_persona, run_migrate
from permanence import run_classify
from floors import run_floor_extraction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seeder")

# Chapter boundaries (1-indexed) — where each book starts
BOOK_BOUNDARIES = [1, 52, 107, 162, 215, 268, 321, 374]

VALID_STEPS = ["all", "scrape", "extract", "classify", "floors", "images", "persona", "migrate"]


def get_db_conn():
    return psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def apply_schema(conn):
    """Apply schema.sql if entities table doesn't exist yet."""
    schema_path = os.path.join(os.path.dirname(__file__), "..", "schema.sql")
    if not os.path.exists(schema_path):
        logger.error(f"schema.sql not found at {schema_path}")
        sys.exit(1)

    cur = conn.cursor()
    cur.execute("SELECT to_regclass('public.entities')")
    exists = cur.fetchone()[0]
    cur.close()

    if exists:
        logger.info("Schema already applied, skipping.")
        return

    logger.info("Applying schema...")
    with open(schema_path) as f:
        sql = f.read()
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()
    logger.info("Schema applied.")


def main():
    parser = argparse.ArgumentParser(description="DCC Codex seeder pipeline")
    parser.add_argument(
        "--step",
        choices=VALID_STEPS,
        default="all",
        help="Which pipeline step to run",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (limits entities/chapters processed per run)",
    )
    args = parser.parse_args()

    # Validate env vars
    required_env = ["POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    if args.step in ("all", "extract", "classify", "images", "persona"):
        required_env.append("GEMINI_API_KEY")
    if args.step in ("all", "images"):
        required_env += ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"]

    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    conn = get_db_conn()
    apply_schema(conn)

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    batch = args.batch or 999999

    if args.step == "migrate":
        logger.info("=== Running migrations ===")
        run_migrate(conn)

    if args.step in ("all", "scrape"):
        logger.info("=== STEP 1: Scraping Royal Road ===")
        count = run_scraper(conn, BOOK_BOUNDARIES)
        logger.info(f"Scraped {count} new chapters.")

    if args.step in ("all", "extract"):
        logger.info("=== STEP 2: Extracting entities with Gemini ===")
        count = run_extractor(conn, gemini_key, batch_size=batch)
        logger.info(f"Extracted {count} entity references.")

    if args.step in ("all", "classify"):
        logger.info("=== STEP 2b: Classifying physical passages as durable/transient ===")
        count = run_classify(conn, gemini_key, batch_size=batch)
        logger.info(f"Classified {count} passages.")

    if args.step in ("all", "floors"):
        logger.info("=== STEP 2c: Extracting dungeon-floor transitions (one-time, hard points) ===")
        count = run_floor_extraction(conn, gemini_key)
        logger.info(f"Recorded {count} floor transitions.")

    if args.step in ("all", "images"):
        logger.info("=== STEP 3: Generating images with Imagen 3 ===")
        count = run_imager(
            conn,
            gemini_key,
            minio_endpoint=os.environ["MINIO_ENDPOINT"],
            minio_access_key=os.environ["MINIO_ACCESS_KEY"],
            minio_secret_key=os.environ["MINIO_SECRET_KEY"],
            batch_size=batch,
        )
        logger.info(f"Generated {count} images.")

    if args.step in ("all", "persona"):
        logger.info("=== STEP 4: Generating System AI persona text ===")
        count = run_persona(conn, gemini_key, batch_size=batch)
        logger.info(f"Generated {count} persona entries.")

    conn.close()
    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
