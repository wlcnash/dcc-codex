"""DCC Codex — FastAPI application."""

import io
import os
import logging
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
import boto3
from botocore.config import Config

from database import get_db, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
from models import Book, Chapter, Entity, Passage, EntityRelationship, EntityTypeEnum

logger = logging.getLogger(__name__)

app = FastAPI(title="DCC Codex", docs_url=None, redoc_url=None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

IMAGE_BUCKET = "dcc-codex"

ENTITY_TYPE_LABELS = {
    "character": "Characters",
    "creature": "Creatures",
    "item": "Items",
    "location": "Locations",
    "floor": "Floors",
    "ability": "Abilities",
    "faction": "Factions",
    "other": "Other",
}


def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# Image proxy (serves from MinIO internally)
# ─────────────────────────────────────────────

@app.get("/images/{slug}.png")
def serve_image(slug: str):
    """Proxy entity images from MinIO so they're served under the app's domain."""
    try:
        minio = get_minio_client()
        obj = minio.get_object(Bucket=IMAGE_BUCKET, Key=f"entities/{slug}.png")
        return StreamingResponse(obj["Body"], media_type="image/png")
    except Exception as e:
        logger.warning(f"Image not found for {slug}: {e}")
        raise HTTPException(status_code=404, detail="Image not found")


# ─────────────────────────────────────────────
# Home page
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    # Stats for home page
    total_entities = db.query(func.count(Entity.id)).scalar()
    total_passages = db.query(func.count(Passage.id)).scalar()
    total_chapters = db.query(func.count(Chapter.id)).scalar()
    imaged_entities = db.query(func.count(Entity.id)).filter(Entity.image_url.isnot(None)).scalar()

    type_counts = (
        db.query(Entity.entity_type, func.count(Entity.id))
        .group_by(Entity.entity_type)
        .all()
    )
    counts_by_type = {t.value: c for t, c in type_counts}

    featured = (
        db.query(Entity)
        .filter(Entity.is_major == True, Entity.image_url.isnot(None))
        .order_by(func.random())
        .limit(6)
        .all()
    )

    return templates.TemplateResponse("home.html", {
        "request": request,
        "total_entities": total_entities,
        "total_passages": total_passages,
        "total_chapters": total_chapters,
        "imaged_entities": imaged_entities,
        "counts_by_type": counts_by_type,
        "entity_type_labels": ENTITY_TYPE_LABELS,
        "featured": featured,
    })


# ─────────────────────────────────────────────
# Entity list (browse by type, search)
# ─────────────────────────────────────────────

@app.get("/browse", response_class=HTMLResponse)
def browse(
    request: Request,
    entity_type: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    major_only: bool = Query(False),
    page: int = Query(1, ge=1),
    per_page: int = Query(48, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = db.query(Entity)

    if entity_type and entity_type in [t.value for t in EntityTypeEnum]:
        query = query.filter(Entity.entity_type == entity_type)

    if q:
        query = query.filter(
            or_(
                Entity.name.ilike(f"%{q}%"),
                Entity.summary.ilike(f"%{q}%"),
            )
        )

    if major_only:
        query = query.filter(Entity.is_major == True)

    total = query.count()
    entities = (
        query
        .order_by(Entity.is_major.desc(), Entity.name)
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    total_pages = (total + per_page - 1) // per_page

    return templates.TemplateResponse("entity_list.html", {
        "request": request,
        "entities": entities,
        "entity_type": entity_type,
        "entity_type_labels": ENTITY_TYPE_LABELS,
        "q": q,
        "major_only": major_only,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
    })


# ─────────────────────────────────────────────
# Entity detail
# ─────────────────────────────────────────────

@app.get("/entity/{slug}", response_class=HTMLResponse)
def entity_detail(slug: str, request: Request, db: Session = Depends(get_db)):
    entity = db.query(Entity).filter(Entity.slug == slug).first()
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    # Get all passages, ordered by type then book/chapter
    passages = (
        db.query(Passage, Chapter, Book)
        .join(Chapter, Passage.chapter_id == Chapter.id)
        .join(Book, Chapter.book_id == Book.id)
        .filter(Passage.entity_id == entity.id)
        .order_by(Passage.passage_type, Book.book_number, Chapter.chapter_number)
        .all()
    )

    # Group passages by type
    passages_by_type: dict[str, list] = {}
    for passage, chapter, book in passages:
        ptype = passage.passage_type.value
        if ptype not in passages_by_type:
            passages_by_type[ptype] = []
        passages_by_type[ptype].append({
            "passage": passage,
            "chapter": chapter,
            "book": book,
        })

    # Get relationships
    relationships = (
        db.query(EntityRelationship, Entity)
        .join(Entity, EntityRelationship.entity_b_id == Entity.id)
        .filter(EntityRelationship.entity_a_id == entity.id)
        .all()
    ) + (
        db.query(EntityRelationship, Entity)
        .join(Entity, EntityRelationship.entity_a_id == Entity.id)
        .filter(EntityRelationship.entity_b_id == entity.id)
        .all()
    )

    # First and last appearance
    first_book = entity.first_book
    first_chapter = entity.first_chapter

    passage_type_order = ["physical", "personality", "ability", "backstory", "action", "other"]

    return templates.TemplateResponse("entity_detail.html", {
        "request": request,
        "entity": entity,
        "passages_by_type": passages_by_type,
        "passage_type_order": passage_type_order,
        "relationships": relationships,
        "first_book": first_book,
        "first_chapter": first_chapter,
        "entity_type_labels": ENTITY_TYPE_LABELS,
    })


# ─────────────────────────────────────────────
# Search API (JSON, for typeahead)
# ─────────────────────────────────────────────

@app.get("/api/search")
def search_api(q: str = Query(..., min_length=2), db: Session = Depends(get_db)):
    results = (
        db.query(Entity.id, Entity.name, Entity.slug, Entity.entity_type, Entity.image_url)
        .filter(Entity.name.ilike(f"%{q}%"))
        .order_by(Entity.is_major.desc(), Entity.name)
        .limit(10)
        .all()
    )
    return [
        {
            "id": r.id,
            "name": r.name,
            "slug": r.slug,
            "type": r.entity_type.value,
            "image_url": r.image_url,
        }
        for r in results
    ]
