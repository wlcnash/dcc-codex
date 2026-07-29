"""DCC Codex - FastAPI application."""
import io, os, logging
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, text
import boto3
from botocore.config import Config
from database import get_db, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
from models import Book, Chapter, ChapterFloor, Entity, EntityAppearance, Floor, Passage, EntityRelationship, EntityTypeEnum

logger = logging.getLogger(__name__)
app = FastAPI(title="DCC Codex", docs_url=None, redoc_url=None)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
IMAGE_BUCKET = "dcc-codex"

ENTITY_TYPE_LABELS = {"crawler":"Crawlers","npc":"NPCs","mob":"Mobs","item":"Items","location":"Locations","floor":"Floors","ability":"Abilities","faction":"Factions","deity":"Deities","media":"Media","other":"Other"}
ENTITY_TYPE_ICONS = {"crawler":"🎯","npc":"👤","mob":"🐉","item":"⚔","location":"🗺","floor":"🏚","ability":"✨","faction":"⚑","deity":"⚡","media":"📺","other":"◈"}

def get_max_floor(request: Request):
    val = request.cookies.get("max_floor")
    if val and val.isdigit(): return int(val)
    return None

def get_floors(db: Session):
    return db.query(Floor).order_by(Floor.floor_number).all()

def base_context(request: Request, db: Session) -> dict:
    max_floor = get_max_floor(request)
    floors = get_floors(db)
    max_floor_obj = next((f for f in floors if f.floor_number == max_floor), None) if max_floor else None
    return {"request": request, "floors": floors, "max_floor": max_floor, "max_floor_obj": max_floor_obj, "entity_type_icons": ENTITY_TYPE_ICONS}

def get_minio_client():
    return boto3.client("s3", endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY, config=Config(signature_version="s3v4"))

def resolve_entity_image_key(slug: str, max_floor, db: Session):
    """Return the MinIO key for an entity image given the user floor context.
    Checks entity_appearances for the highest floor <= max_floor.

    2026-07-28 fix: previously, when no floor cookie was set (max_floor is None),
    this skipped entity_appearances entirely and fell straight to the legacy
    entities/{slug}.jpg blob -- meaning every first-time visitor with no floor
    selected saw the OLDEST, unverified image, while all the floor-scoped work
    (verified against actual passage descriptions) sat unused behind a filter
    nobody sets by default. Now, with no floor selected, we default to the
    EARLIEST available floor-scoped appearance (the least-spoiler choice) instead
    of the legacy fallback. Legacy fallback is now reserved for entities with no
    floor-scoped appearance at all."""
    entity = db.query(Entity).filter(Entity.slug == slug).first()
    if entity:
        appearance_q = (db.query(EntityAppearance)
            .join(Floor, EntityAppearance.floor_id == Floor.id)
            .filter(EntityAppearance.entity_id == entity.id, EntityAppearance.image_url.isnot(None)))
        if max_floor is not None:
            appearance = appearance_q.filter(Floor.floor_number <= max_floor).order_by(Floor.floor_number.desc()).first()
        else:
            appearance = appearance_q.order_by(Floor.floor_number.asc()).first()
        if appearance:
            return f"entities/{slug}/floor_{appearance.floor.floor_number}.jpg"
    return f"entities/{slug}.jpg"

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/set-floor")
def set_floor(floor: Optional[int]=Query(None), redirect_to: str=Query("/")):
    response = RedirectResponse(url=redirect_to, status_code=302)
    if floor is not None:
        response.set_cookie("max_floor", str(floor), max_age=60*60*24*365, path="/")
    else:
        response.delete_cookie("max_floor", path="/")
    return response

@app.get("/images/{slug}.jpg")
def serve_image_jpg(slug: str, request: Request, db: Session=Depends(get_db)):
    """Serve the most appropriate image for the entity given the user selected floor.
    Checks entity_appearances for a per-floor image; falls back to legacy canonical image."""
    max_floor = get_max_floor(request)
    key = resolve_entity_image_key(slug, max_floor, db)
    try:
        minio = get_minio_client()
        obj = minio.get_object(Bucket=IMAGE_BUCKET, Key=key)
        return StreamingResponse(obj["Body"], media_type="image/jpeg")
    except Exception:
        if key != f"entities/{slug}.jpg":
            try:
                minio = get_minio_client()
                obj = minio.get_object(Bucket=IMAGE_BUCKET, Key=f"entities/{slug}.jpg")
                return StreamingResponse(obj["Body"], media_type="image/jpeg")
            except Exception: pass
        logger.warning(f"Image not found: slug={slug} key={key}")
        raise HTTPException(status_code=404, detail="Image not found")

@app.get("/images/{slug}.png")
def serve_image_png(slug: str):
    try:
        minio = get_minio_client()
        obj = minio.get_object(Bucket=IMAGE_BUCKET, Key=f"entities/{slug}.png")
        return StreamingResponse(obj["Body"], media_type="image/png")
    except Exception:
        raise HTTPException(status_code=404, detail="Image not found")

@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session=Depends(get_db)):
    ctx = base_context(request, db)
    max_floor = ctx["max_floor"]
    # "Notable Crawlers" (renamed 2026-07-29 from "Notable Contestants" -- queried passages
    # directly and "contestant" appears exactly once in the whole corpus, in throwaway dialogue
    # about someone else, vs. 228 uses of "crawler"; "crawler" is the book's actual term) --
    # this previously had no entity_type filter, so major NPCs/items/mobs/deities could show up
    # under a section about dungeon competitors, which is factually wrong in-universe. Wes
    # caught this live ("why are there a mix of creatures in contestants").
    featured_q = db.query(Entity).filter(Entity.is_major==True, Entity.image_url.isnot(None), Entity.entity_type==EntityTypeEnum.crawler)
    if max_floor is not None:
        featured_q = featured_q.outerjoin(ChapterFloor, Entity.first_chapter_id==ChapterFloor.chapter_id).filter(or_(Entity.first_chapter_id==None, ChapterFloor.floor_number<=max_floor))
    ctx.update({
        "total_entities": db.query(func.count(Entity.id)).scalar(),
        "total_passages": db.query(func.count(Passage.id)).scalar(),
        "total_chapters": db.query(func.count(Chapter.id)).scalar(),
        "imaged_entities": db.query(func.count(Entity.id)).filter(Entity.image_url.isnot(None)).scalar(),
        "counts_by_type": {t.value: c for t, c in db.query(Entity.entity_type, func.count(Entity.id)).group_by(Entity.entity_type).all()},
        "entity_type_labels": ENTITY_TYPE_LABELS,
        "featured": featured_q.order_by(func.random()).limit(6).all(),
    })
    return templates.TemplateResponse("home.html", ctx)

@app.get("/browse", response_class=HTMLResponse)
def browse(request: Request, entity_type: Optional[str]=Query(None), q: Optional[str]=Query(None),
           major_only: bool=Query(False), page: int=Query(1,ge=1), per_page: int=Query(48,ge=1,le=100),
           db: Session=Depends(get_db)):
    ctx = base_context(request, db)
    max_floor = ctx["max_floor"]
    query = db.query(Entity)
    if entity_type and entity_type in [t.value for t in EntityTypeEnum]:
        query = query.filter(Entity.entity_type==entity_type)
    if q:
        query = query.filter(or_(Entity.name.ilike(f"%{q}%"), Entity.summary.ilike(f"%{q}%")))
    if major_only: query = query.filter(Entity.is_major==True)
    if max_floor is not None:
        query = query.outerjoin(ChapterFloor, Entity.first_chapter_id==ChapterFloor.chapter_id).filter(or_(Entity.first_chapter_id==None, ChapterFloor.floor_number<=max_floor))
    total = query.count()
    entities = query.order_by(Entity.is_major.desc(), Entity.name).offset((page-1)*per_page).limit(per_page).all()
    ctx.update({"entities": entities, "entity_type": entity_type, "entity_type_labels": ENTITY_TYPE_LABELS,
                "q": q, "major_only": major_only, "page": page, "per_page": per_page,
                "total": total, "total_pages": (total+per_page-1)//per_page})
    return templates.TemplateResponse("entity_list.html", ctx)

@app.get("/entity/{slug}", response_class=HTMLResponse)
def entity_detail(slug: str, request: Request, db: Session=Depends(get_db)):
    ctx = base_context(request, db)
    max_floor = ctx["max_floor"]
    entity = db.query(Entity).filter(Entity.slug==slug).first()
    if not entity: raise HTTPException(status_code=404, detail="Entity not found")
    total_passage_count = db.query(func.count(Passage.id)).filter(Passage.entity_id==entity.id).scalar()
    pq = (db.query(Passage, Chapter, Book).join(Chapter, Passage.chapter_id==Chapter.id)
          .join(Book, Chapter.book_id==Book.id).filter(Passage.entity_id==entity.id))
    if max_floor is not None:
        pq = pq.join(ChapterFloor, Chapter.id==ChapterFloor.chapter_id).filter(ChapterFloor.floor_number<=max_floor)
    passages = pq.order_by(Passage.passage_type, Book.book_number, Chapter.chapter_number).all()
    passages_by_type = {}
    for passage, chapter, book in passages:
        ptype = passage.passage_type.value
        passages_by_type.setdefault(ptype, []).append({"passage": passage, "chapter": chapter, "book": book})
    current_appearance = None
    if max_floor is not None:
        current_appearance = (db.query(EntityAppearance).join(Floor, EntityAppearance.floor_id==Floor.id)
            .filter(EntityAppearance.entity_id==entity.id, Floor.floor_number<=max_floor, EntityAppearance.image_url.isnot(None))
            .order_by(Floor.floor_number.desc()).first())
    relationships = (db.query(EntityRelationship, Entity).join(Entity, EntityRelationship.entity_b_id==Entity.id).filter(EntityRelationship.entity_a_id==entity.id).all() +
                     db.query(EntityRelationship, Entity).join(Entity, EntityRelationship.entity_a_id==Entity.id).filter(EntityRelationship.entity_b_id==entity.id).all())

    # Verified stats/skills only -- see seeder/verify_stats.py. A row is trustworthy
    # for public display only once BOTH independent verification passes confirmed it;
    # unverified/flagged rows are never shown on the page.
    stat_rows = db.execute(text("""
        SELECT stat_name, value, value_type, reason
        FROM entity_stats
        WHERE entity_id = :eid AND verify_pass1 = true AND verify_pass2 = true
        ORDER BY stat_name
    """), {"eid": entity.id}).fetchall()
    skill_rows = db.execute(text("""
        SELECT skill_name, level, value_type, reason
        FROM entity_skills
        WHERE entity_id = :eid AND verify_pass1 = true AND verify_pass2 = true
        ORDER BY skill_name
    """), {"eid": entity.id}).fetchall()

    # Curate aliases: show a handful up front, rest behind a "+N more" toggle in the template
    aliases = entity.aliases or []
    aliases_shown = aliases[:8]
    aliases_rest = aliases[8:]

    ctx.update({"entity": entity, "passages_by_type": passages_by_type,
                "passage_type_order": ["physical","personality","ability","backstory","action","other"],
                "relationships": relationships, "first_book": entity.first_book, "first_chapter": entity.first_chapter,
                "entity_type_labels": ENTITY_TYPE_LABELS,
                "hidden_count": total_passage_count-len(passages), "visible_count": len(passages),
                "total_passage_count": total_passage_count, "current_appearance": current_appearance,
                "stat_rows": stat_rows, "skill_rows": skill_rows,
                "aliases_shown": aliases_shown, "aliases_rest": aliases_rest})
    return templates.TemplateResponse("entity_detail.html", ctx)

@app.get("/api/search")
def search_api(q: str=Query(...,min_length=2), db: Session=Depends(get_db)):
    results = db.query(Entity.id, Entity.name, Entity.slug, Entity.entity_type, Entity.image_url).filter(Entity.name.ilike(f"%{q}%")).order_by(Entity.is_major.desc(), Entity.name).limit(10).all()
    return [{"id": r.id, "name": r.name, "slug": r.slug, "type": r.entity_type.value, "image_url": r.image_url} for r in results]
