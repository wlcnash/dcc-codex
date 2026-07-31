"""DCC Codex - FastAPI application."""
import io, os, re, logging
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from markupsafe import escape
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, text
import boto3
from botocore.config import Config
from database import get_db, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
from models import Book, Chapter, ChapterFloor, Entity, EntityAppearance, Floor, Passage, EntityRelationship, EntityTypeEnum, BossTierEnum

logger = logging.getLogger(__name__)
app = FastAPI(title="DCC Codex", docs_url=None, redoc_url=None)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
IMAGE_BUCKET = "dcc-codex"

ENTITY_TYPE_LABELS = {"crawler":"Crawlers","npc":"NPCs","mob":"Mobs","item":"Items","location":"Locations","floor":"Floors","ability":"Abilities","faction":"Factions","deity":"Deities","media":"Media","other":"Other"}
ENTITY_TYPE_ICONS = {"crawler":"🎯","npc":"👤","mob":"🐉","item":"⚔","location":"🗺","floor":"🏚","ability":"✨","faction":"⚑","deity":"⚡","media":"📺","other":"◈"}

# 2026-07-29: boss tier, confirmed against actual passage text (see models.py's BossTierEnum
# docstring) -- ordered smallest to largest administrative scope, matching the book's own
# escalation (a crawler clears neighborhood bosses before borough, city, province, country).
BOSS_TIER_LABELS = {"neighborhood":"Neighborhood Boss","borough":"Borough Boss","city":"City Boss","province":"Province Boss","country":"Country Boss","floor":"Floor Boss"}
BOSS_TIER_ORDER = ["neighborhood","borough","city","province","country","floor"]

# 2026-07-30: relation_type is always stored in the FORWARD direction (the entity being
# scanned's relation TO the other party) -- see seeder/relationships.py's module docstring
# for the full rationale. This dict is the display-layer mirror: it maps a forward
# relation_type to how it should read when shown on the OTHER entity's page instead. Kept in
# sync by hand with SUGGESTED_RELATION_TYPES in seeder/relationships.py -- they're
# independently-built/deployed images that don't import from each other.
# Anything not in this dict falls back to the same word on both sides (symmetric display),
# which is correct for relationships like "ally_of"/"rival_of"/"friend_of"/"sibling_of" and
# a reasonable default for anything not yet curated.
RELATION_REVERSE_LABELS = {
    "guards": "guarded by", "member_of": "has member", "leads": "led by",
    "mentor_of": "student of", "owns_pet": "pet of", "parent_of": "child of",
    "serves": "served by", "worships": "worshipped by", "works_at": "employs",
    "created": "created by",
}

def _pretty_relation(relation_type: str, viewed_as_a: bool) -> str:
    """Human-readable label for a relation_type as seen from one side of the pair.
    See RELATION_REVERSE_LABELS above for why entity_b's page doesn't just echo the
    same forward-direction word entity_a's page shows."""
    label = relation_type if viewed_as_a else RELATION_REVERSE_LABELS.get(relation_type, relation_type)
    return label.replace("_", " ").capitalize()

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
    return {"request": request, "floors": floors, "max_floor": max_floor, "max_floor_obj": max_floor_obj, "entity_type_icons": ENTITY_TYPE_ICONS, "boss_tier_labels": BOSS_TIER_LABELS}

# 2026-07-29: cross-link entity mentions (e.g. a character's gear, other characters) inside
# description text so they're clickable, per Wes's request. Deliberately conservative about
# what counts as a linkable name, since a naive "does this substring appear" check would
# produce constant false-positive links -- this corpus has items literally named things like
# "Hole" and "Shield" (real entities, seen live on Donut's own skill list), and linking every
# incidental use of a common word would be worse than no linking at all. Rules:
#   - Multi-word names (e.g. "Def Leppard cart") are always eligible -- collision with
#     ordinary prose is essentially impossible.
#   - Single-word names are only eligible if capitalized, at least 4 characters, and not a
#     common English word (stoplist below) -- filters out exactly the "Hole"/"Shield" case
#     while still allowing genuine single-word proper nouns ("Carl", "Donut").
# Matching is done on the RAW text with word-boundary regexes, longest name first, tracking
# claimed character spans so a shorter entity name can never match inside a longer one's
# already-claimed span (no nested/double links regardless of substring relationships between
# entity names -- e.g. "Carl" is a substring of "Carl's Book of Boom" but the two never
# collide). HTML-escaping is applied only when reassembling the final string, exactly once
# per character, so this is safe against book text containing literal "<"/"&"/etc.
_LINKIFY_STOPWORDS = {
    "the","a","an","and","or","of","in","on","at","to","for","with","by","is","was","were",
    "are","this","that","it","he","she","they","i","you","we","his","her","its","their",
    "him","them","us","our","your","my","me","be","been","being","as","but","if","so","not",
}

def linkify_text(text: str, exclude_id: int, db: Session) -> str:
    if not text:
        return ""
    rows = db.query(Entity.id, Entity.name, Entity.slug).filter(Entity.id != exclude_id).all()
    candidates = []
    for eid, name, slug in rows:
        if not name:
            continue
        name = name.strip()
        if not name:
            continue
        if " " not in name:
            if len(name) < 4 or name.lower() in _LINKIFY_STOPWORDS or not name[0].isupper():
                continue
        candidates.append((name, slug))
    candidates.sort(key=lambda x: -len(x[0]))

    claimed = []   # list of (start, end) already-linked spans
    matches = []   # list of (start, end, slug)

    def overlaps(s1, e1, s2, e2):
        return s1 < e2 and s2 < e1

    for name, slug in candidates:
        pattern = re.compile(r'\b' + re.escape(name) + r'\b')
        for m in pattern.finditer(text):
            s, e = m.span()
            if any(overlaps(s, e, cs, ce) for cs, ce in claimed):
                continue
            claimed.append((s, e))
            matches.append((s, e, slug))

    if not matches:
        return str(escape(text))

    matches.sort(key=lambda x: x[0])
    out = []
    pos = 0
    for s, e, slug in matches:
        out.append(str(escape(text[pos:s])))
        out.append(f'<a href="/entity/{slug}" class="text-gold hover:underline underline-offset-2">{escape(text[s:e])}</a>')
        pos = e
    out.append(str(escape(text[pos:])))
    return "".join(out)

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
    # "Notable Entities" (2026-07-29, round 8): the site is an entity knowledge base where
    # crawler is one entity_type among 11 (crawler/npc/mob/item/ability/location/floor/faction/
    # deity/media/other) -- 257 of 1948 entities are crawlers. Round 6 restricted this query to
    # crawler-only because the heading was "Notable Contestants"/"Notable Crawlers", which
    # falsely implied a crawler-specific section. Now that the heading and eyebrow line are
    # generic ("Notable Entities"/"Entity Knowledge Base"), a mix of entity_type values here is
    # correct and matches the site's actual scope -- reverted to no entity_type filter.
    featured_q = db.query(Entity).filter(Entity.is_major==True, Entity.image_url.isnot(None))
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
           major_only: bool=Query(False), boss_tier: Optional[str]=Query(None),
           page: int=Query(1,ge=1), per_page: int=Query(48,ge=1,le=100),
           db: Session=Depends(get_db)):
    ctx = base_context(request, db)
    max_floor = ctx["max_floor"]
    query = db.query(Entity)
    if entity_type and entity_type in [t.value for t in EntityTypeEnum]:
        query = query.filter(Entity.entity_type==entity_type)
    if q:
        query = query.filter(or_(Entity.name.ilike(f"%{q}%"), Entity.summary.ilike(f"%{q}%")))
    if major_only: query = query.filter(Entity.is_major==True)
    if boss_tier and boss_tier in [t.value for t in BossTierEnum]:
        query = query.filter(Entity.boss_tier==boss_tier)
    if max_floor is not None:
        query = query.outerjoin(ChapterFloor, Entity.first_chapter_id==ChapterFloor.chapter_id).filter(or_(Entity.first_chapter_id==None, ChapterFloor.floor_number<=max_floor))
    total = query.count()
    entities = query.order_by(Entity.is_major.desc(), Entity.name).offset((page-1)*per_page).limit(per_page).all()
    ctx.update({"entities": entities, "entity_type": entity_type, "entity_type_labels": ENTITY_TYPE_LABELS,
                "q": q, "major_only": major_only, "boss_tier": boss_tier, "boss_tier_order": BOSS_TIER_ORDER,
                "page": page, "per_page": per_page,
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
    # 2026-07-30: previously rendered rel.relation_type verbatim regardless of which side
    # of the pair this entity is on, which reads backwards for asymmetric relationships (an
    # entity guarded by a Bopca Protector would have shown "Guards -> Bopca Protector" instead
    # of "Guarded by -> Bopca Protector"). Now resolved per-row via _pretty_relation() before
    # it ever reaches the template. See RELATION_REVERSE_LABELS above.
    relationships_as_a = db.query(EntityRelationship, Entity).join(Entity, EntityRelationship.entity_b_id==Entity.id).filter(EntityRelationship.entity_a_id==entity.id).all()
    relationships_as_b = db.query(EntityRelationship, Entity).join(Entity, EntityRelationship.entity_a_id==Entity.id).filter(EntityRelationship.entity_b_id==entity.id).all()
    relationships = (
        [{"related_entity": other, "label": _pretty_relation(rel.relation_type, True), "evidence": rel.evidence} for rel, other in relationships_as_a]
        + [{"related_entity": other, "label": _pretty_relation(rel.relation_type, False), "evidence": rel.evidence} for rel, other in relationships_as_b]
    )

    # Verified stats/skills only -- see seeder/verify_stats.py. A row is trustworthy
    # for public display only once BOTH independent verification passes confirmed it;
    # unverified/flagged rows are never shown on the page.
    #
    # 2026-07-29: floor_id has been 100% populated on both tables since extraction, but this
    # query never used it -- it pulled every verified row regardless of floor, so an entity
    # whose stat was mentioned at multiple points in the story (a level-up, a different scene)
    # showed ALL of those values stacked on one page with no indication of when each applied.
    # Verified example: Princess Donut's CHA showed 2, 37, 39, 41, 276 simultaneously (floors
    # 1 and 6). Fixed with DISTINCT ON: per stat_name/skill_name, keep only the row from the
    # highest floor_number at or before the current spoiler-shield position (or highest overall
    # if the shield is off) -- i.e. the most recently known value as of where the reader is in
    # the story, same floor-awareness already applied to passages/images on this page.
    stat_rows = db.execute(text("""
        SELECT stat_name, value, value_type, reason, source_passage_id FROM (
            SELECT DISTINCT ON (s.stat_name) s.stat_name, s.value, s.value_type, s.reason,
                   s.source_passage_id, f.floor_number
            FROM entity_stats s
            JOIN floors f ON f.id = s.floor_id
            WHERE s.entity_id = :eid AND s.verify_pass1 = true AND s.verify_pass2 = true
              AND (:max_floor IS NULL OR f.floor_number <= :max_floor)
            ORDER BY s.stat_name, f.floor_number DESC
        ) latest
        ORDER BY stat_name
    """), {"eid": entity.id, "max_floor": max_floor}).fetchall()
    skill_rows = db.execute(text("""
        SELECT skill_name, level, value_type, reason, source_passage_id FROM (
            SELECT DISTINCT ON (s.skill_name) s.skill_name, s.level, s.value_type, s.reason,
                   s.source_passage_id, f.floor_number
            FROM entity_skills s
            JOIN floors f ON f.id = s.floor_id
            WHERE s.entity_id = :eid AND s.verify_pass1 = true AND s.verify_pass2 = true
              AND (:max_floor IS NULL OR f.floor_number <= :max_floor)
            ORDER BY s.skill_name, f.floor_number DESC
        ) latest
        ORDER BY skill_name
    """), {"eid": entity.id, "max_floor": max_floor}).fetchall()

    # Curate aliases: show a handful up front, rest behind a "+N more" toggle in the template
    aliases = entity.aliases or []
    aliases_shown = aliases[:8]
    aliases_rest = aliases[8:]

    # 2026-07-29: cross-link other known entities (gear, other characters, etc.) mentioned
    # in this entity's own description text -- see linkify_text() above for the matching
    # rules and why a naive substring check isn't safe on this corpus.
    linked_persona_html = linkify_text(entity.persona_text, entity.id, db) if entity.persona_text else None
    linked_summary_html = linkify_text(entity.summary, entity.id, db) if (not entity.persona_text and entity.summary) else None

    ctx.update({"entity": entity, "passages_by_type": passages_by_type,
                "passage_type_order": ["physical","personality","ability","backstory","action","other"],
                "relationships": relationships, "first_book": entity.first_book, "first_chapter": entity.first_chapter,
                "entity_type_labels": ENTITY_TYPE_LABELS,
                "hidden_count": total_passage_count-len(passages), "visible_count": len(passages),
                "total_passage_count": total_passage_count, "current_appearance": current_appearance,
                "stat_rows": stat_rows, "skill_rows": skill_rows,
                "aliases_shown": aliases_shown, "aliases_rest": aliases_rest,
                "linked_persona_html": linked_persona_html, "linked_summary_html": linked_summary_html})
    return templates.TemplateResponse("entity_detail.html", ctx)

@app.get("/api/search")
def search_api(q: str=Query(...,min_length=2), db: Session=Depends(get_db)):
    results = db.query(Entity.id, Entity.name, Entity.slug, Entity.entity_type, Entity.image_url).filter(Entity.name.ilike(f"%{q}%")).order_by(Entity.is_major.desc(), Entity.name).limit(10).all()
    return [{"id": r.id, "name": r.name, "slug": r.slug, "type": r.entity_type.value, "image_url": r.image_url} for r in results]
