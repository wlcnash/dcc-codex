"""SQLAlchemy models for DCC Codex."""
from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Integer, String, Text, ARRAY, UniqueConstraint, JSON
from sqlalchemy.orm import DeclarativeBase, relationship
import enum

class Base(DeclarativeBase):
    pass

class EntityTypeEnum(str, enum.Enum):
    character="character"; creature="creature"; item="item"; location="location"
    floor="floor"; ability="ability"; faction="faction"; other="other"

class PassageTypeEnum(str, enum.Enum):
    physical="physical"; personality="personality"; backstory="backstory"
    ability="ability"; action="action"; other="other"

class Book(Base):
    __tablename__="books"
    id=Column(Integer,primary_key=True); title=Column(String(255),nullable=False)
    book_number=Column(Integer,nullable=False); royal_road_url=Column(Text)
    created_at=Column(DateTime,default=datetime.utcnow)
    chapters=relationship("Chapter",back_populates="book")
    appearances=relationship("EntityAppearance",back_populates="book")

class Chapter(Base):
    __tablename__="chapters"
    id=Column(Integer,primary_key=True)
    book_id=Column(Integer,ForeignKey("books.id",ondelete="CASCADE"),nullable=False)
    chapter_number=Column(Integer,nullable=False); chapter_title=Column(String(512))
    url=Column(Text); raw_text=Column(Text,nullable=False); word_count=Column(Integer)
    scraped_at=Column(DateTime,default=datetime.utcnow)
    __table_args__=(UniqueConstraint("book_id","chapter_number"),)
    book=relationship("Book",back_populates="chapters")
    passages=relationship("Passage",back_populates="chapter")

class Entity(Base):
    __tablename__="entities"
    id=Column(Integer,primary_key=True); name=Column(String(255),nullable=False,unique=True)
    slug=Column(String(255),nullable=False,unique=True)
    entity_type=Column(Enum(EntityTypeEnum,name="entity_type"),nullable=False)
    aliases=Column(ARRAY(Text)); first_book_id=Column(Integer,ForeignKey("books.id"))
    first_chapter_id=Column(Integer,ForeignKey("chapters.id")); summary=Column(Text)
    persona_text=Column(Text); image_url=Column(Text); image_prompt=Column(Text)
    image_source_passages=Column(ARRAY(Text)); is_major=Column(Boolean,default=False)
    created_at=Column(DateTime,default=datetime.utcnow)
    updated_at=Column(DateTime,default=datetime.utcnow,onupdate=datetime.utcnow)
    first_book=relationship("Book",foreign_keys=[first_book_id])
    first_chapter=relationship("Chapter",foreign_keys=[first_chapter_id])
    passages=relationship("Passage",back_populates="entity")
    appearances=relationship("EntityAppearance",back_populates="entity",order_by="EntityAppearance.book_id")
    relationships_as_a=relationship("EntityRelationship",foreign_keys="EntityRelationship.entity_a_id",back_populates="entity_a")
    relationships_as_b=relationship("EntityRelationship",foreign_keys="EntityRelationship.entity_b_id",back_populates="entity_b")

class EntityAppearance(Base):
    """Per-book image for an entity. One row per (entity, book) pair.
    Physical passages from THIS book drive the image prompt.
    Mordecai looks like a cat in Book 1, something else later."""
    __tablename__="entity_appearances"
    id=Column(Integer,primary_key=True)
    entity_id=Column(Integer,ForeignKey("entities.id",ondelete="CASCADE"),nullable=False)
    book_id=Column(Integer,ForeignKey("books.id",ondelete="CASCADE"),nullable=False)
    image_url=Column(Text); image_prompt=Column(Text); image_source_passages=Column(ARRAY(Text))
    created_at=Column(DateTime,default=datetime.utcnow)
    __table_args__=(UniqueConstraint("entity_id","book_id"),)
    entity=relationship("Entity",back_populates="appearances")
    book=relationship("Book",back_populates="appearances")

class Passage(Base):
    __tablename__="passages"
    id=Column(Integer,primary_key=True)
    entity_id=Column(Integer,ForeignKey("entities.id",ondelete="CASCADE"),nullable=False)
    chapter_id=Column(Integer,ForeignKey("chapters.id",ondelete="CASCADE"),nullable=False)
    passage_text=Column(Text,nullable=False)
    passage_type=Column(Enum(PassageTypeEnum,name="passage_type"),nullable=False,default=PassageTypeEnum.physical)
    context_before=Column(Text); context_after=Column(Text); char_offset=Column(Integer)
    created_at=Column(DateTime,default=datetime.utcnow)
    entity=relationship("Entity",back_populates="passages")
    chapter=relationship("Chapter",back_populates="passages")

class EntityRelationship(Base):
    __tablename__="entity_relationships"
    id=Column(Integer,primary_key=True)
    entity_a_id=Column(Integer,ForeignKey("entities.id",ondelete="CASCADE"),nullable=False)
    entity_b_id=Column(Integer,ForeignKey("entities.id",ondelete="CASCADE"),nullable=False)
    relation_type=Column(String(100),nullable=False); evidence=Column(Text)
    chapter_id=Column(Integer,ForeignKey("chapters.id"))
    __table_args__=(UniqueConstraint("entity_a_id","entity_b_id","relation_type"),)
    entity_a=relationship("Entity",foreign_keys="[EntityRelationship.entity_a_id]",back_populates="entity_a")
    entity_b=relationship("Entity",foreign_keys="[EntityRelationship.entity_b_id]",back_populates="entity_b")
    chapter=relationship("Chapter")

class PipelineRun(Base):
    __tablename__="pipeline_runs"
    id=Column(Integer,primary_key=True); step=Column(String(50),nullable=False)
    status=Column(String(20),nullable=False,default="running")
    items_processed=Column(Integer,default=0); items_failed=Column(Integer,default=0)
    error_detail=Column(Text); meta=Column(JSON)
    started_at=Column(DateTime,default=datetime.utcnow); finished_at=Column(DateTime)
