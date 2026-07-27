-- DCC Codex Database Schema
-- Dungeon Crawler Carl description-first compendium

CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for full-text search

-- ─────────────────────────────────────────────
-- Books & Chapters (source material)
-- ─────────────────────────────────────────────

CREATE TABLE books (
    id          SERIAL PRIMARY KEY,
    title       VARCHAR(255) NOT NULL,
    book_number INTEGER NOT NULL,
    royal_road_url TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chapters (
    id              SERIAL PRIMARY KEY,
    book_id         INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter_number  INTEGER NOT NULL,
    chapter_title   VARCHAR(512),
    url             TEXT,
    raw_text        TEXT NOT NULL,
    word_count      INTEGER,
    scraped_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (book_id, chapter_number)
);

CREATE INDEX idx_chapters_book ON chapters(book_id);

-- ─────────────────────────────────────────────
-- Entities (the compendium entries)
-- ─────────────────────────────────────────────

CREATE TYPE entity_type AS ENUM (
    'character',   -- Named persons, AI beings, aliens
    'creature',    -- Monsters, beasts, enemies
    'item',        -- Weapons, armor, consumables, magical items
    'location',    -- Named places, rooms, areas
    'floor',       -- Dungeon floors
    'ability',     -- Skills, spells, class abilities
    'faction',     -- Groups, organizations, factions
    'other'
);

CREATE TABLE entities (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL UNIQUE,
    slug            VARCHAR(255) NOT NULL UNIQUE,  -- URL-safe name
    entity_type     entity_type NOT NULL,
    aliases         TEXT[],                          -- other names used in books
    first_book_id   INTEGER REFERENCES books(id),
    first_chapter_id INTEGER REFERENCES chapters(id),
    summary         TEXT,                            -- Gemini-generated summary
    image_url       TEXT,                            -- MinIO URL
    image_prompt    TEXT,                            -- prompt used to generate image
    image_source_passages TEXT[],                    -- passages used for image prompt
    is_major        BOOLEAN DEFAULT FALSE,           -- prominent entity
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_slug ON entities(slug);
CREATE INDEX idx_entities_name_trgm ON entities USING gin(name gin_trgm_ops);

-- ─────────────────────────────────────────────
-- Description Passages (the core differentiator)
-- ─────────────────────────────────────────────

CREATE TYPE passage_type AS ENUM (
    'physical',      -- Physical appearance (color, size, shape, materials)
    'personality',   -- Character traits, behavior
    'backstory',     -- History, origin
    'ability',       -- Powers, skills described
    'action',        -- Notable actions
    'other'
);

CREATE TABLE passages (
    id              SERIAL PRIMARY KEY,
    entity_id       INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    chapter_id      INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    passage_text    TEXT NOT NULL,       -- exact text from book
    passage_type    passage_type NOT NULL DEFAULT 'physical',
    context_before  TEXT,               -- ~200 chars before
    context_after   TEXT,               -- ~200 chars after
    char_offset     INTEGER,            -- position in chapter text
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_passages_entity ON passages(entity_id);
CREATE INDEX idx_passages_chapter ON passages(chapter_id);
CREATE INDEX idx_passages_type ON passages(passage_type);
CREATE INDEX idx_passages_text_trgm ON passages USING gin(passage_text gin_trgm_ops);

-- ─────────────────────────────────────────────
-- Entity Relationships
-- ─────────────────────────────────────────────

CREATE TABLE entity_relationships (
    id              SERIAL PRIMARY KEY,
    entity_a_id     INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_b_id     INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type   VARCHAR(100) NOT NULL,  -- 'ally', 'enemy', 'owner', 'member_of', 'companion', etc.
    evidence        TEXT,                   -- passage that establishes this relationship
    chapter_id      INTEGER REFERENCES chapters(id),
    UNIQUE (entity_a_id, entity_b_id, relation_type)
);

CREATE INDEX idx_relationships_a ON entity_relationships(entity_a_id);
CREATE INDEX idx_relationships_b ON entity_relationships(entity_b_id);

-- ─────────────────────────────────────────────
-- Full-text search view
-- ─────────────────────────────────────────────

CREATE VIEW entity_search AS
SELECT
    e.id,
    e.name,
    e.slug,
    e.entity_type,
    e.summary,
    e.image_url,
    e.is_major,
    b.title AS first_book,
    b.book_number,
    COUNT(DISTINCT p.id) FILTER (WHERE p.passage_type = 'physical') AS physical_passage_count,
    COUNT(DISTINCT p.id) AS total_passage_count
FROM entities e
LEFT JOIN books b ON e.first_book_id = b.id
LEFT JOIN passages p ON p.entity_id = e.id
GROUP BY e.id, e.name, e.slug, e.entity_type, e.summary, e.image_url, e.is_major, b.title, b.book_number;

-- ─────────────────────────────────────────────
-- Seed data: DCC book list
-- ─────────────────────────────────────────────

INSERT INTO books (title, book_number, royal_road_url) VALUES
    ('Dungeon Crawler Carl',                              1, 'https://www.royalroad.com/fiction/12518/dungeon-crawler-carl'),
    ('Carl''s Doomsday Scenario',                         2, 'https://www.royalroad.com/fiction/12518/dungeon-crawler-carl'),
    ('The Dungeon Anarchist''s Cookbook',                 3, 'https://www.royalroad.com/fiction/12518/dungeon-crawler-carl'),
    ('The Gate of the Feral Gods',                        4, 'https://www.royalroad.com/fiction/12518/dungeon-crawler-carl'),
    ('The Butcher''s Masquerade',                         5, 'https://www.royalroad.com/fiction/12518/dungeon-crawler-carl'),
    ('The Eye of the Bedlam Bride',                       6, 'https://www.royalroad.com/fiction/12518/dungeon-crawler-carl'),
    ('This Inevitable Ruin',                            7, 'https://www.royalroad.com/fiction/12518/dungeon-crawler-carl'),
    ('A Parade of Horribles',                          8, 'https://www.royalroad.com/fiction/12518/dungeon-crawler-carl');
