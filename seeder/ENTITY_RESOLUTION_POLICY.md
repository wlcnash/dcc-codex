# Entity resolution policy -- RIGID, ENFORCED IN CODE, NO EXCEPTIONS

This document, [[feedback_never_guess_read_source]], and `seeder/entity_resolution.py`'s
module docstring together are the binding rule set for how entities get
created and deduplicated in this project. They apply to every agent, every
session, and every future contributor -- there is no "just this once" or
"just for a quick test" exception. If a change would violate a rule below,
the change is wrong, not the rule.

## Why this exists

On 2026-07-27/28, a full audit of the `entities` table found this project had
accumulated real, in-production duplicate records purely because entity
creation only ever matched on EXACT name/alias string equality:

- A 64-group casing/leading-article cluster ("Borant Corporation" /
  "Borant corporation" / "The Borant Corporation").
- A 4-way split of the same dungeon-overseer NPC ("AI", "System AI",
  "dungeon AI", "Syndicate neutral observer AI").
- ~17 inconsistently-named floor entities (the same numbered floor extracted
  under different wording across chapters).
- A 3-way split of the same crawler ("Katia", "Katia Grim",
  "Katia Grimmsdottir").
- A duplicate Princess Donut record under a different title string.

Every one of these traces back to the same single gap: nothing between
extraction and insertion ever asked "does this already exist under different
wording?" `upsert_entity()`'s original implementation only ran
`name = %s OR %s = ANY(aliases)` -- a wording change of any kind produced a
brand-new row, silently, forever, until someone happened to notice by eye.

A first attempt at a fix -- pg_trgm trigram string similarity with a fixed
threshold -- was calibrated against these real cases and rejected. No
threshold cleanly separates true duplicates from true non-duplicates:

- Known TRUE duplicates scored as LOW as 0.233-0.316 similarity
  ("Katia" vs "Katia Grimmsdottir").
- Known genuinely DIFFERENT entities scored as HIGH as 0.454 similarity
  (short/generic names collide on trigrams incidentally).

Any fixed cutoff would have either merged unrelated entities or missed real
duplicates, often both, depending where the line was drawn. This is the same
underlying lesson as [[feedback_never_guess_read_source]]: a name/pattern
match is not evidence of identity. Only reading the actual content is.

## The rule

**No code path may create a new row in `entities` without first calling
`seeder/entity_resolution.resolve_entity()`.** This applies to
`extractor.py`'s `upsert_entity()` (already wired in) and to any future
extraction, backfill, or seeding script that creates entities. There is no
"skip resolution for speed" or "skip resolution for a one-off script"
exception -- a one-off script is exactly how three of the five bug classes
above were created in the first place (ad hoc scripts, one-off inserts, and
early extraction runs that predate this policy).

Resolution is two stages, and both are mandatory together -- neither stage
alone is sufficient (this is the calibration finding above, stated as a
rule):

1. **Blocking (`find_candidates`)**: a loose pg_trgm similarity search
   (threshold 0.15, deliberately below the lowest known true-duplicate score)
   against existing entities of the same `entity_type`. This never decides
   same/different by itself. It only bounds how many expensive judgment
   calls stage 2 makes. Raising or lowering this threshold is a cost/recall
   tuning knob, not a correctness fix -- it must never be treated as "the"
   fix for a bad merge or a missed duplicate. If merges are wrong, the
   problem is in stage 2's grounding or prompt, not this threshold.

2. **Content judgment (`judge_same_entity`)**: an LLM call given BOTH
   entities' actual descriptive content (persona_text, or raw passage text
   if no persona exists yet) -- never just the names -- and asked for a
   grounded `same` / `different` / `uncertain` verdict with a citation of
   what drove the answer. Any failure mode (missing grounding content on
   either side, API error, malformed JSON, an out-of-set verdict) is forced
   to `uncertain`. This function must never return `same` or `different` on
   anything less than a clean, validated response grounded in real content
   on both sides. This is the same "never trust unvalidated LLM output"
   discipline already used in `floors.py`, `permanence.py`, and
   `reclassify_types.py` -- extend it here, don't relax it.

`uncertain` is a first-class, expected outcome, not an error state to be
minimized or bypassed. An entity mention that gets an `uncertain` verdict is
**always** created normally (data is never silently dropped or force-merged
on a guess) **and always** logged to `entity_resolution_log` with
`reviewed = false`. Silently ignoring an uncertain verdict, or treating it as
equivalent to `different`, reintroduces exactly the bug class this exists to
prevent.

## Every resolution attempt is logged, always

`entity_resolution_log` records every `same`, `different`, and `uncertain`
verdict reached, with the candidate compared against, the similarity score,
the full reasoning text, the action taken, and the resulting entity id. This
is not optional instrumentation -- it is what makes every merge (or
non-merge) decision independently verifiable after the fact, per
[[feedback_never_guess_read_source]]'s rule to always show the receipt, not
just the conclusion. A merge or a "this is different" call with no log row
behind it is not a valid resolution outcome under this policy.

To find the current human-review backlog at any time:

```sql
SELECT * FROM entity_resolution_log WHERE verdict = 'uncertain' AND reviewed = false ORDER BY created_at;
```

Reviewing an entry means reading the actual passages/personas for both sides
(per [[feedback_never_guess_read_source]] -- not re-judging from the names
in the log row) and either merging via `merge_entities.py`, leaving the
entities separate, and then setting `reviewed = true` with a `review_note`
either way.

## The `entity_type` taxonomy must stay in sync with the live schema

Reading `extractor.py` for this fix also surfaced a second, unrelated but
real bug: `VALID_ENTITY_TYPES` and the extraction prompt's type list were
still the OLD 8-value taxonomy (`character`, `creature`, `item`, `location`,
`floor`, `ability`, `faction`, `other`) from before an earlier session's
migration to the current 11-value enum (`crawler`, `npc`, `mob`, `item`,
`ability`, `location`, `floor`, `faction`, `deity`, `media`, `other`).
`character` and `creature` no longer exist in the live Postgres enum at all
-- any extraction run that classified an entity as either would have failed
with an invalid-enum-value error on every such entity. This had not yet been
hit only because no extraction had run since the taxonomy migration.

**Rule**: `VALID_ENTITY_TYPES` in `extractor.py` must always be verified
against the live enum before trusting it, not assumed from a prior session's
memory or an older script's constant (e.g. `reclassify_types.py`'s
`ALLOWED_NEW_TYPES`, which still includes `species` -- also now stale, since
`species` was later split into its own `species` table with
`entities.species_id`, but `reclassify_types.py` was a one-time migration
script that already ran and is not on any live code path, so it was left
as-is rather than edited). Verify with:

```sql
SELECT enumlabel FROM pg_enum e JOIN pg_type t ON e.enumtypid = t.oid
WHERE t.typname = 'entity_type' ORDER BY e.enumsortorder;
```

Also fixed in the same pass: `extractor.py` called Gemini as
`gemini-3.5-flash`, while every other seeder script (`floors.py`,
`imager.py`, `reclassify_types.py`) uses `gemini-3.6-flash`. Verified via
`grep` across the repo, not assumed -- this was a real drift, now aligned to
`gemini-3.6-flash` everywhere.

## Applying this policy retroactively

This policy is not just a prevention mechanism for future extraction. Per
the 2026-07-28 directive that created it, it must also be run once against
the existing backlog of same-type near-name-match groups already flagged by
`seeder/audit_alias_collisions.py`, to catch any remaining real duplicates
that predate this fix (Katia and Donut were found and fixed this way before
this module existed; there is no reason to believe they were the only two).
See `entity_resolution_log` entries with `action_taken` values other than
`created_separately` from the retroactive pass for what that run found.
