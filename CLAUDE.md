# CLAUDE.md — standing rules for anyone (human or agent) changing this pipeline

This is a general-purpose dubbing pipeline, not a tool for one episode.
Every change is judged by one question first: **does it still work on a
video we have never seen, with speakers we have never heard, in both
modes (mix and multitrack)?** The Gelbart episode and its speakers are
the current test material, not the specification.

README.md is the source of truth for how the pipeline works. TODO.md
holds the backlog (problem / evidence / fix / verification). tracker/
is the review board (ids like s16, d5, b13 — put the id in the commit).
RU_INSTRUCTIONS.md is for translation contractors only; code and
README never point readers there.

## 1. Generalize, don't overfit

- No speaker names, episode ids, or run names in `scripts/`. They may
  appear in tests and docs as evidence, never in logic.
- A threshold must be **relative to the input** whenever possible
  (share of the speaker's own level, ratio between stems, fraction of
  the slot), not an absolute number that happened to work on one clip.
  An absolute constant needs a comment with its unit, how it was
  derived, and on which runs.
- A constant tuned on one run is a hypothesis. It becomes a rule only
  after it holds on the three regression sets: `baseline_traps` (2:41,
  fast), `baseline20` (20 min, representative), one full episode
  (currently e1v6). State the numbers for all three in the TODO entry.
- Every heuristic must say what property of the input it relies on
  and what input breaks it ("assumes one mic per speaker", "assumes
  the reference text matches its audio", "assumes the transcript is
  complete"). If the breaking input is plausible for another video,
  add the guard or the doctor check in the same change.
- Before shipping a rule, ask: a two-person podcast with a Russian
  reference and no music stem — does this rule still do the right
  thing, or at least fail loudly?

## 2. Measure first, then change, then measure again

- No synthesis-affecting change without a pre-flight table on
  `baseline_traps`: QC flagged count and median sim vs the previous
  run, clip length / original span before atempo (target 0.95–1.10),
  count of lines needing atempo > 1.15, count of `auto_track` pastes
  and how many contain Whisper-recognizable words. Green on the table
  before anyone listens; the ear decides only after.
- "Verified" means: on which run, which numbers, compared to what.
  "Measured" and "assumed" are different words — use the right one.
- The doctor (`check_pipeline.py`) must report 0 failures on the
  regression sets after every change. `python -m unittest discover
  -s tests` must pass. A pure-logic change gets a test in the same
  commit.
- A fix for a symptom found in one episode is not done until the
  doctor would catch the next occurrence automatically.

## 3. Keep it simple — rules for not over-engineering

- **Deletion is a first-class move.** Before adding a pass, print the
  drop counts of the existing passes on the three regression sets;
  delete any pass that is zero on all three. Adding a seventh
  heuristic to fix what the sixth missed is the pattern to refuse.
- **One rule, one home.** When two stages must agree (slot, voiced,
  freshness, normalization), the rule lives in one function both
  stages import. Never a second copy "for now".
- **Enforce invariants, don't reconstruct state.** If a fact must be
  known (what a clip was sung from, which text an accent file matched),
  record it when it is created (manifest, versioned, full fingerprint).
  Never infer it later from mtimes, old hash schemes, or heuristics.
- **Loud, not quiet.** When a stage cannot decide (speech or laugh,
  skipped or silent, stale or fresh), it does not pick the convenient
  answer and move on — it fails in the doctor with the span and the
  reason, and a human accepts it on record (`doctor_accept.json`).
- **A gate before a subsystem.** The first time a failure mode is
  seen, add the cheapest check that makes it visible (a cap, a doctor
  FAIL). Build the full mechanism only when it recurs or when the
  cheap check blocks real work.
- **Scope check before coding.** A change that touches more than two
  scripts, adds a file, or changes a stored format gets a short design
  note in TODO.md first: the rule in one sentence, what it replaces,
  how it is verified, how it is reverted. If the rule cannot be stated
  in one sentence, it is not understood yet.
- **Comments carry the rule, one line of why, and the tracker id.**
  The investigation narrative goes to `docs/<topic>.md`
  (example: `docs/whisper-gap-skip-leak.md`).
- `main()` and importable pure functions for any script you touch;
  no work at import time.

## 4. Working with the reviewers

- Report changes with numbers, not adjectives. Three lines per
  change: what rule changed, the pre-flight table, what is still open.
- When a reviewer's claim conflicts with the data, say so with the
  measurement — the review notes in TODO.md are full of accepted
  corrections in both directions, and that is how it should stay.
- Never close a tracker item by narrowing its definition. If the fix
  covers less than the item, split the item.
