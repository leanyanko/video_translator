# Listening set — the fixed moments to hear after every pacing change

The ear is the only QC for crumpled phrases, swallowed endings and
unnatural tempo. To spend it well (and to turn "sounds worse" into
"moment 4 regressed"), listen to exactly these moments, in this order,
after every change that touches synthesis, pacing or assembly. Run the
pre-flight table first (CLAUDE.md §2); ears only when it is green.

Regression dirs: `work/exp_traps_v9_scratch` (2:41), `work/exp_bl20_v3_span`
(20 min), `work/exp_e1v7` (full episode). Timecodes are in each clip's
own video.

## Traps (2:41)

| # | Time | What it tests | Known history |
|---|------|---------------|---------------|
| 1 | 0:00–0:25 | Dan's opener: tempo vs lips, three-piece line, sentence boundaries | v7 too fast (15 ch/s); pace7 ok |
| 2 | 0:50–0:52 | Feynman's one-line intro «Встречайте: Дэн Гельбарт» — short clip, name | swallowed in v7 (21 ch/s) |
| 3 | 1:04–1:10 | Dan «Теперь смотрите, смотрите: этот кусочек левитирует» — phrase boundary inside a piece | rendering lottery: crumpled phrase + hole in one rendering (#17) |
| 4 | 1:07 | Misha's overlap «Левитирует, да» — 1 s backchannel over Dan | was «Грузится, да» (whisper misheard) |
| 5 | 1:21–1:27 | Misha's question (long words) | «иттрий-бариевым», «самоё» — translation + ёfication fixed |
| 6 | 1:27–1:33 | Three voices: Dan / Mike «Сверхпроводник?» / Misha «Нет-нет, конечно нет» | tight overlap slots, short clips |
| 7 | 1:41–1:44 | Misha's snicker at the Nobel joke — a PASTE (laugh with a word) | lost once by the speech check; kept via PANNs rule |
| 8 | 2:06–2:12 | Misha «А сверхпроводник по сути диамагнетик…» — reference quality | fine in every run; the control |
| 9 | 2:07–2:09 | Dan «А сверхпроводник — это красиво» — hand-added skipped phrase over Misha | whisper gap (#13) |
| 10 | 2:36–2:41 | Dan's last line «…левитация невозможна… Так, и всё.» — last syllable, clip end | swallowed last word in pace4 |

## baseline20 (20 min)

| # | Time | What it tests |
|---|------|---------------|
| 11 | 1:50 | id 19 — a formerly squeezed line (s16) |
| 12 | 4:45–5:30 | Misha's long lines (ids 55/57) — slow-speaker tempo |
| 13 | 8:04 | id 83 — formerly at the 1.35 cap |
| 14 | 10:10 | id 103 «Да.» — one-word clip |
| 15 | 14:20–14:40 | Mike (ids 164/175) — reference-leak class (LEAD): must open clean |

## Full episode (e1v7)

| # | Time | What it tests |
|---|------|---------------|
| 16 | 22:25–22:55 | The Lenz's-law hole (1345.9–1373.6) — four hand-added lines after a whisper skip (#13) |
| 17 | any laugh paste | 24 pastes: no English words in them |

Record verdicts per moment (ok / regressed / improved) in the TODO entry
of the change under test, next to the pre-flight numbers.
