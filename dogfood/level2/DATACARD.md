# DATACARD — taskloop-triage-json (synthetic, Path C)

- **Provenance:** 100% synthetic. 800 tickets generated 2026-06-10 by 16 session-native batches
  (50 each), each batch assigned a distinct diversity slice (personas × length mix × edge-case
  quotas × rare-category quotas × scenario seed) over the fictional Taskloop SaaS, plus the 25
  spot-check pairs. Inputs authored ticket-first (labels judged after writing, not written
  toward).
- **Teacher:** tier = **session-native** (model unpinned — Claude Code session, Claude Fable 5,
  2026-06-10); labels schema-constrained to the frozen enums at generation time.
- **Labeling prompt (frozen 2026-06-10, after 25-sample spot-check with user rc):** verbatim in
  `data/teacher_prompt.txt`; identical system turn in every training record.
- **Dedupe:** 825 → 825 at char-4-gram Jaccard 0.80 — zero exact or near duplicates. The
  per-batch diversity slicing prevented synthetic collapse outright (the failure mode the ~130%
  overgeneration budget existed for).
- **Splits:** train 659 / valid 83 / test 83; seed 42; stratified by `label` (= category).
- **Label distribution:** bug 154 · billing 138 · how_to 123 · data_integration 107 ·
  account_access 90 · feature_request 88 · cancellation 69 · other 56. Urgency: normal 314 ·
  low 277 · high 194 · critical 40.
- **Known gaps:** (1) teacher label-policy drift at the bug↔data_integration boundary — batch 1
  classed export/sync/API defects as data_integration while the spot-check classed a CSV-export
  timeout as bug; per-item judge eval absorbs this, but category-accuracy on that boundary will
  read slightly noisy. (2) `critical` urgency is rare (40/825 ≈ 5%) — realistic, but the model
  sees few examples. (3) Single fictional product domain; no transfer claim beyond Taskloop-like
  SaaS tickets.
- **Intended use:** narrow internal triage model for the tunelab dogfood; synthetic data from a
  session-native teacher (see the distillation concepts note bundled with the tunelab plugin for
  the provider-ToS discussion — this dataset trains a narrow task model, not a competing
  general model).
