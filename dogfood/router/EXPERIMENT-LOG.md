# llm-auto-router — EXPERIMENT-LOG

## 2026-06-11 — tune-decide: level decision
- Decision: **Level 1** (embeddings + classifier) — routing is binary classification over query
  text; the <10ms latency budget (§8) rules out an SLM router by construction.
- Data: rc's real Claude Code transcripts (local, gitignored): 5,294 events sampled from ~90k
  session files → 1,747 unique queries at 0.95 dedupe → **149 template families** at 0.80
  (top-10 families = 63% of traffic). Origin split: 323 harness-template / 1,424 interactive-ish.
- Pre-registered (rc, 2026-06-11, before any score): **bar = family-held-out macro-F1 ≥ 0.90 AND
  zero observed false-cheap at the shipped threshold**; threshold tuned only on a validation
  slice carved inside training families; receipts computed volume-true over raw events.
- Labels: session-native (model unpinned — Claude Fable 5, 2026-06-11). Rubric frozen after a
  25-family spot-check read with rc: cheap_ok = bounded transforms (classify/summarize/extract/
  enum-select); needs_frontier = planning/decomposition, agentic side effects, open synthesis;
  uncertain → frontier. Family labels propagated to variants; payload-flip check on the top-10
  families: 10/10 safe.

## 2026-06-11 — Round 1: bar FAILED honestly (one look, spent)
- Run: train 920 (74 families) / val 294 / family-held-out 533 (29 families). LR on local static
  embeddings; val threshold sweep: false-cheap 0 at every t, coverage ~90% → t=0.5.
- Result: holdout accuracy 0.998, macro-F1 0.988 (≥0.90 ✓) — but **false-cheap 1/23 ✗**.
- Root cause (verified by reading both templates in full): twin templates sharing ~90% of their
  text — "architecture advisor" asking ONLY for enum selections ({blocks, profile,
  design_profile} → cheap_ok) vs the near-twin whose schema adds `expanded_vision` + 3–5
  designed questions (→ needs_frontier). The labels were CORRECT; the embedding router blurred
  the twins — the discriminating signal is a short response-schema suffix inside a long shared
  template, and the failing family was entirely unseen (grouped split working as intended).
- Lesson: embedding routers have a structural blind spot for template twins that differ in
  output demands, exactly what the false-cheap guardrail exists to catch.

## 2026-06-11 — Round 2: fix per the decision matrix → PASS on fresh test
- Fix: hard-mined 19 boundary variants from unseen sessions (hash-excluded from all prior
  sampling), labeled under the frozen rubric with the boundary calibration made explicit (the
  response-schema line decides, not the shared boilerplate); spent holdout folded into training
  (train2 = 1,472); bar UNCHANGED.
- Fresh test: 60 never-seen families → 1,328 records (274 needs_frontier — harder mix than
  round 1).
- Result (one look): **accuracy 0.999, macro-F1 0.999, false-cheap 0/274 at t=0.5,
  coverage 79.3% — BAR PASSED.**
- Receipts (volume-true over all 7,794 raw events): **39.0% routed cheap at zero observed
  false-cheap**; est. cost $72.06 all-frontier → $53.04 hybrid = **26.4% saving** (estimate
  model: chars/4 input tokens + 300 output tokens/event; Opus 4.8 $5/$25 vs Haiku 4.5 $1/$5 per
  MTok, verified pricing 2026-06). Router latency: 7,794 predictions in ~5.0s including model
  load ≈ 0.64ms/query amortized (~0.2ms marginal) — ~15× under the 10ms §8 budget.
- Verdict: ship-shaped for the recipe; the false-cheap=0 claim is bounded by the 274-record
  frontier sample and the session-native label quality (both stated in the recipe).
