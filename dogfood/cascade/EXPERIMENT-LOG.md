# cascade-flagship (Recipe 1 v2) — EXPERIMENT-LOG

## 2026-06-12 — tune-decide: level/architecture decision (PLAN-V2 §4)
- Decision: **three-tier hybrid cascade** on Banking77 — tier 1 embeddings+LR (Level 1),
  tier 2 fine-tuned SLM (Level 2; dev at Qwen3-0.6B per PLAN-V2 §16, headline at Qwen3-4B),
  tier 3 frontier with kNN few-shot. Architecture *selection* is the experiment: offline
  counterfactual composition over per-tier predictions (PLAN-V2 §4.3).
- Data: Banking77, canonical PolyAI CSVs (github task-specific-datasets), 10,003 train /
  3,080 test / 77 intents — counts match the source. Local carve: stratified 80/20 seed 42 →
  **train 8,005 / valid 1,998**; the official 3,080-record test is THE test set, untouched.
  Known dataset property (logged before any score): ~14% of train utterances flagged as
  potential label errors (ACL 2022.insights-1.19) — eval adds a noise-aware secondary read.
- Validation contract: valid steers everything (calibration, thresholds, conformal risk
  control, architecture selection); test is consumed ONCE, jointly by the selected
  architecture and the single-tier baselines.

## 2026-06-12 — PRE-REGISTERED BARS — LOCKED (rc confirmed "go" before any score existed)
- **Primary:** on the untouched test set, the validation-selected architecture achieves
  accuracy ≥ best single-tier baseline measured in the same run AND ≥ 0.936 (published
  fine-tuned-BERT anchor, measured under the same label noise).
- **Stretch (reported, not pass/fail):** ≥ 0.95 — flagged as possibly above the noisy-gold
  ceiling.
- **Guardrails:** frontier-tier share ≤ 15% of test traffic at selected thresholds; local
  (tier 1+2) share ≥ 85%; cost+latency per 1k queries vs frontier-only; calibration checks on
  both confidence signals.
- Metrics: accuracy, macro-F1, per-tier coverage/accuracy, Pareto surface, certified operating
  point (conformal risk control), false-escalation/false-local rates.
- Budget: $0 target (session-native tier 3), $5 hard cap (gpt-5.4-mini fallback).

## 2026-06-12 — tier 1 built + scored on the shared valid; composition tooling shipped
- Run: `train_classifier.py --data data/train.jsonl --seed 42` (local static embeddings) →
  internal-holdout 0.86 acc / 0.86 macro-F1; refit-on-all predictions on the shared valid:
  **0.8904** (1,779/1,998). Floor established — ~4.6 points under the 0.936 anchor.
- Threshold preview (raw conf): t=0.8 keeps 90% of traffic at 0.9350. The residual 10% is
  tier 2's job.
- **Calibration finding (logged for the concepts doc):** raw LR confidences are badly
  overconfident here — p25 confidence 0.98 / median 1.00 against 89% actual accuracy. Ranking
  is healthy (accuracy rises monotonically with threshold), probability meaning is not —
  isotonic recalibration in the composer is doing real work, not ceremony.
- Shipped + evidence: `skills/tune-eval/scripts/llm_classify.py` (MLX tier-2 classifier,
  token-margin + mean-logprob confidence, greedy, thinking-disabled, label-space matching —
  Metal smoke test queued behind distiller leg 9) and
  `skills/tune-eval/scripts/cascade_compose.py` (offline counterfactual composition over all
  order-preserving tier subsequences × threshold grid; per-tier isotonic calibration;
  Clopper-Pearson-certified local-risk operating points; accuracy→cost→latency selection).
  `--self-test` PASSES: on a synthetic fixture with an oracle terminal, the composer finds
  a→b→c at 12.9% terminal share / $0.26 per 1k beating frontier-solo at $2.00 per 1k at equal
  accuracy, with certified local-risk UCB 0.0009 ≤ 0.05. The first self-test run also caught a
  real selection bug (accuracy ties resolved arbitrarily instead of by cost) — fixed and
  re-proven.
- Parallelization note: everything above ran CPU-only while distiller leg 9 holds Metal
  (PLAN-V2 §16 rule: one MLX process at a time). Queued for the Metal lane the moment leg 9
  lands: 0.6B llm_classify smoke → 0.6B LoRA on train.jsonl → tier-2 valid predictions.
  Queued otherwise: frontier ceiling probe (session-native), flywheel.py.
