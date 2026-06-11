# cfpb-complaint-triage — EXPERIMENT-LOG

## 2026-06-10 — tune-decide: level decision
- Decision: **Level 1** (embeddings + classifier) for complaint→product-team triage; Level 2 ruled
  out (output is one of 10 fixed labels — never start at Level 2 for classification); Level 0 run
  first as the cheap probe, expected to miss (static-embedding centroid weakness, benchmarked
  0.44 at ~20/class).
- Interview: input = consumer complaint narrative (one sentence to a few paragraphs); output = one
  of 10 product categories; 3,000 labeled rows on disk (real CFPB narratives, public domain,
  balanced 300/class — gold labels, not teacher labels); motive = demonstrate the cost/latency
  case for routing tickets without a frontier call; hardware = M1 Pro 16GB (Apple Silicon,
  local embeddings, no API key).
- Pre-registered (2026-06-10, before any score was seen): **bar = held-out macro-F1 ≥ 0.70**;
  guardrail = no gold class with F1 < 0.50; metrics = accuracy, macro-F1, per-class P/R/F1,
  confusion matrix read against error costs. Set by user (rc) against the known benchmark ceiling
  of ~0.73 on this data.
- Teacher tier: n/a (gold labels — Path A shape); embeddings = local default
  (static-retrieval-mrl-en-v1 via model2vec, no API key).
- Escalation trigger: macro-F1 < 0.70 after one more-data round → Level 2, or revisit taxonomy.

## 2026-06-10 — Level 0 probe (centroids, 20 examples/class)
- Run (config): `centroid_classify.py` — 200 examples (10×20, seed 42), 2,800 classified, local
  embeddings. Report: `level0_report.md`.
- Result: accuracy **0.437**, macro-F1 0.453; margins tight (median 0.024). Best classes
  mortgage (F1 0.63) / credit_reporting (0.55); worst money_transfer (0.30).
- Predicted-vs-actual: predicted ~0.45 (milestone-1 benchmark) → actual 0.437. The static-centroid
  weakness reproduced almost exactly.
- Lesson: Level 0 is the 10-minute probe, not the answer, for a 10-class fuzzy-boundary task.
  → escalate to Level 1 as planned.

## 2026-06-10 — Level 1 result (vs pre-registered bar)
- Run (config): `train_classifier.py --data data/raw.jsonl --seed 42` — LogisticRegression on
  local static embeddings (script's stated reason: fast, interpretable, calibrated probabilities
  for confidence routing); 3,000 rows, stratified 20% holdout (n=600); refit on all rows before
  saving `classifier.joblib`. Wall time: seconds; embedding 3,000 texts <1s on M1 Pro CPU.
- Result: **held-out accuracy 0.730, macro-F1 0.730 — PASSES the bar (≥ 0.70)**. Guardrail holds:
  weakest classes money_transfer_or_service F1 0.59, checking_savings_account 0.61 — both ≥ 0.50.
  Strongest: student_loan 0.89, mortgage 0.83, debt_collection 0.82.
- Confusion reading: the money_transfer ↔ checking_savings pair carries the bulk of the errors —
  domain-overlapping categories (a disputed transfer often IS a checking-account complaint).
  Cost-wise both route to adjacent teams; harmless cell.
- Confidence routing sweep (fresh 80/20 split, threshold on held-out predictions):
  ≥0.5 keeps 96% at 0.721 · ≥0.6 keeps 90% at 0.744 · ≥0.8 keeps 80% at 0.789. The curve is the
  product knob; the OpenAI embedding upgrade (`--backend openai`) lifts the whole curve
  (benchmark: +3 pts at the classifier level).
- Cost receipts: $0.00 — no API key anywhere in the pipeline; training is seconds on CPU;
  inference is sub-millisecond after a one-time ~125MB embedding-model download.
- Decision: **ship-shaped**. Verdict = Level 1 confirmed; LoRA never needed for this task.
  test-set note: the 20% holdout steered nothing (single look) but raw.jsonl is now partially
  spent for future rounds — carve fresh test data from new complaints for any re-eval.
