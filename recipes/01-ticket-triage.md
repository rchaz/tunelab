# Recipe 1 — Ticket triage (the tutorial)

**Level 0 → 1 · the canonical tunelab walkthrough · every number below is from a real run
(2026-06-10, M1 Pro 16GB, $0.00 in API spend).**

## The problem

Tickets, emails, or complaints arrive as free text and someone — or some frontier-model call —
routes each one to the right queue. The output is one of N fixed labels. That last sentence is the
whole decision: **fixed labels means classification, and classification never starts at Level 2.**
A frontier model doing this job at, say, 2,000 tickets/day is a four-figure annual bill for a
decision boundary a 10 MB classifier can learn in seconds.

## Why this level

Walking tunelab's ladder top-down (first match wins):

- **Level -1** (better prompt/cheaper tier) — keeps the per-call bill, just smaller. Worth doing
  while you build the real thing, not instead of it.
- **Level 0** (embedding centroids, no training) — the 10-minute probe. Run it first; if your
  buckets are crisp it ends the project right there.
- **Level 1** (embeddings + classifier) — the destination for almost every triage task: a few
  hundred labeled examples, trains in seconds, calibrated confidences for routing.
- **Level 2** (LoRA) — only if the output was never really a fixed label, or Level 1 misses a
  pre-registered bar *after* a more-data round.

## Data source

This walkthrough uses the **CFPB Consumer Complaint Database** — real, public-domain,
consumer-written complaint narratives with product-category labels; the closest public thing to
"support tickets with gold routing labels." 3,000 rows balanced across 10 classes, fetched
without auth from the consumerfinance.gov search API. Substitute your own logged tickets 1:1 —
if a frontier model has been doing your triage, its logged decisions ARE the labels
(that's distillation at its cheapest — see `concepts/distillation.md`).

Record shape throughout: `{"text": "...", "label": "..."}`, one per line.

## Pre-registered bar (set before any score existed)

> **bar:** held-out macro-F1 ≥ 0.70 · **guardrail:** no class with F1 < 0.50
> **metrics:** accuracy, macro-F1, per-class P/R/F1, confusion matrix read against error costs

Macro-F1 (not accuracy) because every class counts equally regardless of size; the bar is written
into EXPERIMENT-LOG.md *first* because a bar chosen after seeing results is a rationalization
(see `concepts/validation-vs-test.md`).

## The walkthrough

### 1. Level 0 probe — centroids, no training (10 minutes)

20 examples per class, average each class into a centroid, classify by nearest centroid:

```bash
uv run <tunelab>/skills/tune-decide/scripts/centroid_classify.py \
  --examples centroid_examples.jsonl --classify the_rest.jsonl --output preds.jsonl
```

**Real result:** accuracy **0.437**, macro-F1 0.453, confidence margins razor-thin
(median 0.024). Predicted beforehand: ~0.45 — the static local embeddings are documented to be
weak at few-shot centroids (0.44 vs 0.62 for a torch-stack model in our benchmark). For crisp
2–4-class problems Level 0 regularly ends the project; for 10 fuzzy classes it's the probe that
says "train the classifier."

### 2. Level 1 — logistic regression on the same embeddings (seconds)

```bash
uv run <tunelab>/skills/tune-decide/scripts/train_classifier.py \
  --data raw.jsonl --model-out classifier.joblib --seed 42
```

The script states its model choice out loud — logistic regression: fast, interpretable,
**calibrated probabilities** (a 0.9 means ~90%, which is what confidence routing needs) — holds
out a stratified 20%, prints honest metrics, then refits on everything before saving.

**Real result (n=600 holdout):**

| metric | value |
|---|---|
| accuracy | **0.730** |
| macro-F1 | **0.730 — bar passed (≥ 0.70)** |
| weakest classes | money_transfer 0.59 · checking_savings 0.61 (guardrail ≥ 0.50 holds) |
| strongest | student_loan 0.89 · mortgage 0.83 · debt_collection 0.82 |

**Read the confusion matrix against costs, not vibes:** here the errors concentrate in
money_transfer ↔ checking_savings — semantically overlapping queues, cheap mistakes. 94% with
all errors in a harmless cell can be shippable while 97% with errors in a costly cell is not.

### 3. Confidence routing — the hybrid that beats both

Predictions under a confidence threshold route to the frontier model; everything else stays local
and free. Swept on a fresh 600-row holdout (real numbers):

| threshold | kept locally | accuracy on kept | routed to frontier |
|---|---|---|---|
| ≥ 0.5 | 96% | 0.721 | 4% |
| ≥ 0.6 | 90% | 0.744 | 10% |
| ≥ 0.8 | 80% | 0.789 | 20% |

The threshold is a product knob — pick it on validation data against *your* error costs. Two
bonuses: the routed cases are exactly the ones worth labeling next, and the whole curve lifts if
you switch the embedding backend to OpenAI (`--backend openai`, +3 pts in our benchmark).

### 4. Ship, log, monitor

- Log inputs/outputs/corrections from day one — they're the next training round, free.
- Re-score monthly on ~100 fresh labeled tickets (the eval pipeline IS the drift monitor).
- Escalate to Level 2 only if a more-data round still misses the bar — and first ask whether the
  *taxonomy*, not the model, is what's wrong.

## Cost receipts (the actual run)

| item | cost |
|---|---|
| API spend, entire pipeline | **$0.00** (local embeddings, no key) |
| embedding 3,000 texts (M1 Pro CPU) | <1 second |
| training | seconds |
| inference | sub-millisecond per ticket |
| one-time download | ~125 MB embedding model |
| vs. frontier triage at 2k tickets/day | whatever you pay today → ~$0 for the kept 80–96% |

Full run log: `dogfood/level1/EXPERIMENT-LOG.md` in the tunelab repo — including the Level 0
prediction-vs-actual and the per-class table.
