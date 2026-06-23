# Recipe 4 — Cheap ticket triage (replacing a frontier classifier)

**Who this is for:** you pay a frontier model (GPT-4o, Claude, etc.) to sort support tickets into a handful of fixed buckets — every ticket, every day — and you want that to run free and local without losing accuracy. The catch this recipe is really about: **you don't have a clean labeled dataset to train on, and the one you think you have is lying to you.**

**The plain idea.** For fixed-label triage the *model* is the easy part — a free classifier on embeddings usually clears the bar (that's [Recipe 1](01-hybrid-cascade.md)). The hard part is the **data** and the **eval**. This recipe is the honest, dead-ends-kept-in story of going from "150 tickets and a fake 100% score" to a free classifier that matches a frontier model on real tickets — and of three traps that each looked like success: leakage that faked a perfect score, a fine-tune that won on a tiny test set and lost on a big one, and an eval that graded its own homework.

This is a real run (2026-06-22, M1 Pro 16GB). Unlike Recipes 1–3, it is **not** $0: dataset generation was free (session-native Opus 4.8, no API key), but the winning embedding ran through OpenAI (~$0.013 per 1,000 tickets) and the label audit used one GPT-5.5 pass. Every number below is reproduced from the run artifacts; the full decision trail is in [`dogfood/ticket-triage/EXPERIMENT-LOG.md`](../dogfood/ticket-triage/EXPERIMENT-LOG.md).

> **Jargon, once:** *distillation* = using a strong "teacher" model's outputs as training data for a small "student." *Label ceiling* = the best accuracy any model can reach when the labels themselves are fuzzy (see [label ceiling](../concepts/label-ceiling-and-annotator-agreement.md)).

## The first score was a perfect 1.000 — which is exactly why it was wrong

The starting data was `tickets.jsonl`: 150 rows, balanced 30 per class. Train a classifier, hold out 20%, score it:

| Check | Accuracy |
|---|---|
| Single 20% holdout | 1.000 |
| 5-fold cross-validation | 1.000 ± 0.000 |

Ship it? No. That's **leakage**. Those 150 rows are only **25 unique texts** (5 per class), each repeated ~6× on average. Identical tickets land in both the training and the test half, so the model is "graded" on examples it already memorized. The 100% measures recall of a lookup table, not skill.

Dedup to the 25 unique tickets and cross-validate honestly:

> **Deduped (25 unique), 5-fold CV: accuracy 0.560 ± 0.233, macro-F1 0.480 ± 0.268** — folds swinging between 0.40 and 1.00.

*That's* the real number, and it says the blocker isn't the model — it's that 5 examples per class is nowhere near enough to learn from or to measure on. The lesson is the first rule of this whole repo: **a perfect score is an alarm, not a trophy.** Dedup before you trust any held-out number (leakage is [the #1 way evaluations lie](02-llm-auto-router.md)).

## Build the data from scratch (the teacher you already pay for)

No clean labels? The frontier model you're already paying *is* your teacher — that's [distillation](../concepts/distillation.md). Here there was no usable production log, so the dataset was generated from scratch by **session-native Opus 4.8** — no API key, $0:

- **Fan-out:** 25 Opus subagents over partitioned scenario slices, ~40 tickets each.
- **Four realism axes:** real-world mess (typos, ALL CAPS, frustration), multi-intent tickets, embedded identifiers/context (order IDs, org IDs), and a wide range of length and emotion.
- **A frozen tie-break rule:** ~20–25% of tickets carry more than one intent; each is labeled by its **primary actionable intent**, under a rule ratified by hand on 5 borderline cases.
- **Spot-check 25 by hand before scaling** — a bad teacher prompt poisons the entire dataset (see [distillation](../concepts/distillation.md)).

That produced **train 1,174 / valid 207**, balanced. But the two *eval* sets matter more than the training set:

- An **independent synthetic eval** (465 tickets) from *separate* generators with a deliberately different distribution (B2B-SaaS/fintech + consumer-mobile/e-commerce/marketplace, fresh phrasing), **hard cross-deduped vs train** (max Jaccard 0.74, 0 dropped). A real generalization test, not a memorization re-run.
- A **25-ticket real-gold set** — the actual human-verified tickets, the one set no model authored. The honesty anchor. (Why you need *both*: [synthetic evals & circularity](../concepts/synthetic-eval-and-circularity.md).)

## Find the ceiling before you chase points

Before scoring any model, ask: how good can *any* model be on these labels? Bring in a second competent labeler — a GPT-5.5 pass over all 465 eval tickets — and measure agreement with Opus:

> **Opus 4.8 vs GPT-5.5 agreement = 0.951 — the label ceiling.**

Per class: feature_request 1.00, shipping 0.99, billing 0.97, bug_report 0.96, **account_access 0.84**. Four classes are crisp; one is soft, and almost all the disagreement is a single boundary — account_access↔bug_report (is a failed login an access problem or a malfunction?), 15 of the 23 total disagreements.

This reframes everything below. ~0.95 is the most a model can score here before it's just matching one annotator's coin-flips. A classifier at 0.942 isn't "5.8 points from perfect" — it's **~1 point under the ceiling.** (See [label ceiling & annotator agreement](../concepts/label-ceiling-and-annotator-agreement.md).)

## The scoreboard (where the surprises live)

Every approach, scored on the *same* 465-ticket independent eval and the 25-ticket real gold, against the 0.951 ceiling:

| Approach | Independent eval (n=465) | Real gold (n=25) | Cost / 1k tickets |
|---|---|---|---|
| Static-embedding classifier | 0.800 | 0.960 | **$0** (local) |
| text-embedding-3-small classifier | 0.927 | 0.960 | ~$0.002 |
| **text-embedding-3-large classifier** | **0.942** | **1.000** | ~$0.013 |
| LoRA fine-tune (Qwen2.5-1.5B) | 0.912 | 0.960 | local + model hosting |
| GPT-5.5 | 0.951 | 1.000 | frontier $ |
| Opus 4.8 (teacher) | 1.000 | 1.000 | frontier $ |

**Read the bottom two rows with suspicion first.** Opus's 1.000 is *definitional* — it wrote the answer key. GPT-5.5's 0.951 is *exactly* the agreement number — *circular*, it measures closeness-to-Opus, not correctness. Neither is a real ranking; that's what the real-gold column and the [circularity concept](../concepts/synthetic-eval-and-circularity.md) are for.

Two findings that are real:

- **The embedding is the lever, not the model.** Changing *only* the embedding — static → 3-small → 3-large — moved accuracy **0.800 → 0.927 → 0.942**, all for fractions of a cent. The biggest, cheapest win wasn't a bigger brain; it was a better way to turn text into vectors. (See [classic ML vs LLM vs SLM](../concepts/ml-vs-llm-vs-slm.md).)
- **Fine-tuning lost.** The LoRA was a real Qwen2.5-1.5B fine-tune — GPU-time, training discipline, a model to host — and it scored **0.912**, *below* a $0.013 embedding swap. On fixed-label triage the classifier wasn't just cheaper; it was more accurate. (Recipe 1's thesis again: the *right tool per slice* wins, not the biggest.)

## Two traps that each looked like a win

**Trap 1 — the small test set lied.** In an earlier round scored on just the 25 real tickets, the LoRA hit **1.000** and the classifier **0.920** — fine-tuning looked like the clear winner. Scaled to 465 independent tickets, the order reversed: classifier **0.927–0.942** > LoRA **0.912**. At n=25 the binomial error bars on 0.92 and 1.00 overlap so heavily the test simply can't separate them — a 2-ticket gap is noise. Don't let a ship decision swing on 25 examples (see [validation vs test](../concepts/validation-vs-test.md)).

**Trap 2 — generalization, not memorization.** The static classifier scored **0.932 on its own held-out validation set but only 0.800 on the independent eval — a 13-point drop.** The weak embedding had leaned on surface phrasing that was consistent within the training generator's style but didn't transfer to the different-distribution eval. The better embedding generalized far better: 3-small went **0.966 → 0.927**, a 4-point drop. A same-distribution holdout would have hidden the gap entirely; the independent-distribution eval is what exposed it.

## What ships

**The text-embedding-3-large classifier** — 0.942 on the independent eval, 1.000 on real gold, ~$0.013 per 1,000 tickets. That's ~1 point under the label ceiling, and effectively free next to the frontier classifier it replaces. Plus **confidence routing**: the residual misses sit on the account_access↔bug_report boundary, where the labels themselves are only ~84% agreed — so route low-confidence tickets to a frontier model instead of forcing a guess. Don't try to "fix" an irreducible boundary with more training data; resolve it with routing (see [calibration & selective prediction](../concepts/calibration-and-selective-prediction.md)).

The LoRA and any bigger small-language-model are **unnecessary** here: they cost more to run and didn't beat the classifier. The ladder did its job — it talked us *out* of fine-tuning.

## What's solid and what isn't

- The robust results are the **rankings**, not the third decimal: better-embedding > fine-tune, and classifier ~at the label ceiling — both hold on the 465-record independent eval.
- The dataset is **100% synthetic** (teacher-mimicry), so the synthetic eval over-credits whoever matches Opus; the real-gold anchor (n=25) is the only cross-model-fair number, and n=25 is *small* — a sanity check, not a certificate.
- The **0.951 ceiling** is itself measured by two models, not a panel of humans — a strong proxy, not gospel. account_access's residual error is mostly irreducible: a routing problem, not a data problem.
- One honest cost: this build **wasn't $0**. Generation was free; the winning embedding and the audit cost a few cents and one OpenAI pass. The real trade is cents-of-embeddings vs dollars-of-frontier per thousand tickets.

Full run log — every dead end, every number, including the leakage autopsy and the LoRA retrain: [`dogfood/ticket-triage/EXPERIMENT-LOG.md`](../dogfood/ticket-triage/EXPERIMENT-LOG.md).
