# Recipe 1 — The hybrid cascade (the flagship)

**Who this is for:** you have a classification or routing task — sorting inputs into categories — and you want *both* higher accuracy and lower cost than a single model gives you.

**The plain idea.** Instead of sending every input to one model, stack three tiers cheapest-first and only escalate the cases the cheaper tier isn't sure about:

1. **Tier 1 — a tiny classifier** (~1ms, free): handles the easy majority it's confident about.
2. **Tier 2 — a small fine-tuned model** (~0.1–1s, free, runs locally): handles the medium-hard cases.
3. **Tier 3 — a frontier model** (seconds, costs money): handles only the genuinely ambiguous tail.

This is a **cascade**. Because each input is answered by the tier that's actually best at it, the whole system can be **more accurate than any single tier — and far cheaper**, since the expensive model only ever sees a small slice of traffic.

The key principle: **the right combination is chosen by measurement, not opinion.** tunelab runs each tier once, then simulates every possible arrangement and picks the best one on the accuracy/cost trade-off.

Everything below is a real run (2026-06-12, M1 Pro 16GB, **$0 API spend**) on [Banking77](https://huggingface.co/datasets/banking77) — a public dataset of 77 fine-grained banking questions, with gold-standard labels nobody here controls.

## The surprising result that shapes everything

On Banking77, measured on the exact same inputs:

| Approach | Accuracy |
|---|---|
| Frontier model, no examples ("zero-shot") | 0.818 |
| **Free classifier** (logistic regression on embeddings) | **0.883** |

The free classifier **beats the frontier model by 6.5 points.** This isn't a fluke — for fine-grained classification, a small model trained on your labels routinely beats a big general model that's only seen the question. Two consequences:

- Using a bare frontier model as the top tier would actually *lower* the cascade's accuracy. It only earns its slot when you **show it the most similar labeled examples first** (called "few-shot"), pulled from the same index Tier 1 already built.
- The real accuracy headroom lives in the **fine-tuned** tier, not the general one.

## Set the bar before you look

Good evaluation writes down what "success" means *before* seeing any score, so results can't be rationalized afterward. The bar for this run:

> The chosen system must beat every single model measured in the same run, **and** reach ≥ 0.936 accuracy (the published fine-tuned-BERT benchmark) on a held-out test set it never trained on.

(Banking77 is known to have ~14% label errors, so the report uses raw accuracy — comparable to the benchmark — plus a noise-aware second read.)

## The build

### 1. Tier 1 — the free classifier (seconds)

```bash
uv run <tunelab>/skills/tune-decide/scripts/train_classifier.py \
  --data data/train.jsonl --model-out tier1.joblib --seed 42
```

Result: **0.883** accuracy. A useful lesson falls out immediately — the classifier's confidence scores are wildly overconfident (it says "100% sure" when it's actually right 89% of the time). The *ranking* is fine, the *numbers* aren't, which is why the composer recalibrates them before using them to decide what to escalate. (See [calibration](../concepts/calibration-and-selective-prediction.md) — does a "90% confident" prediction actually come true 90% of the time?)

### 2. Tier 2 — the fine-tuned model

A small model fine-tuned on the same labels. Develop the whole pipeline with a tiny model first (Qwen3-0.6B — minutes per iteration), then run the real model (Qwen3-4B) once at the end. (See [why cascades work](../concepts/why-cascades-work.md) for why the tier that's *better on a given slice* wins, not the biggest one.)

### 3. Tier 3 — the frontier model, with examples

Reuse Tier 1's index to find the most similar labeled examples and put them in the frontier model's prompt. (Skipping this is a trap — see the surprising result above.)

### 4. Compose — let the system pick the best arrangement

```bash
uv run <tunelab>/skills/tune-eval/scripts/cascade_compose.py \
  --tier t1=tier1_preds.jsonl --conf-key t1=confidence \
  --tier t2=tier2_preds.jsonl --conf-key t2=conf_margin \
  --tier t3=tier3_preds.jsonl \
  --cost t1=0 --cost t2=0 --cost t3=0.002 \
  --max-terminal-share 0.15 --target-local-risk 0.10
```

This simulates every arrangement across every confidence threshold and reports the best operating point — with a **statistical guarantee** on how often the local tiers will be wrong ("conformal" prediction gives a distribution-free, finite-sample bound, not an eyeballed guess).

**The result (154-record probe; 0.6B dev model for Tier 2):**

| Arrangement | Accuracy | Handled locally | $/1k queries |
|---|---|---|---|
| **Tier1 → Tier2 → Tier3 (chosen)** | **0.942** | 87.7% | **$0.25** |
| Tier1 → Tier3 | 0.942 | 87.0% | $0.26 |
| Tier1 alone | 0.883 | 100% | $0 |
| Tier3 alone (frontier) | 0.818 | 100% | $2.00 |
| Tier2 alone (0.6B dev) | 0.513 | 100% | $0 |

The cascade **beats every single tier** (+5.9 over the classifier, +12.3 over the frontier) at **8× lower cost** than frontier-only, keeping 87.7% of traffic local.

How does it beat the frontier-alone score of 0.818 by 12 points? Because Tier 1 answers the 87% it's confident about *very* accurately, and only the genuinely hard residual escalates. Most inputs never reach the frontier at all — they're handled by the tier that's best at them.

The system also did the honest thing: with the weak 0.6B dev model, Tier1→Tier3 *ties* the full three-tier setup, meaning the fine-tuned tier "isn't pulling its weight yet." That's the measurement-driven decision working — it tells you when a tier isn't earning its slot. (The real 4B model is what makes the middle tier count.)

### 5. The flywheel — the system improves itself

Every prediction gets logged. Corrections and escalations become free training data, which feeds `flywheel.py` — it watches for drift and proposes a retrained challenger. **Demonstrated on Banking77:** a model trained on a starved 2,000-record slice (0.813) was beaten by a challenger retrained after one feedback cycle (0.890, **+7.8 points**). See [data flywheels](../concepts/data-flywheels-and-active-learning.md).

## The honest test result (3,080 records, looked at exactly once)

The settings were locked on validation data, then applied **once** to the untouched official test set:

| Approach | Accuracy (official test) |
|---|---|
| Tier 1 alone (classifier) | 0.885 |
| Tier 2 alone (4B fine-tuned) | 0.863 |
| **Cascade Tier1 → Tier2 (fully local, $0, 9.4% escalated)** | **0.901** |

**Against the pre-set bar: one part met, one part missed — reported straight.**

- ✅ The cascade **beats every single tier** (+1.6 over the best). The core thesis — "beats what any single approach can do" — holds on real locked data, fully local, at $0.
- ❌ It does **not** reach the 0.936 benchmark.

Why, honestly: this was the fully-local **two-tier** setup (the frontier Tier 3 was left out at test scale), and Tier 2 was a quick dev-grade fine-tune (0.863 on its own). The machinery is proven; closing the last gap to the benchmark is a "better Tier 2 + add Tier 3" job, both well understood. No spin: the comparative win is earned, the absolute top-end number is not, for this configuration.

## What's solid and what isn't

- The 154-record table is the **machinery demonstration** (every arrangement, statistical certification); the official-test numbers are the **binding receipt**.
- "The cascade beats every single tier" holds on both validation and the locked test.
- Full run log — including the dead ends and bug fixes — is in [`dogfood/cascade/EXPERIMENT-LOG.md`](../dogfood/cascade/EXPERIMENT-LOG.md).
