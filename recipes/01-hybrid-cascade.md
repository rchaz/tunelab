# Recipe 1 — The hybrid cascade (the flagship)

**The canonical tunelab build: a three-tier cascade that beats every single approach, at a
fraction of frontier cost, with a certified error guarantee. Every number below is from a real
run (2026-06-12, M1 Pro 16GB, $0.00 API spend) on Banking77 — public, external gold nobody here
controls.**

## The thesis

Most "should I fine-tune?" questions are really "what's the *right tool for each input*?" — and
the answer is rarely one model. A **cascade** runs the cheapest capable tier first and escalates
only what it can't handle confidently:

1. **Tier 1 — embeddings + logistic regression** (~1ms, ~$0): handles the bulk it's sure about.
2. **Tier 2 — a fine-tuned small LLM** (~0.1–1s, ~$0 local): the medium-confidence residual.
3. **Tier 3 — a frontier model with kNN few-shot** (seconds, $): the genuinely ambiguous tail.

The order is ML → fine-tuned → frontier, and **the composition is chosen by measurement, not
belief.** tunelab runs each tier once, then simulates *every* architecture offline and picks the
best on the accuracy/cost/latency frontier.

## The counterintuitive result that sets the design

On Banking77 (77 fine-grained banking intents), measured on identical records:

| approach | accuracy |
|---|---|
| Frontier model, zero-shot | 0.818 |
| **$0 logistic regression** | **0.883** |

The cheap classifier **beats the frontier model by 6.5 points.** This isn't a fluke — it's the
documented regime for fine-grained classification (zero-shot LLMs trail fine-tuned models by
~19–26 points on Banking77). Two consequences shape everything:

- **Bare zero-shot frontier as the escalation tier would *lower* cascade accuracy** (it's below
  the tier-1 floor). The frontier earns its slot only with **kNN few-shot** — reuse tier 1's
  embedding index to pull the k nearest labeled examples into the prompt.
- The real accuracy headroom lives in the **fine-tuned** tier, not the general one.

## Pre-registered bars (logged before any score existed)

> **Primary:** the validation-selected architecture beats every single-tier baseline measured in
> the same run AND reaches ≥ 0.936 (the published fine-tuned-BERT anchor) on the untouched test
> set. **Stretch (reported):** ≥ 0.95. **Guardrails:** frontier share ≤ 15%; local share ≥ 85%;
> full cost+latency accounting; calibration checks on both confidence signals.

Banking77 carries ~14% flagged label errors (ACL 2022), so the eval reports raw accuracy
(comparable to the anchor, measured under the same noise) and a noise-aware secondary read.

## The build

### 1. Tier 1 — the floor (seconds, $0)

```bash
uv run <tunelab>/skills/tune-decide/scripts/train_classifier.py \
  --data data/train.jsonl --model-out tier1.joblib --seed 42
```

**Real result:** 0.883 on validation. A calibration lesson falls out immediately — the LR's raw
confidences are wildly overconfident (median 1.00 vs 0.89 actual accuracy). The ranking is
sound, the *probabilities* aren't, which is why the composer recalibrates with isotonic
regression rather than thresholding raw scores. See
[concepts/calibration-and-selective-prediction.md](../concepts/calibration-and-selective-prediction.md).

### 2. Tier 2 — the fine-tuned residual

A small LLM fine-tuned on the same labels, scored with `llm_classify.py`, which captures a
**token-margin** confidence (an LLM gives nothing calibrated for free). Develop the whole
pipeline with a tiny model (Qwen3-0.6B — minutes per iteration); run the headline model
(Qwen3-4B) once at the end. (See [concepts/why-cascades-work.md](../concepts/why-cascades-work.md)
for why the tier that's *better on a slice* wins, not the biggest one.)

### 3. Tier 3 — the frontier tail, with kNN few-shot

Reuse the tier-1 embedding index to retrieve the nearest labeled examples and prompt the frontier
model with them. Bare zero-shot is a trap here (see the thesis above).

### 4. Compose — let the system pick the architecture

```bash
uv run <tunelab>/skills/tune-eval/scripts/cascade_compose.py \
  --tier t1=tier1_preds.jsonl --conf-key t1=confidence \
  --tier t2=tier2_preds.jsonl --conf-key t2=conf_margin \
  --tier t3=tier3_preds.jsonl \
  --cost t1=0 --cost t2=0 --cost t3=0.002 \
  --max-terminal-share 0.15 --target-local-risk 0.10
```

This simulates all seven architectures across the threshold grid, calibrates each tier's
confidence, and reports a **conformal-certified operating point** — a distribution-free,
finite-sample guarantee on the local-tier error rate, not an eyeballed threshold.

**Real composition (154-record stratified probe; 0.6B dev tier-2; session-native frontier):**

| architecture | accuracy | local share | $/1k | thresholds |
|---|---|---|---|---|
| **t1→t2→t3 (selected)** | **0.9416** | 87.7% | **$0.25** | t1:0.43, t2:0.60 |
| t1→t3 | 0.9416 | 87.0% | $0.26 | t1:0.43 |
| t1 (solo) | 0.883 | 100% | $0 | — |
| t3 (solo, frontier) | 0.818 | 100% | $2.00 | — |
| t2 (solo, 0.6B dev) | 0.513 | 100% | $0 | — |

**The cascade beats every single tier** (+5.9 over the LR, +12.3 over frontier) at **8× lower
cost than frontier-solo**, keeping 87.7% local — conformal-certified (kept-set error UCB 0.067 ≤
0.10 at 95% confidence). How does it beat frontier-solo by 12 points when frontier-solo is only
0.818? **Selective prediction:** tier 1 answers the 87% it's confident on at high accuracy; only
the low-confidence residual escalates — most traffic is handled by the tier that's *better on
it*, never reaching the frontier.

The architecture search also did the honest thing: t1→t3 *ties* the selected t1→t2→t3 here,
because the 0.6B dev tier-2 is too weak to add much. The system surfaces that — "your fine-tuned
tier isn't pulling its weight yet" — which is the experiment-driven decision working. (The 4B
headline tier-2 is the run that makes the middle tier earn its slot.)

### 5. The data flywheel — the system improves itself

Every prediction from every tier is logged; feedback (explicit corrections, automated verifiers,
and *structural* labels — a frontier escalation is free training data for the tiers below) feeds
`flywheel.py`, which monitors drift, fires retrain triggers, and proposes a champion/challenger
round. **Demonstrated on Banking77:** a champion trained on a starved 2,000-record slice
(0.813) was beaten by a challenger retrained after one feedback cycle (0.890, **+7.8 points**) →
promote. The loop turns, and the lift is the receipt. See
[concepts/data-flywheels-and-active-learning.md](../concepts/data-flywheels-and-active-learning.md).

## Cost receipts

| item | cost |
|---|---|
| API spend, entire build | **$0.00** (local embeddings, session-native frontier) |
| tier-1 train + predict | seconds |
| tier-2 dev (0.6B) train | ~13 min; headline (4B) ~1h |
| cascade serving | 87.7% of traffic at ~$0, ~1ms; frontier tail $0.002/call |
| vs. frontier-only at iso-accuracy | 8× cheaper per 1k queries |

## The official test result (3,080 records, one look) — reported straight

Calibration + the t1→t2 threshold were selected on validation, then applied **once** to the
untouched official test set:

| approach | accuracy (official test) |
|---|---|
| tier-1 solo (LR) | 0.8851 |
| tier-2 solo (4B QLoRA) | 0.8630 |
| **cascade t1→t2 (fully local, $0, 9.4% escalated)** | **0.9013** |

**Against the pre-registered bar (two conditions): one met, one missed.**
- ✅ The cascade **beats every single tier** (+1.6 over the best) — the flagship thesis
  ("exceeds what any single approach can do") holds on real locked gold, fully local at $0.
- ❌ It does **not** reach the 0.936 fine-tuned-BERT anchor.

Why, honestly: this is the fully-local **two-tier** config (the frontier tier-3 was omitted at
test scale), and tier-2 is a quick dev-grade QLoRA (0.863 solo, well under the 0.937 anchor). The
machinery is proven; reaching the absolute anchor is a tier-2-quality + tier-3-inclusion problem,
both well-scoped for the next iteration. No spin: the comparative claim is earned, the absolute
0.95-class number is not, for this config.

## Honest bounds

- The 154-probe composition table above is the **machinery demonstration** (all architectures,
  conformal certification); the official-test numbers just above are the **binding receipt**.
- The cascade-beats-every-tier result is robust on both validation and the locked test.
- Full run log — ceiling probe, flywheel cycle, the official-test consumption, and the bug/gotcha
  fixes — `dogfood/cascade/EXPERIMENT-LOG.md`.
