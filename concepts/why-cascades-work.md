# Why cascades work (and when they don't)

A cascade runs the cheap model first and only escalates the inputs it can't
handle confidently. Most production "AI systems" in 2025–26 — including the big
labs' own routers — are cascades or routers underneath. Here's the why.

## The core bet

Real traffic is **mostly easy**. On a typical classification or routing stream,
a large fraction of inputs are unambiguous, and a tiny model nails them. The
hard tail is small but expensive. A cascade pays the big-model cost *only on
the tail*:

```
cost ≈ Σ (fraction reaching tier i) × (cost of tier i)
```

If tier 1 confidently handles 85% at ~$0, you pay frontier prices on 15%, not
100%. The accuracy can *exceed* any single tier, because each input is answered
by the tier best suited to it — provided escalation decisions are right.

## Routing vs. cascading

- **Routing** decides *up front* (from the input alone) which single model
  answers — one model runs.
- **Cascading** runs a model, checks confidence, and escalates on low
  confidence — more than one model may run for one input.

tunelab's hybrid is a cascade: the confidence signal that drives escalation is
the thing that has to be trustworthy (see
[calibration-and-selective-prediction.md](calibration-and-selective-prediction.md)).

## The same shape shows up everywhere

- **Speculative decoding** (e.g. EAGLE-3): a small draft model proposes tokens,
  the big model verifies them in parallel — cheap-first, verify-on-top, exactly
  the cascade pattern at the token level. Same idea, different granularity.
- **Mixture-of-Experts** routes *within* a model to a subset of experts per
  token; a tunelab cascade routes *between* models in a system. The analogy is
  real but loose — see [moe-routing.md](moe-routing.md).

## When a cascade can't help

1. **No cheap tier is good enough on any slice.** If every input is genuinely
   hard, there's nothing to skim off cheaply — you just pay the escalation
   overhead on top of the frontier cost.
2. **Confidence is uncalibrated and can't be fixed.** Escalation needs a
   trustworthy signal; garbage confidence routes hard cases cheap (a
   false-cheap) and wastes the frontier on easy ones.
3. **The ceiling is the gold, not the model.** A cascade can't exceed the
   accuracy its *strongest* tier reaches on the *labels you have*. Measured in
   this repo: on CFPB the frontier itself only hit 72% against the gold labels —
   the labels were the ceiling, so no architecture could do better. Always
   probe the ceiling (best available model on a validation slice) before
   assuming more tiers buy more accuracy.

## The counterintuitive result tunelab measured

On Banking77 (fine-grained 77-way intent), a **$0 logistic regression beat the
frontier model used zero-shot by 6.5 points** (0.883 vs 0.818 on identical
records). The lesson isn't "small always wins" — it's that the *right tool per
slice* wins. A bare frontier call as the escalation tier would have *lowered*
cascade accuracy; the frontier only earns its slot with retrieval/few-shot
help. Build the cascade from what the evidence says each tier is good at, not
from the intuition that bigger is better.
