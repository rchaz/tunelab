# Mixture-of-Experts routing (and why a cascade is only "MoE-ish")

People reach for "Mixture of Experts" to describe tunelab's cascade. The
analogy is useful but loose — worth getting right so you don't over-claim.

## MoE: experts inside one model

A Mixture-of-Experts model replaces some dense layers with many parallel
"expert" sub-networks plus a small **gating network** that, per token, routes to
a few experts (e.g. 2 of 8). Only the selected experts run, so the model has a
huge parameter count but activates a fraction of it per token — more capacity at
roughly constant compute. The router is trained *jointly* with the experts;
"expert 4" has no human-meaningful specialty, it's whatever the optimizer
carved out. This is one model, one weights file, one forward pass.

## A tunelab cascade: experts inside a system

A cascade routes *between separate models* (a classifier, a fine-tuned SLM, a
frontier LLM), each a standalone artifact you can train, swap, and evaluate on
its own. The "gating" is an explicit confidence threshold you can read, certify,
and tune — not a learned latent. Routing happens **per request**, not per token.

## Where the analogy holds and breaks

| | MoE | tunelab cascade |
|---|---|---|
| Unit of routing | token | request |
| Experts | learned sub-networks | whole models you chose |
| Router | trained jointly, opaque | explicit calibrated threshold |
| Granularity | inside one forward pass | across separate systems |
| Why it saves | sparse activation | skip expensive tiers on easy inputs |

So: "the cascade is MoE at the system level" is fine as intuition — both pick a
specialist instead of paying for the whole thing every time — but they're
different mechanisms. A cascade is interpretable and modular by construction; an
MoE is a single trained object. tunelab builds cascades because the parts are
independently trainable, evaluable, and swappable by the
[data flywheel](data-flywheels-and-active-learning.md) — properties an MoE's
internal experts don't give you.

## Practical note

If you're tempted to *train* an MoE locally, that's a pretraining-scale effort
and out of tunelab's scope. The system-level "MoE" you can actually build and
maintain on a laptop is the cascade — which is why it's the flagship.
