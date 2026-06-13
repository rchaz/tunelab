# Data flywheels & active learning

The most valuable output of a deployed model isn't its predictions — it's the
data those predictions generate. A **data flywheel** is the loop that turns
served traffic back into training data, so the system improves the longer it
runs. In tunelab this is a first-class capability, not a footnote.

## The loop

```
serve → log every prediction → collect feedback → curate → retrain a
challenger → promote if it beats the champion → serve …
```

Each turn, three kinds of feedback accrue:

- **Explicit** — a human corrects a label (a support agent re-routes a ticket).
  Highest quality, lowest volume.
- **Automated** — a programmatic verifier scores the output. The distiller's
  grounding gate is one: it can say "this compression invented a value" with no
  human. Downstream success signals (did the agent's task succeed?) are another.
- **Structural** — free labels that fall out of the architecture itself. In a
  cascade, whenever a case escalates to the frontier tier, the frontier's answer
  is a training label for the cheaper tiers *on exactly the inputs they found
  hard*. The cascade mines its own training set.

## Active learning: label the right things

You can't label everything, so label what teaches the most. **Uncertainty
sampling** — prioritize the inputs the model was least confident on — is the
cheap, effective default, and a cascade gives it to you for free: the cases
that escalated are precisely the low-confidence ones. Routed-to-frontier traffic
*is* your active-learning queue.

## The trap: feedback is biased

The cases that get feedback (escalations, complaints, audits of low-confidence
calls) are **not** a random sample of traffic — they over-represent the hard
tail. Train naively on them and you optimize for a distribution you don't
actually serve, and your measured "accuracy" is pessimistic and unrepresentative.
Two defenses, both in tunelab's `flywheel.py`:

1. **A uniform random audit slice** — a small fixed fraction of *all* traffic
   gets gold feedback regardless of confidence. That slice, not the biased
   feedback pile, is your honest accuracy estimate. (Measured example: biased
   feedback read 0.72 while the audit slice read 0.94 — same system.)
2. **Report tiers separately**; never average the audit slice into the
   escalation pile.

## When does the flywheel turn? (triggers)

Retraining has a cost, so it fires on a rule, not a whim: enough new curated
labels, measured drift (a confidence-distribution shift — PSI), or an accuracy
floor breach on the audit slice. tunelab models this as a **MAPE loop**
(Monitor–Analyze–Plan–Execute): the script monitors and analyzes and emits a
*retrain manifest*; a human approves; training executes; evaluation adjudicates
a **challenger** against the **champion** on a fresh, never-used slice. The
challenger ships only if it beats a pre-registered bar — otherwise the champion
stays. Repeated no-promotions lengthen the interval, so the system settles
instead of churning.

## The discipline that separates this from AutoML slop

A self-improving loop will happily consume its own test sets and declare
victory. The guardrails: **pre-registered promotion bars**, **rolling frozen
eval slices each used exactly once**, and **never training on holdout ids**.
The flywheel is only as trustworthy as the eval hygiene wrapped around it.
