# Label ceiling & annotator agreement

The label ceiling is the highest accuracy any model can *meaningfully* reach on
a task — and it's set by your labels, not your model. If two competent labelers,
working independently, only agree X% of the time, then the "right answer" itself
is X% fuzzy, and a model graded against one labeler's labels can't truly exceed
X%. Past that point it isn't getting more correct; it's matching one annotator's
coin-flips.

## Why agreement caps accuracy

A test score measures one thing: does the model match the label. When the label
is the verdict of a careful labeler, matching it means being right. But on
genuinely ambiguous inputs, careful labelers *disagree* — and no fact decides
which one is correct. The model is then graded against an answer key two experts
couldn't agree on, so its ceiling is exactly how often those experts agree: the
**inter-annotator agreement**.

A corollary worth posting on the wall: **a model that scores *above* the ceiling
is a red flag, not a triumph.** It almost always means leakage, or that the model
has overfit one annotator's idiosyncrasies — not that it discovered a truth two
humans missed.

## How to measure it

Have a *second* competent labeler independently label a held-out slice, then
compare:

- **Raw agreement** — the fraction of items where the two labels match. Simple,
  and what tunelab reports.
- **Chance-corrected agreement (Cohen's κ)** — discounts the agreement you'd get
  by luck. With 5 balanced classes, random labeling already agrees ~20% of the
  time, so a raw 0.95 is genuinely high; with 2 classes, raw 0.95 is less
  impressive (chance is 50%). Reach for κ when classes are few or imbalanced.

The second labeler can be a different strong model, not only a human — fast and
cheap, and good enough to *locate* the fuzzy classes even when it isn't gospel.

## Read it per class, not just overall

The headline number hides where the fuzz lives. Measured in tunelab on a 5-way
support-ticket eval (n=465), Opus 4.8 vs GPT-5.5 agreement was **0.951
overall** — but per class: feature_request 1.00, shipping 0.99, billing 0.97,
bug_report 0.96, **account_access 0.84**. Four classes are crisp; one is soft,
and nearly all the disagreement is a single boundary — account_access↔bug_report
(is a failed login an access problem or a malfunction?) accounts for 15 of the 23
disagreements. That tells you precisely where extra labels *won't* help.

## What to do with it

- **Stop chasing points above the ceiling.** A classifier at 0.942 against a
  0.951 ceiling is ~1 point away — effectively done. The remaining gap is label
  noise; squeezing it is rarely worth the effort.
- **Don't densify an irreducible boundary.** If a class is fuzzy because the
  world is fuzzy, more training data won't fix it. Resolve genuinely-ambiguous
  inputs by [routing them](calibration-and-selective-prediction.md) to a stronger
  model (or a human), not by forcing a label.
- **Fix inconsistency, accept ambiguity.** Some disagreement is *inconsistent*
  labeling — two annotators using different conventions — and a single written
  rule plus a re-label pass fixes it. The rest is *irreducible*; accept it and
  route.

Measure the ceiling *before* you read any model score: it turns "we're 5 points
from perfect" into "we're 1 point from the best achievable," which are very
different project plans. And measure it on a [real-labeled anchor
set](synthetic-eval-and-circularity.md), not on data the models also authored —
otherwise the ceiling is circular too.
