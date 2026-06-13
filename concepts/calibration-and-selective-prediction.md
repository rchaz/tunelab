# Calibration & selective prediction

A cascade routes each input to the cheapest tier that can be *trusted* on it.
"Trusted" is a number — a confidence — and the whole design lives or dies on
whether that number means what it says. This is the load-bearing idea behind
tunelab's hybrid cascades.

## Confidence ≠ probability

A model can rank its answers well (the ones it's surest about are likeliest
right) and still be **miscalibrated** (its "0.99" is actually right 88% of the
time). Ranking is enough to *order* candidates; calibration is what lets you
set a *threshold* and predict the error rate you'll get.

Measured on Banking77 in this repo (2026-06-12): the tier-1 logistic
regression had median confidence **1.00** while its actual accuracy was
**0.89** — badly overconfident, but monotone (higher confidence really did
mean higher accuracy). So its raw scores can't be thresholded by eye, but they
*can* be calibrated.

## Two practical calibrators

- **Isotonic regression** — fits a monotone step function from raw score →
  P(correct) on a validation set. Non-parametric, robust, the default in
  tunelab's `cascade_compose.py`. It's what turns "1.00 that means 0.89" into a
  usable 0.89.
- **Platt / temperature scaling** — fits a sigmoid (one or two parameters).
  Cheaper, smoother, weaker when the miscalibration isn't sigmoid-shaped.

An LLM tier gives you nothing calibrated for free: token log-probs aren't
probabilities of correctness. tunelab derives a raw signal (the **token-margin**
— the gap between the top-1 and top-2 token while it writes the label) and
calibrates *that* the same way. Raw margins are never thresholded directly.

## Selective prediction: abstain, don't guess

A selective classifier may answer or **abstain** (here: escalate to the next
tier). The trade is a **risk–coverage curve**: as you raise the threshold,
coverage (fraction you answer locally) drops and accuracy-on-answered rises.
There's no single right point — it's a product knob you set against *your* cost
of a wrong local answer vs. the cost of escalating.

## From a knob to a guarantee: conformal risk control

Picking the threshold "where the validation curve looks good" gives you no
promise on fresh data. **Conformal risk control** does: using a held-out
calibration set, it picks a threshold such that the error rate on everything
kept locally is provably ≤ ε with confidence 1−δ — *distribution-free* and
*finite-sample* (no assumption that the model is right about its own
confidence). tunelab's composer reports a **certified operating point** (via a
Clopper–Pearson bound on the kept-set error) next to the accuracy-optimal one,
so "the local tiers are wrong at most 5% of the time, at 95% confidence"
becomes a line you can put in a recipe instead of a hope.

The catch every honest deployment states: the guarantee holds only while new
traffic looks like the calibration set. When the distribution drifts, the
[data flywheel](data-flywheels-and-active-learning.md) re-calibrates on fresh
labels — calibration is a maintenance contract, not a one-time fit.
