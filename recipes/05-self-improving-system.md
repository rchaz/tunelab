# Recipe 5 — The self-improving system

**Who this is for:** you've shipped something from one of the other recipes, and now you want it to *keep getting better* on its own as real usage accumulates — without you babysitting it or letting it quietly degrade.

**The plain idea.** Recipes 1–4 build models. This one builds the *loop* that keeps them improving:

```
serve → log every prediction → collect feedback → try better versions →
keep one only if it beats the current best → repeat
```

This is **champion/challenger**: there's always a current "champion" in production. The loop trains "challengers," tests them fairly, and promotes one *only* when it genuinely wins. Driven by `tune-loop`, with the other four skills as its tools.

## The pieces (all already built across tunelab)

| Stage | Tool | What it does |
|---|---|---|
| Describe | `descriptor.json` | the current system, written down as data so versions can be compared |
| Watch | `flywheel.py status` | tracks accuracy and drift, decides when a retrain is worth it |
| Plan | `flywheel.py plan` | curates new data (keeping the test set separate) and proposes challengers |
| Train & grade | tune-train + tune-eval | trains the challengers, scores them on fresh data |
| Promote | `promote.py` | compares to the champion against a pre-set bar; promotes only a real winner |

## Demonstrated: one turn of the loop, measured

A champion classifier trained on a deliberately starved 2,000-record slice of Banking77:

- `flywheel.py status` reads the prediction log: real accuracy **0.72**, enough new labels arrived, and accuracy is below the floor → **retrain triggered.**
- A challenger is retrained after one feedback cycle (now on the full 8,005 records).
- `promote.py`: champion **0.813** vs challenger **0.890** = **+7.8 points** → **promote**, version bumped.

Run it again and the system *refuses* to re-grade on the same test slice — the discipline is enforced by code, not by good intentions.

## Why a self-improving loop is easy to get wrong — and how this one doesn't

Three classic traps, each with a built-in guardrail:

1. **Feedback is biased.** People report problems more than successes, so the feedback pile looks worse than reality. Fixed by also keeping a small **random sample** as the honest accuracy estimate, reported separately.
2. **The loop burns through test sets.** Reuse a test set and it stops being a fair test. Fixed by **frozen slices used exactly once**, tracked in a ledger that errors on reuse.
3. **Too many things to try.** The space of possible changes is endless. Fixed by a **staged search** (pick the architecture first, then one refinement per round) with a declared budget and a minimum-improvement bar, so it converges instead of churning.

## What separates this from "AutoML that hill-climbs forever"

Bars set in advance · test slices used once · declared budgets · append-only logs · **human sign-off at the points that spend money or change production** (you can delegate these, but they exist). The whole thing stays auditable and teaching-grade — the experiment log a learner walks away with is still the real product.

## Where it runs

Two stages: (1) the machinery on a replayed Banking77 stream (controlled, clearly labeled as simulated — shown above); (2) the showcase on real router traffic (Recipe 2), where logs accumulate daily and a blinded judge acts as automated feedback. No serving infrastructure ships — everything is local scripts, logs, and version descriptors. `tune-loop` is the crank; the other four skills are the machine.

Driver and full schema: [`skills/tune-loop/SKILL.md`](../skills/tune-loop/SKILL.md). Run log: [`dogfood/cascade/EXPERIMENT-LOG.md`](../dogfood/cascade/EXPERIMENT-LOG.md).
