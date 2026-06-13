# Recipe 5 — The self-improving system (the capstone)

**The whole point of everything else. A deployed AI system — one model, a cascade, an agent, or a
deterministic multi-model workflow — that logs every prediction, collects feedback, and runs
champion/challenger experiments to discover and adopt a better architecture over time. Driven by
`tune-loop`. Every mechanism below is demonstrated on real Banking77 data (2026-06-12, $0.00).**

## The shift

Recipes 1–4 build models. This one builds the *loop* that keeps them improving:

```
serve → log every prediction → collect feedback → curate → experiment across
architectures × methods → promote only what beats the champion → repeat
```

The system boundary covers data prep, training, evaluation, serving, and feedback. What sits
inside is just the current champion — the loop is free to replace it (swap a tier, add an RLVR
round, change the whole architecture) when evidence says so. This is champion/challenger from
classical MLOps, generalized to compound-AI architecture search, run as a Monitor–Analyze–
Plan–Execute control loop.

## The pieces (all already built across tunelab)

| Stage | Tool | What it does |
|---|---|---|
| Descriptor | `system/descriptor.json` | the champion architecture as versioned data — enumerable, comparable |
| Monitor + Analyze | `flywheel.py status` | audit-slice accuracy, drift (PSI), per-tier coverage, trigger evaluation |
| Plan | `flywheel.py plan` + `cascade_compose.py` | curate (holdout-excluded), generate challengers, score cheap proxies offline |
| Execute | tune-train + tune-eval | train survivors, score on a fresh slice |
| Promote | `promote.py` | adjudicate vs the pre-registered bar on a one-look slice; bump the descriptor on a win |

## Demonstrated: one turn of the loop, measured

A champion classifier trained on a **starved 2,000-record** slice of Banking77:

- `flywheel.py status` reads the prediction log: **audit-slice accuracy 0.72** (the honest served
  estimate, reported separately from the biased feedback pile at 0.72 — bias-awareness working),
  triggers FIRE (307 new labels ≥ 200; accuracy 0.72 < 0.90 floor) → retrain candidate.
- Challenger retrained after one feedback cycle (full 8,005 records).
- `promote.py`: champion **0.8128** vs challenger **0.8904** = **+7.8 points**, clears the bar,
  beats the champion → **PROMOTE**, descriptor bumped to the next version.

The lift *is* the receipt. Run it again and the slice-reuse guard refuses to adjudicate twice on
the same eval slice — the discipline is mechanical, not aspirational.

## The three failure modes — and how the design owns each

A self-improving loop is easy to build badly. The guardrails that make it trustworthy:

1. **Feedback bias** — feedback over-samples hard/escalated cases. Owned by a uniform **random
   audit slice**; that slice (0.72 in the demo, vs an even more misleading biased read) is the
   honest accuracy, reported separately.
2. **Eval burn** — the loop eats test sets. Owned by **rolling frozen slices consumed exactly
   once** — `promote.py` keeps a consumed-slices ledger and hard-errors on reuse.
3. **Search explosion** — architectures × methods is unbounded. Owned by **staged search**
   (architecture family first, then one refinement per round), a declared per-round budget, and a
   minimum-improvement bar so rounds converge instead of churning.

## The discipline that separates this from AutoML slop

Pre-registered promotion bars · one-look eval slices · declared budgets · append-only logs ·
**human checkpoints at spend and promotion** (delegable). tunelab v1 named autonomous
hill-climbing a non-goal; the capstone reverses that deliberately — and these disciplines are the
reason it's teaching-grade. The EXPERIMENT-LOG a learner walks away with is still the product.

## Where it runs

Two dogfood stages: (1) the machinery on a **Banking77 replay stream** (controlled, simulated
feedback, labeled as such — done above); (2) the showcase on **real router traffic** (Recipe 2 —
logs that accrue daily, with a blinded judge as automated feedback). No serving infrastructure is
shipped: artifacts stay local scripts, logs, and descriptors. `tune-loop` is the crank; the other
four skills are the machine.

Driver + full schema: `skills/tune-loop/SKILL.md`. Run log: `dogfood/cascade/EXPERIMENT-LOG.md`
(flywheel cycle).
