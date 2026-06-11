# Recipe 2 — LLM auto-router

**Level 1 · route every query to the cheapest capable model · every number below is from a real
run on the author's own agent-harness traffic (June 2026, M1 Pro, $0.00 API spend).**

## The problem

If an LLM app sends every call to a frontier model, it pays frontier prices for work a cheap
model handles — and most production traffic is exactly that work. A router decides, per query,
*before* the call: cheap model or frontier? The constraint that shapes everything: the router
sits in the request path, so it gets a **latency budget of ~10ms** — which is why this is
embeddings + logistic regression (Level 1), not an SLM.

## Metric card (pre-registered before any score existed)

- **Primary:** cost reduction at iso-quality — % of traffic routed cheap while frontier-bound
  work stays frontier-bound. Report the threshold curve, not one point: the threshold is a
  product knob.
- **The guardrail that owns the design: false-cheap rate.** A hard query routed cheap fails a
  user; a cheap query routed expensive just costs margin. Bar for this run: family-held-out
  macro-F1 ≥ 0.90 **and zero observed false-cheap at the shipped threshold**.
- Router latency < 10ms. Calibration of the cheap-suffices probability (LR gives it natively).

## Data: your own logs, with their real mess

The training population was the author's Claude Code transcripts — ~92k logged prompts. Real
logs are not flat samples; structure them before labeling:

1. **Dedupe at 0.95** (5,294 events → 1,747 unique): agent-harness traffic is templated, and the
   collapse warning fires exactly as designed. Keep template *variants* (payloads differ).
2. **Cluster into families at 0.80** (→ 149 families; top-10 = 63% of traffic). Label at the
   family level, propagate to variants, and *verify propagation* (payload-flip check: would any
   realistic payload flip the route? 10/10 top families: no).
3. **Split by family, never by record.** Variants of one template in both train and test is
   leakage that reads as 99% accuracy and means nothing.

Labels were session-native (the Claude Code session labels its own logs — no API key) under a
frozen rubric: `cheap_ok` = bounded transforms (classify into fixed labels, summarize given
text, extract/reformat, select from enums); `needs_frontier` = planning/decomposition, agentic
side effects, open-ended synthesis; **uncertain → frontier**.

## What actually happened: the bar failed first

Round 1 (29 held-out families): macro-F1 0.988 — and **one of 23 frontier records routed
cheap. Bar failed.** The autopsy is the most useful part of this recipe:

The false-cheap was a template twin. Two real prompts share ~90% of their text ("you are an
architecture advisor… here are the available blocks…"); one asks ONLY for selections from the
given enums (`cheap_ok`), the other's response schema also demands an `expanded_vision` and 3–5
designed questions (`needs_frontier`). The labels were right. The router was wrong: **embedding
routers blur template twins whose difference is a short output-schema suffix inside a long
shared prompt** — and the failing family was entirely unseen, which is the grouped split doing
its job. This failure mode is invisible to record-level splits and headline accuracy; only the
false-cheap guardrail catches it.

The fix, straight from the decision matrix — more data for the weak boundary, not a new bar:
19 boundary variants hard-mined from unseen sessions, the rubric's deciding line made explicit
(*the response schema decides, not the boilerplate*), retrain, and judge on a **fresh** test
(the round-1 holdout was spent the moment it triggered a fix).

## Round 2: pass, on harder data

Fresh test = 60 never-before-seen families, 1,328 records, 274 frontier-bound:

| metric | result | bar |
|---|---|---|
| macro-F1 | **0.999** | ≥ 0.90 ✅ |
| false-cheap at t=0.5 | **0 / 274** | 0 ✅ |
| coverage at t=0.5 | 79.3% | (the knob) |

## Receipts (volume-true, all 7,794 raw events)

| | |
|---|---|
| routed cheap at zero observed false-cheap | **39.0% of events** |
| est. cost, all-frontier | $72.06 |
| est. cost, hybrid | $53.04 — **26.4% saving** |
| router latency | ~0.6ms/query incl. model load (~0.2ms marginal) — 15× under budget |
| API spend to build | $0.00 (local embeddings, session-native labels) |

Cost model stated plainly: chars/4 input tokens + 300 output tokens per event; Opus-tier
$5/$25 vs Haiku-tier $1/$5 per MTok (verified June 2026). Your traffic shape will differ — the
volume-weighted family structure (here: 63% of events in 10 templates) is what makes routing
pay, and it's also why receipts must be computed over the raw event stream, not unique queries.

## Honest bounds

- "Zero false-cheap" is bounded by a 274-record frontier sample and session-native label
  quality — both improve with scale; the drift monitor (re-score fresh traffic monthly) is the
  maintenance contract.
- Cold start without logs: a static category→model table, replaced by the learned router as
  logs accrue — public benchmark scores correlate loosely with *your* distribution.
- The routed-to-frontier 21% includes the borderline cases by design (uncertain → frontier);
  those are the next round's most valuable training data.

Full run log including both rounds: `dogfood/router/EXPERIMENT-LOG.md`.
