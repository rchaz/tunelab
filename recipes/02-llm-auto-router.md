# Recipe 2 — LLM auto-router

**Who this is for:** your app sends every request to one expensive model, but most requests are easy. You want to automatically send the easy ones to a cheap model and keep only the hard ones on the expensive one.

**The plain idea.** A *router* looks at each incoming query and decides — *before* making the call — whether a cheap model can handle it or it needs the frontier model. The catch: the router sits in front of every request, so it has to be nearly instant (a ~10ms budget). That's why it's a tiny classifier on text embeddings, not another LLM.

Every number below is from a real run on the author's own coding-agent traffic (June 2026, M1 Pro, **$0 API spend**).

## Set the bar first

- **Main goal:** route as much traffic as possible to the cheap model *without* hurting quality. Report the whole trade-off curve, not one point — how aggressive to be is a dial you set later.
- **The guardrail that matters most: don't send a hard query to the cheap model.** Getting that wrong fails a user; the reverse (sending an easy query to the expensive model) just wastes a little money. Bar: accuracy ≥ 0.90 on never-seen request types, **and zero hard-queries-routed-cheap** at the shipped setting.
- Router latency < 10ms.

## The data: your own logs, mess and all

The training data was ~92,000 logged prompts from real coding sessions. Real logs aren't a clean sample — you have to structure them first:

1. **Remove near-duplicates** (5,294 → 1,747 unique). Agent traffic is heavily templated; the de-duplication warning fires exactly as designed. Keep template *variants* where the actual payload differs.
2. **Group into families** by similarity (→ 149 families; the top 10 cover 63% of all traffic). Label at the family level, then spread the label to variants — and verify that spreading was safe.
3. **Split by family, never by individual record.** If variants of the same template land in both training and test data, you get a fake "99% accuracy" that means nothing. This is called *leakage*, and it's the #1 way evaluations lie.

Labels were generated for free by the Claude Code session itself (no API key needed), under a frozen rule: cheap-OK = bounded tasks (classify, summarize given text, reformat); needs-frontier = planning, open-ended synthesis; **when in doubt → frontier.**

## What actually happened: the bar failed first (and that's the useful part)

Round 1 scored 0.988 accuracy — but **one hard query out of 23 got routed cheap. Bar failed.** The autopsy is the most instructive part of this recipe:

The mistake was a "template twin." Two prompts shared ~90% of their text; one asked only for selections from a fixed menu (cheap-OK), the other *also* demanded open-ended designed questions (needs-frontier). The labels were correct — the *router* was fooled, because **an embedding-based router blurs two prompts whose only real difference is a short instruction buried inside a long, near-identical prompt.** This failure is invisible to record-level splits and to headline accuracy. Only the "zero hard-queries-routed-cheap" guardrail caught it.

The fix was *more data for the weak spot*, not a weaker bar: 19 boundary examples mined from unseen sessions, the deciding rule made explicit, retrain, and test on a **fresh** set (the round-1 test set was "spent" the moment it triggered a fix and can't be reused).

## Round 2: passes, on harder data

Fresh test = 60 never-before-seen families, 1,328 records:

| Metric | Result | Bar |
|---|---|---|
| Accuracy | **0.999** | ≥ 0.90 ✅ |
| Hard-queries-routed-cheap | **0 / 274** | 0 ✅ |
| Traffic handled by cheap model | 79.3% | (the dial) |

## The payoff (measured over all 7,794 real events)

| | |
|---|---|
| Routed to cheap model with zero misses | **39.0% of events** |
| Cost, all-frontier | $72.06 |
| Cost, hybrid | $53.04 — **26.4% saved** |
| Router latency | ~0.6ms per query — 15× under budget |
| API spend to build it | **$0.00** |

Cost model stated plainly: input tokens ≈ characters/4 plus 300 output tokens per event; Opus-tier $5/$25 vs Haiku-tier $1/$5 per million tokens (verified June 2026). Your traffic will differ — the reason routing pays is that a few templates dominate the volume, which is also why savings must be measured over the *raw event stream*, not unique queries.

## What's solid and what isn't

- "Zero misses" is bounded by a 274-record sample and the quality of session-generated labels; both improve with scale. Re-checking fresh traffic monthly is the maintenance plan.
- No logs yet? Start with a simple category→model table and let the learned router replace it as logs accumulate.
- The ~21% kept on the frontier includes the borderline cases on purpose — those are the most valuable data for the next round.

Full run log, both rounds: [`dogfood/router/EXPERIMENT-LOG.md`](../dogfood/router/EXPERIMENT-LOG.md).
