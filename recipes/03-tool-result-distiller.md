# Recipe 3 — MCP tool-result distiller

**Level 2 · stop paying frontier prices to read JSON · every number below is from a real run on
the author's own agent transcripts (June 2026, M1 Pro 16GB, $0.00 API spend) — including the
honest ending: this one hits a hardware boundary, and the recipe documents exactly where.**

## The problem

An agent's tool results — file reads, test output, search hits, API JSON — re-enter the context
window and get re-billed as input tokens at *every subsequent step* of the loop. A compressor
that cuts a 4,000-char blob to 1,000 chars pays for itself dozens of times per session. But a
compressor that *invents* data is worse than no compressor: downstream consumers can't detect
the corruption. That asymmetry — the cost win is nice, the grounding guarantee is mandatory —
shapes everything below.

## Metric card (pre-registered 2026-06-11, before any training or eval)

- **Hallucinated-value rate ≈ 0** (the guardrail that owns the design): every identifier/number
  in an output must be atomically groundable in its source blob; bar = ≥99% of test outputs with
  zero atomic hallucinations. Non-negotiable.
- **Field recall ≥ 90%** of consumer-used identifiers retained — measured against what the
  downstream turn verbatim-reused, so it's a *floor* (models paraphrase; the judge carries the
  faithfulness weight).
- **Compression ratio:** median ≤ 0.35 on test (the cost win needs < 0.5).
- **Blinded judge vs teacher:** tuned ≥ 60% equivalent-or-better — the dogfood proxy for the
  production metric (downstream answer equivalence ≥ 98% with compressed vs raw inputs, which
  requires re-running live agent tasks).

The bar was logged before a single score existed. It still is: no eval has run (see the ending),
so the bar remains **unconsumed** — any future training run against this dataset inherits it
as-is. That is what pre-registration means.

## Data: free distillation from logs you already have

1. **Extract** (blob → consuming assistant turn) pairs from agent transcripts: 1,200 pairs,
   blobs 809–11,955 chars (p50 4,263). The consuming turn is free ground truth: the identifiers
   it verbatim-reused become the field-recall metric (median 2 per pair).
2. **Dedupe** at char-4-gram 0.80: 1,200 → 1,149.
3. **Teacher-compress** with a session-native teacher (the Claude Code session itself — no API
   key): 25-blob spot-check first, then 775 more in batches once the prompt froze.

## The centerpiece: the mechanical grounding gate

Eyeballs miss what string matching catches. The gate: split every output into atomic tokens,
verbatim-match each identifier/number against the source blob, and hard-cap the length budget.
Run it on the **teacher** before training, and on the **student** at eval — same gate, both ends.

Spot-check round 1 (draft prompt, 25 real blobs) had zero invented values to the eye — and the
gate still failed it: value *reformatting* ("5.54 kB" → "5.54kB") and *composed tokens*
("ClassName.method" joined from separate source values), both of which break verbatim grounding
and would teach the student bad habits. Plus 8/25 over the length budget. The fix was prompt
rev 2; round 2 came back p50 ratio 0.24, zero over-cap, zero atomic hallucinations. **Prompt
frozen.** The core of it:

> Copy every value EXACTLY as it appears in the input — never re-space ("5.54 kB" must NOT
> become "5.54kB"), never re-case, never abbreviate a value, and never join multiple values into
> composed tokens with separators. New connective English words are fine; new or altered VALUES
> are failures. LENGTH BUDGET: before writing, compute budget = 25% of the input's character
> count; anything over 40% is a failed compression. If the tool result is an error, the error
> message, code, and probable locus are the payload — keep them exact and spend the budget there.

At scale the gate kept earning: **683 of 800 teacher outputs passed (85.4%)**; 117 were dropped
rather than trained on. Survivors: ratio p50 0.245, max 0.325. Full provenance:
`dogfood/distiller/DATACARD.md`.

## Where it ended: the 16GB boundary, with receipts

Training `Qwen3.5-2B-4bit` (QLoRA, `--grad-checkpoint`, `--mask-prompt`) on 365 pairs OOM'd six
consecutive times on a 16GB M1 Pro (`kIOGPUCommandBufferCallbackErrorOutOfMemory`):

| leg | batch / grad-accum / seqlen / layers | outcome |
|---|---|---|
| 1 | 2 / 2 / 4096 / 16 | OOM |
| 2 | 1 / 4 / 3072 / 16 | OOM |
| 3 | 1 / 4 / 2048 / 16 (train trimmed to ≤1,850 tokens) | OOM |
| 4 | 1 / 4 / 2048 / 8 | OOM |
| 5 | 1 / 4 / 2048 / 8 — **freshly rebooted machine** | OOM at the iter-1 val pass |
| 6 | 1 / 1 / 2048 / 8 — minimal possible config | OOM at the iter-1 val pass |

Legs 5–6 rule out the convenient explanations (memory pressure, zombie processes): a clean
machine, a minimal config, and both legs computed a full val pass (val loss 1.079, identical)
then died before training iteration 1.

**The diagnosis is the lesson: `--max-seq-length` is a cap, not a cost — your actual token
distribution is the cost.** The same machine trained a *larger* model (Qwen3-4B, the Level 2
triage dogfood run, 12.4GB peak) under the same 2048 cap, because those records were ~100–400
actual tokens. Compression data is the opposite shape by definition: the long blobs are the use case.
Records up to 1,850 actual tokens blow the activation budget that short-record SFT never
touches.

What we did **not** do, on purpose: trim training to ≤1,024 tokens (guts the dataset below the
sizing minimum and biases it toward short blobs — precisely the cases where a compressor matters
least), or swap to the 0.8B classification-tier checkpoint (untested on the hardest task family
in the metrics table). Six progressively minimal configs is evidence; a seventh is denial.

**Requirement: 32GB+ unified memory for variable-length multi-thousand-token SFT** (or a cloud
backend — on the roadmap). 16GB remains fine for short-record SFT — the Level 2 triage run
(`dogfood/level2/`) proves that on the same machine.

## Receipts

| | |
|---|---|
| verified training pairs from real agent logs | **683** (85.4% gate pass) |
| teacher compression ratio | **p50 0.245** (4× cheaper context), max 0.325 |
| atomic hallucinations in survivors | **0** (by construction — the gate) |
| API spend for the entire dataset | **$0.00** (session-native teacher) |
| training on 16GB | **blocked** — 6/6 Metal OOM, documented above |

## Honest bounds

- No student model exists yet, so no claim about student quality is made — the pre-registered
  bar is waiting for hardware that fits. The dataset, frozen prompt, and gate are the artifact.
- Single-user traffic (one developer's coding-agent sessions); the *pipeline* transfers, the
  dataset's distribution doesn't claim to.
- Field recall is a floor metric (median 2 verbatim-reused identifiers/pair; 190/683 pairs have
  zero) — the blinded judge is the faithfulness instrument.
- If ~2% of your compressor's outputs have "close enough" values — rounded numbers, merged IDs —
  that is corruption, not noise. Run the gate over every output; optimize ratio only subject to
  hallucinated-value rate ≈ 0.

Full run log including all six OOM legs: `dogfood/distiller/EXPERIMENT-LOG.md` and
`dogfood/distiller/runs/20260611-qwen35-2b-distiller/` (state.json + per-leg train logs).
