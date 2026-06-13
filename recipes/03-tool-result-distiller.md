# Recipe 3 — MCP tool-result distiller

**Level 2 · stop paying frontier prices to read JSON · every number below is from a real run on
the author's own agent transcripts (June 2026, M1 Pro 16GB, $0.00 API spend) — including the
honest ending: the SFT student compresses beautifully, fools a frontier judge, and still fails
the grounding gate — which is exactly why the gate, not the judge, owns the guarantee.**

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

## The memory wall — and how it fell

Early training (`Qwen3.5-2B-4bit`) OOM'd six consecutive times on this 16GB M1 Pro
(`kIOGPUCommandBufferCallbackErrorOutOfMemory`), dying at the iter-1 val pass even on a minimal
config and a freshly rebooted machine. The lesson from those six legs stands: **`--max-seq-length`
is a cap, not a cost — your actual token distribution is the cost.** Compression data is
long-blob by definition, so it blows the activation budget that short-record SFT never touches.

But the wall itself turned out to be one `sysctl` away. The six legs all ran under macOS's
**default Metal wired-memory limit** (~⅔ of RAM ≈ 10.9GB on 16GB) — never raised:

```bash
sudo sysctl iogpu.wired_limit_mb=13312   # ~13GB; reserves ~2.7GB for macOS; reverts on reboot
```

With the limit raised, **Qwen3-4B trained fine** — the model *twice the size* of the one that
OOM'd six times. It cleared the iter-1 val pass that killed the earlier legs and trained to
completion at a 6.86GB peak. The boundary was a default, not the hardware. (A NaN detour along the
way taught its own lesson: loss went NaN at a fixed iteration regardless of learning rate — the
culprit was *data*, not LR. Records whose templated prompt fills the whole `--max-seq-length`
window leave zero trainable completion tokens under `--mask-prompt`, dividing the masked loss by
zero. A two-minute tokenizer scan found them; a full-fidelity rebuild at a larger cap fixed it.
`tune-data`'s validator now hard-fails such records.)

## The student exists — and the result is the recipe's whole point

Trained to the validation bottom, evaluated against the pre-registered bar on the untouched
68-record test set (same mechanical gate on student and teacher):

| pre-registered bar | result | |
|---|---|---|
| compression ratio p50 ≤ 0.35 | **0.215** | ✅ PASS |
| blinded judge ≥ 0.60 equiv-or-better vs teacher | **0.94** (15 ties, 1 teacher win / 16) | ✅ PASS |
| hallucinated-value rate ≈ 0 (≥ 0.99 zero-hallucination) | **0.82** (teacher 0.93) | ❌ **FAIL** |

**The blinded judge rated the student equivalent to its teacher — and the mechanical gate caught
the student inventing identifiers ~18% of the time** (`GoalDecomposer.replan`, a fabricated
`1970-01-01T00:00:00Z` timestamp, status-code lists). This is the recipe's thesis proven on
itself: *a frontier judge's eye misses the exact corruption the gate catches by string-matching.*
The gate, not the judge, owns the grounding guarantee — and grounding is non-negotiable here, so
**the SFT student does not ship.**

## The fix is the next round, and it's motivated by a measured failure

SFT transferred the compression *behavior* but not the grounding *discipline*. That's exactly the
case for **RLVR** (reinforcement learning with verifiable rewards): the grounding gate is already
a mechanical verifier, so it becomes the reward. SFT → GRPO-against-the-gate (gate-pass +
ratio-budget terms) teaches the student the property imitation didn't transfer. `mlx-lm-lora`
ships GRPO locally; the 117 gate-failed teacher outputs are the natural ORPO rejected-set
alternative. See [concepts/sft-vs-preference-tuning.md](../concepts/sft-vs-preference-tuning.md)
— the decision rule is "do you have a verifier?" first, and here the answer is yes.

## Receipts

| | |
|---|---|
| verified training pairs from real agent logs | **683** (85.4% gate pass) |
| teacher compression ratio | **p50 0.245** (4× cheaper context) |
| student (SFT) — ratio / judge / grounding | **0.215 / 0.94 / 0.82** (2 of 3 bars; grounding owed to RLVR) |
| API spend for the entire dataset + eval | **$0.00** (session-native teacher + judge) |
| training on 16GB | **works** — `iogpu.wired_limit_mb` raised; 4B peak 6.86GB |

## Honest bounds

- The grounding gate survived the original dogfood prep only as prose; the committed
  `grounding_gate.py` reconstructs it, calibrated to the teacher's documented ~85% population
  pass rate. A pinned original gate might move absolute numbers a point or two; the
  **student-vs-teacher gap (−10 points on grounding) is the robust result.**
- Judge n=16 is directional (the noise threshold wants ~100); field recall not recomputed.
  Neither rescues the grounding FAIL.
- Single-user traffic (one developer's coding-agent sessions); the *pipeline* transfers, the
  dataset's distribution doesn't claim to.
- If ~2% of your compressor's outputs have "close enough" values — rounded numbers, merged IDs —
  that is corruption, not noise. Run the gate over every output; optimize ratio only subject to
  hallucinated-value rate ≈ 0.

Full run log including the six OOM legs, the NaN root-cause, and the eval:
`dogfood/distiller/EXPERIMENT-LOG.md`.
