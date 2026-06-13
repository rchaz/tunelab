# Recipe 3 — Tool-result distiller

**Who this is for:** you run an AI agent that calls tools (file reads, test output, search results, API responses). Those bulky results get re-sent to the model at every step, and you pay for them again every time. You want to compress them first.

**The plain idea.** Train a small local model to shrink a tool's output — say a 4,000-character blob down to 1,000 — while keeping every important value intact. A good compressor pays for itself dozens of times per session. But a compressor that *invents* data is worse than useless, because nothing downstream can tell it's wrong. That asymmetry — saving money is nice, never inventing facts is mandatory — shapes the whole recipe.

Every number below is from a real run on the author's own agent transcripts (June 2026, M1 Pro 16GB, **$0 API spend**) — including the honest ending: **the small model compressed beautifully, fooled an expert judge, and still failed the one bar that mattered.** That's why a mechanical check, not a human-style judge, owns the guarantee.

> **Jargon, once:** *distillation* = using a strong "teacher" model to generate training examples for a small "student" model. *SFT* (supervised fine-tuning) = training the student to imitate those examples.

## Set the bar first (before any training)

- **Never invent values** (the bar that owns the design): every ID and number in the output must be copy-able from the source. Target: ≥ 99% of outputs with zero invented values. Non-negotiable.
- **Keep ≥ 90%** of the values that downstream steps actually used.
- **Compress to ≤ 0.35** of original size (the cost win needs well under half).
- **Blinded judge:** the student rated as good as its teacher at least 60% of the time.

## The data: free, from logs you already have

1. **Pair up** each tool-output blob with the agent turn that consumed it: 1,200 pairs. That consuming turn is free ground truth — the IDs it actually reused tell you what the compressor must keep.
2. **Remove near-duplicates** (1,200 → 1,149).
3. **Compress with a teacher** (the Claude Code session itself — no API key): spot-check 25 by hand first, then the rest once the prompt was frozen.

## The centerpiece: a mechanical grounding check

Human eyes miss what string-matching catches. The "grounding gate" splits every output into individual tokens and verifies each ID/number appears *verbatim* in the source. Run it on the teacher before training, and on the student at evaluation.

It earned its keep immediately. The first spot-check looked perfect to the eye — but the gate caught the teacher *reformatting* values ("5.54 kB" → "5.54kB") and *gluing* separate values together. Both look harmless and both would teach the student bad habits. The fix was a stricter prompt:

> Copy every value EXACTLY as it appears — never re-space, re-case, abbreviate, or join values together. New connecting words are fine; new or altered VALUES are failures. Budget: aim for 25% of the input length; over 40% is a failed compression.

At scale, **683 of 800 teacher outputs passed (85.4%)**; the other 117 were thrown out rather than trained on.

## A real-world wall: out of memory (and how it fell)

Early training (on a 2B model) hit out-of-memory crashes **six times in a row** on the 16GB laptop. The lesson holds: **the `--max-seq-length` setting is a cap, not a cost — your actual data length is the cost.** Compression data is long by nature, so it blows a memory budget that short-text training never touches.

But the wall itself was one setting away. macOS caps how much memory the GPU can use (~⅔ of RAM by default). Raising it:

```bash
sudo sysctl iogpu.wired_limit_mb=13312   # ~13GB; reverts on reboot
```

With the cap raised, a **4B model — twice the size of the one that crashed six times — trained fine** (6.86GB peak). The wall was a default setting, not the hardware.

(A side lesson: training loss kept going to NaN ["not a number"] at a fixed step regardless of settings. The cause was *data*, not the learning rate — a few records had a prompt so long it left zero room for the answer to learn from. tune-data's validator now catches these automatically.)

## The result — and the recipe's whole point

Trained, then evaluated against the pre-set bar on an untouched 68-record test set:

| Pre-set bar | Result | |
|---|---|---|
| Compress to ≤ 0.35 | **0.215** | ✅ PASS |
| Judge rates student ≥ 60% as-good-as teacher | **0.94** | ✅ PASS |
| Never invent values (≥ 0.99 clean) | **0.82** (teacher: 0.93) | ❌ **FAIL** |

**The expert judge rated the student as good as its teacher — and the mechanical gate caught the student inventing IDs about 18% of the time** (fake method names, a made-up timestamp, invented status codes). This is the recipe's thesis, proven on itself: *a human-style judge misses the exact corruption a string-matching check catches.* Grounding is non-negotiable here, so **the student does not ship.**

## The fix is the next round

Imitation (SFT) taught the compression *behavior* but not the grounding *discipline*. The fix is to train the model with the grounding gate as a *reward* — it's already a mechanical pass/fail check, which is exactly what reinforcement learning needs. (This is called RLVR — reinforcement learning with verifiable rewards; the decision rule is simply "do you have an automatic checker?" Here, yes.) See [SFT vs preference tuning](../concepts/sft-vs-preference-tuning.md).

## The bottom line

| | |
|---|---|
| Verified training pairs, free from logs | **683** |
| Teacher compression | **~4× smaller context** |
| Student: size / judge / grounding | **0.215 / 0.94 / 0.82** (2 of 3 bars; grounding owed to the next round) |
| API spend, whole build + eval | **$0.00** |
| Training on 16GB | **works** (memory cap raised; 4B peak 6.86GB) |

## What's solid and what isn't

- The robust result is the **10-point grounding gap between student and teacher**, not the exact absolute numbers.
- The judge sample (16) is directional; a bigger sample would tighten it. Neither rescues the grounding failure.
- This is one developer's traffic — the *pipeline* transfers; the specific dataset doesn't claim to.
- If even ~2% of a compressor's outputs have "close enough" values, that's corruption, not noise. Run the gate over every output.

Full run log, including the six crashes and the NaN hunt: [`dogfood/distiller/EXPERIMENT-LOG.md`](../dogfood/distiller/EXPERIMENT-LOG.md).
