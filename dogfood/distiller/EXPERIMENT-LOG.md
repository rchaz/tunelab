# mcp-tool-result-distiller — EXPERIMENT-LOG

## 2026-06-11 — tune-decide: level decision
- Decision: **Level 2** (LoRA SFT) — compression is open-text generation under hard grounding
  constraints; no fixed label space. Input: raw tool-result blob; output: compressed text
  preserving everything an agent would act on.
- Data: rc's real Claude Code transcripts (local, gitignored) — 1,200 extracted
  (tool_result blob → consuming assistant turn) pairs, 800–12,000 chars/blob, deduped to 1,149
  (51 near-dup blobs). Fields-used ground truth computed from what the downstream turn
  verbatim-reused (median 2 identifier tokens/pair — models paraphrase, so measured field
  recall is a FLOOR metric; the blinded judge carries faithfulness weight).
- Teacher tier: session-native (model unpinned — Claude Fable 5, 2026-06-11).
- Model plan: `mlx-community/Qwen3.5-2B-4bit` (verified id, 1.72GB), memory-light from launch:
  `--grad-checkpoint`, batch 2 + grad-accum 2, `--max-seq-length 4096` (blobs ≈ up to 3k tokens).

## 2026-06-11 — PRE-REGISTERED BAR (logged before any training or eval; §8 metric card)
- **Hallucinated-value rate ≈ 0**: every identifier/number in a test output must be atomically
  groundable in its source blob (composed-token splitting allowed in the checker); ≥99% of test
  outputs with zero atomic hallucinations. Non-negotiable — a compressor that invents data is
  worse than no compressor.
- **Field recall (measured floor)**: ≥90% of consumer-used identifier fields retained on test.
- **Compression ratio**: median ≤0.35 on test (teacher achieved 0.24; the cost win needs <0.5).
- **Blinded judge vs teacher compressions**: tuned ≥60% equivalent-or-better (dogfood proxy for
  §8's production metric — downstream answer equivalence ≥98% with compressed vs raw inputs —
  which requires re-running live agent tasks and is documented as the production-grade eval).
- Registered 2026-06-11 by the session under rc's standing "continue, don't wait" delegation;
  rc holds a veto window until eval (training takes hours).

## 2026-06-11 — tune-data: spot-check rounds (the checkpoint earning its keep)
- Round 1 (draft prompt, 25 real blobs): ZERO invented values, but 8/25 over the 40% ratio cap,
  and two compliance defects the mechanical checker surfaced — value REFORMATTING ("5.54 kB" →
  "5.54kB") and composed-token notation ("ClassName.method" joining separate source values),
  both of which break verbatim grounding and would teach the student bad habits.
- Fix: prompt rev 2 — exact-copy rule (no re-spacing/re-casing/joining), numeric per-blob
  character budget (25% computed, 40% = failure), cut-priority list.
- Round 2 (same 25 blobs): **p50 ratio 0.24, max 0.30, zero over-cap, zero atomic
  hallucinations** (single flag "POSTs" = connective verb). Fields-used retention clean once
  markdown ordinals were excluded from the metric (extractor noise, not content loss).
- **Prompt frozen 2026-06-11** (`data/teacher_prompt_frozen.txt`); scaling to 775 more blobs
  in 39 session-native batches (25 spot-check compressions kept).
- Lesson: the mechanical grounding checker (atomic token split + verbatim match) catches teacher
  defects the eye glosses over; it becomes the same gate the student is judged by.

## 2026-06-11 — tune-data: scale, verification gate, splits
- Scaled the frozen prompt over 775 more blobs (39 session-native batches) + the 25 kept
  spot-check compressions = 800 teacher compressions attempted.
- **Verification gate (same mechanical checker as the spot-check): 683/800 kept (85.4%)** —
  117 outputs dropped for grounding violations or budget overrun. *Logged gap: the per-reason
  breakdown of the 117 was not retained — a future round should log gate failures by type.*
- Survivor stats (`verified_pairs.jsonl`, n=683): blobs 809–11,955 chars (p50 4,263);
  compression ratio p50 0.245, p90 0.249, max 0.325 — the 0.40 hard cap held with margin.
  Fields-used ground truth: median 2 verbatim-reused identifiers/pair; 190/683 pairs have zero
  (paraphrase-heavy consumers) — confirming field recall is a floor metric here.
- Chat-format conversion with the student system prompt (`data/student_system.txt`), exact-dedupe
  683 → 682, seeded 80/10/10 split → **train 546 / valid 68 / test 68**.
- Training legs later length-trimmed train/valid for memory (546 → 514 at ≤3072 tokens → 365 at
  ≤1850; valid 68 → 44). **Test untouched: 68 records over the full length range.**

## 2026-06-12 — tune-train: six legs, six Metal OOMs — RUN CLOSED at the hardware boundary
- Machine: M1 Pro, 16GB. Model: `mlx-community/Qwen3.5-2B-4bit` (1.72GB). Every leg used
  `--grad-checkpoint` and 4-bit QLoRA. Full configs + timestamps in `runs/*/state.json`.

| leg | config (batch / grad-accum / seqlen / layers) | context | outcome |
|---|---|---|---|
| 1 | 2 / 2 / 4096 / 16 | external memory pressure | Metal OOM |
| 2 | 1 / 4 / 3072 / 16 | crashed leg stuck holding ~5GB | Metal OOM |
| 3 | 1 / 4 / 2048 / 16 · train trimmed to 365 recs ≤1850 tok | ~100MB unused even idle | Metal OOM |
| 4 | 1 / 4 / 2048 / 8 | Chrome quit; 2 Metal-zombie trainers resident | Metal OOM |
| 5 | 1 / 4 / 2048 / 8 | **post-reboot, clean machine, wired 1.9GB** | Metal OOM at iter-1 val pass |
| 6 | 1 / 1 / 2048 / 8 | minimal possible config | Metal OOM at iter-1 val pass, exit 137 |

- Legs 5 and 6 disprove the zombie/pressure theories: a freshly rebooted machine, minimal
  config, and both legs computed the full 16-batch val pass (val loss 1.079, identical) and
  died before training iteration 1.
- **Diagnosis: activation memory scales with *actual* sequence length, not the seqlen cap.**
  The Level 2 triage run trained a *larger* model (Qwen3-4B, peak 12.4GB) on this same machine
  with the same nominal `--max-seq-length 2048` — but its records were ~100–400 actual tokens.
  This dataset's records run to 1,850 actual tokens; the val pass batches them and blows the
  budget. `max-seq-length` is a cap, not a cost; your token distribution is the cost.
- **Scope decision (Option C): accept and document the boundary.** Alternatives rejected:
  (a) trimming train to ≤1024 tokens guts the set below the sizing minimum *and* biases it to
  short blobs — exactly the cases where a compressor matters least; (b) Qwen3.5-0.8B-4bit is
  the classification-tier model on the hardest task family in the metrics table — an untested
  multi-hour gamble against a frustrated clock. Six progressively minimal configs, two on a
  clean reboot, is enough evidence.
- **Requirement recorded: 32GB+ unified memory for this sequence-length profile** (or a cloud
  backend — Phase 3 roadmap). The 12.4GB-peak Level 2 run shows 16GB is fine for short-record
  SFT; it is variable-length multi-thousand-token records that cross the line.
- **The pre-registered bar (2026-06-11, above) remains logged and UNCONSUMED.** No eval was
  run; the test set has still been looked at zero times. Any future training run against this
  dataset inherits the bar as-is — that is what pre-registration means.
- What ships regardless (the recipe's actual value): the extraction → teacher spot-check →
  frozen prompt → mechanical grounding gate → verified dataset pipeline, with receipts.
  `recipes/03-tool-result-distiller.md` + `DATACARD.md` document it; $0.00 API spend.
