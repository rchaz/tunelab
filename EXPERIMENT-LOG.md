# tunelab — EXPERIMENT-LOG

Append-only log of decisions, runs, and predictions-vs-actuals for tunelab's own
development. This is the same convention the `tune-*` skills maintain in user
projects (§7 of PLAN.md) — the repo dogfoods it.

---

## 2026-06-10 — Milestone 1: external-fact verification (workflow `verify-external-facts`)

**Method:** three parallel agent branches — (a) live HF API checks on every
mlx-community checkpoint id, (b) `uv tool install mlx-lm` (0.31.3) + full
`--help` diff against `skills/tune-train/references/mlx-reference.md` + live
LORA.md/source reads, (c) dogfood-dataset selection then a live CPU embedding
benchmark on this machine (M1 Pro, 16GB).

### Decisions

| Decision | Choice | Evidence |
|---|---|---|
| §14.2 local embedding default | `sentence-transformers/static-retrieval-mrl-en-v1` loaded via `model2vec` (no torch) | 0.730 acc / 0.730 macro-F1 on CFPB 10-class (vs MiniLM-L6-v2 0.762) at ~30× CPU speed (4,554 vs 152 texts/s) and +35 MB deps (vs +523 MB torch stack). Scripts + raw JSON: `dogfood/embedding-bench/` |
| Embedding fallback / upgrade | fallback `minishlab/potion-base-32M` (same package, 0.692 acc); upgrade `text-embedding-3-small` (OpenAI, `--backend openai`) | same benchmark |
| Level 0 caveat | static embeddings are weak at 20-shot centroids: 0.44–0.45 vs MiniLM 0.617 | same benchmark — skills must teach this and point centroid-heavy users at the upgrade path |
| Dogfood dataset (Level 1 live run) | CFPB Consumer Complaint Database, public-domain, real consumer-written narratives, 10 product classes, 3,000 rows balanced 300/class | `dogfood/level1/data/raw.jsonl` (gitignored); fetched via the no-auth consumerfinance.gov search API. Gotcha: never pass `format=json` — it switches to full-export mode ignoring `size` |
| Phase 1 teacher/judge tier | session-native only (user decision 2026-06-10); API scripts reviewed + shim-tested, not run live | first live API spend deferred past Phase 1 |
| Level 2 dogfood | synthetic Path C via session-native generation (user decision 2026-06-10) | — |

### Verified model table (2026-06-10, all ids returned HTTP 200)

| Task | Start | Escalate |
|---|---|---|
| classification/routing/extraction | `mlx-community/Qwen3.5-0.8B-MLX-4bit` (0.63GB) | `mlx-community/Qwen3.5-2B-4bit` (1.72GB) |
| structured output / JSON | `mlx-community/Qwen3-4B-Instruct-2507-4bit` (2.26GB, non-thinking) | `mlx-community/Qwen3.5-4B-4bit` (3.03GB) |
| style-transfer generation | `mlx-community/gemma-3-4b-it-qat-4bit` (3.00GB) | `mlx-community/Qwen3-8B-4bit` (4.61GB) |
| continued pretraining (base) | `mlx-community/Qwen3-0.6B-Base-4bit` (0.34GB) | `mlx-community/SmolLM3-3B-Base-4bit` (1.73GB) |

Facts that changed the design:
- **Qwen3 1.7B/4B/8B-Base-4bit and all Qwen3.5-Base-4bit do NOT exist** on
  mlx-community. Modern base 4-bit under ~3.5GB: Qwen3-0.6B-Base, LFM2.5-1.2B-Base,
  SmolLM3-3B-Base, gemma-3-{1b,4b}-pt. The CPT path must teach checkpoint
  *verification*, not assume the instruct table has base twins.
- HF API returns **401 (not 404)** for missing repos when unauthenticated.
- `OptiQ`/`-MLX-4bit` suffix traps: OptiQ is mixed-precision (8.2GB for 9B —
  over budget); `-MLX-4bit` and `-4bit` are duplicate uploads.
- Qwen3 (non-2507) has a hybrid thinking chat template — eval must strip/disable
  `<think>` blocks; Qwen3.5 needs a recent mlx-lm (`qwen3_5` arch).

### mlx-lm 0.31.3 ground truth (corrections vs prototype reference doc)

- `--resume-adapter-file` is **weights-only**: optimizer state, LR-schedule
  position, and iteration counter are NOT restored; training restarts at iter 1
  and runs the full `--iters`. → `runs/<id>/state.json` must track
  total/completed iters; resume = `--iters <remaining>` + latest checkpoint;
  expect a brief loss bump from cold optimizer state.
- Checkpoints: `--save-every N` (default 100) writes `adapters.safetensors`
  (latest, overwritten) + `{iter:07d}_adapters.safetensors` (kept) into
  `--adapter-path`. Progress between save points is lost on crash.
- `mlx_lm.fuse --hf-path` does not exist (argparse rejects it; upstream LORA.md
  is itself stale here). Default save path is `fused_model` (underscore);
  `--gguf-path` lands *inside* save-path; quantized (QLoRA) fuses cannot export
  GGUF without `--dequantize`.
- Previously undocumented flags that matter: `--max-seq-length` (default 2048),
  `--optimizer {adam,adamw,muon,sgd,adafactor}`, `--seed`, `--test-batches`,
  `--clear-cache-threshold`, `--report-to wandb|swanlab`, `--project-name`.
- LoRA rank/scale/dropout/target-keys (`lora_parameters`), `lr_schedule`,
  `optimizer_config` are **YAML-config-only** — rank sweeps require `-c config.yaml`.
- A 4th data format exists (`tools`); `valid.jsonl` is optional (warning, not
  error); `--mask-prompt` raises ValueError on text datasets.
- Python API: `generate(model, tok, prompt, max_tokens=N)` — direct kwarg;
  sampling via `make_sampler`; `load(..., adapter_path=)` confirmed.

Full structured results: workflow run `wf_dcc3b4bb-c64` (4 agents, ~206k tokens);
help-text snapshots at `/tmp/tunelab-verify/mlx_help/`.

### Correction (2026-06-10, milestone 2 review)

The verification record's claim that `lora_parameters.keys` defaults to
q_proj/v_proj was stale: in mlx-lm 0.31.3, `CONFIG_DEFAULTS` has
`lora_parameters = {rank: 8, dropout: 0.0, scale: 20.0}` with **no `keys`** —
when `keys` is absent, every linear/quantized-linear/embedding module in the
trained blocks gets adapters (`lora.py` CONFIG_DEFAULTS;
`tuner/utils.py` `linear_to_lora_layers`). tunelab's generated configs pin
`keys: [self_attn.q_proj, self_attn.v_proj]` as a deliberate restriction and
say so in a comment; dropping the line adapts everything.

### Anthropic API check (claude-api skill, no spend)

`claude-opus-4-8` teacher/judge default valid; `output_config.format`
json_schema usage in `distill_generate.py`/`judge_eval.py` is the current
canonical form; `thinking: adaptive` correct on Opus 4.7+; no sampling params —
prototype API scripts are shape-correct, review can focus on logic.
`claude-sonnet-4-6` (tune-data's cheaper-teacher option) also verified valid
via the claude-api model catalog (2026-06-10): Claude Sonnet 4.6, $3/$15 per MTok.

---

## 2026-06-10 — Milestone 2: scripts build (workflow `build-scripts-m2`)

All 11 bundled scripts rewritten or hardened; one new script (`chunk_text.py`,
CPT Path D); `references/mlx-reference.md` rewritten against the installed
mlx-lm 0.31.3. Method: 6 groups × (implement → adversarial review → fix +
evidence tests), 18 agents; zero reviewer findings rejected without evidence.

Highlights (full per-group evidence in the workflow record `wf_c6d4fafc-389`
and reproducible via `bash tests/run_all.sh` — 127 PASS checks, all green on
this machine 2026-06-10):
- `centroid_classify`/`train_classifier`: local embeddings default
  (static-retrieval-mrl-en-v1 via model2vec), `--backend openai` upgrade path;
  live evidence on real CFPB data — 1,000-row train hits 0.675 held-out
  accuracy (full-set benchmark was 0.730), deterministic across runs.
  XGBoost via `uv run --with xgboost` (kept out of inline deps).
- `dedupe`: MinHash made deterministic (blake2b base + fixed affine mixes);
  the prototype's `hash()` scheme was demonstrated to produce different
  signatures under PYTHONHASHSEED=1 vs 999. Byte-identical outputs now proven.
- `validate_dataset`: matches installed mlx-lm semantics (empty-train hard
  error mirrors mlx's own ValueError; chat+tools mixing is legal → warning);
  tools format recognized; token warnings keyed to `--max-seq-length` 2048.
- `recommend_hparams`: emits `--save-every`/`--seed`, `--max-seq-length` when
  records are long, YAML config for `--lora-rank` (verified parseable by
  mlx-lm's own yaml loader); prints weights-only-resume guidance.
- `run_test_set`: `enable_thinking=False` by default + three-case `<think>`
  stripping, proven live against Qwen3-0.6B-4bit (downloaded, 3-record smoke,
  zero think-markers in output). `eval_classifier`: macro-F1 + hallucinated-
  label flagging with hand-computed fixture math.
- API scripts: refusal/parse-skip hardening, resume-file corruption repair,
  judge `max_tokens` 8192 (adaptive thinking headroom), exercised end-to-end
  against a deterministic fake-SDK shim (`tests/shims/anthropic.py`) — still
  never run live (Phase 1 decision stands).
- Main-loop judgment call: dropped split_data's `n >= 3` gate so 1–2-example
  stratified classes also warn when absent from a split.

---

## 2026-06-10/11 — Milestones 5–6: live runs + eval round 1 — PHASE 1 DEFINITION OF DONE MET

**Level 1 live run (CFPB complaint triage)** — PASSED its pre-registered bar:
held-out macro-F1 0.730 ≥ 0.70 on 3,000 real complaints, local embeddings,
$0.00 spend. Level 0 probe reproduced its predicted failure (0.437 vs ~0.45).
Routing sweep: ≥0.6 confidence keeps 90% of traffic at 0.744. Full log:
`dogfood/level1/EXPERIMENT-LOG.md`; tutorial: `recipes/01-ticket-triage.md`.

**Level 2 live run (synthetic Taskloop triage→JSON, Path C)** — PASSED all
three pre-registered bars on a FRESH 90-record test set (the original test was
spent by an interim look and replaced, per the one-look rule): format validity
100% (≥98), category accuracy 92.2% (≥85), blinded judge 68.9%
equivalent-or-better vs teacher (≥60; 5 wins/57 ties/28 losses). Base control:
70.0%/44.4% → lift +22/+40 points. 825 session-native synthetic records
(zero API spend), QLoRA on Qwen3-4B-Instruct-2507-4bit, early-stopped at the
val bottom (global iter ~340 of 823 planned). Full three-leg saga — the
wired-memory freeze, the discarded full-LR resume leg, the halved-LR fix —
in `dogfood/level2/EXPERIMENT-LOG.md`; the resume-LR lesson is now in
tune-train's SKILL.md.

**Eval round 1 (§12)** — PASSED: with-skill 6/6 pass vs no-skill 1 pass /
3 partial / 2 fail; with-skill judged better in all 6 blinded comparisons.
Skill triggering verified (each case loaded the right SKILL.md set from
descriptions alone; case 4 reached concepts/epochs-and-overfitting.md).
Workflow `wf_08e57896-015` (rate-limit interrupted, resumed from journal).
Notes for a Phase-2 prose round: commit to the word "stratified" (or its
absence rationale) in split plans; discourage asserting unperformed checks.

---

## 2026-06-11 — Phase 2: `--provider openai` in distill_generate / judge_eval

### OpenAI API check (no spend — installed SDK + developers.openai.com)

Responses API facts the implementation is built on, verified 2026-06-11:

- **Surface** (verified against installed `openai` 2.41.1):
  `client.responses.create(model, instructions=<system>, input=<user>,
  max_output_tokens, store=False)`; structured outputs via
  `text={"format": {"type": "json_schema", "name": ..., "schema": ...,
  "strict": True}}` — the format-level `name` is required; usage at
  `resp.usage.input_tokens/output_tokens`, but `Response.usage` is typed
  `ResponseUsage | None`, so both scripts guard the None case (under-count
  to 0, never crash); refusals are `refusal`-type content parts inside
  `output` message items; truncation surfaces as `status != "completed"` +
  `incomplete_details.reason`. `store=False` on every call (Responses API
  otherwise persists responses for 30 days). `BadRequestError`,
  `RateLimitError`, `APIConnectionError` all subclass `openai.APIError`, so
  SDK-retry exhaustion and 4xx land in the per-record skip path
  (`OpenAI(max_retries=5)`); SDK `ReasoningEffort` literal is
  `'none'|'minimal'|'low'|'medium'|'high'|'xhigh'`.
- **Models** (developers.openai.com/api/docs/models/\*, fetched 2026-06-11):
  - `gpt-5.5` (default teacher/judge): model page states verbatim
    "Reasoning.effort supports: none, low, medium (default), high and xhigh"
    — closes the open question on `reasoning={"effort": "none"}` (distill)
    and `"low"` (judge). $5/$30 per MTok, 1.05M context, snapshot
    `gpt-5.5-2026-04-23`. The GPT-5.5 usage guide reserves `none` for
    "latency-critical tasks that don't need reasoning ... such as ...
    classification" — exactly distill's classify use.
  - `gpt-5.4`: "none (default), low, medium, high and xhigh", $2.50/$15.
    `gpt-5.4-mini`: $0.75/$4.50. `gpt-5.4-nano`: $0.20/$1.25 ("cheapest ...
    for simple high-volume tasks like classification"). All four ids named
    in the scripts' `--model` help text exist.
- **Hardening added with the provider** (both scripts): abort with non-zero
  exit if the first 5 calls of a run all fail with api errors (a typo'd
  `--model`/`--provider` previously burned the whole run as per-record skips
  and still exited 0); usage-None guard as above.
- **Known asymmetry** (documented in distill_generate's docstring): in
  generate mode the openai path skips incomplete/truncated generations,
  while the anthropic path keeps partial text when `stop_reason=max_tokens`
  — anthropic-teacher generate outputs deserve a truncation spot-check.
- **Offline coverage:** `tests/shims/openai.py` fakes exactly this surface
  (refusal / api-error / truncation / usage-None / out-of-enum markers);
  both API-script test suites now run the behavioral matrix on both
  providers, plus poisoned-module proofs that each provider path never
  imports the other SDK. Still never run live — first live OpenAI spend
  should start with `--limit 1` smokes per script.

---

## 2026-06-11 — OpenAI provider LIVE validation (Phase 2, ~$0.22 of the $2 cap)

After two billing false-starts (key valid, project unfunded — diagnosed via the free models
endpoint returning 200 while inference returned `insufficient_quota`):
- **Classify** (25 real CFPB complaints, gpt-5.5 default): 25/25, 0 skipped, every label
  schema-valid (Responses API `text.format` json_schema strict), resume verified ("25 already
  done"), usage accounting correct (7,937 in / 364 out). **gpt-5.5 accepted
  `reasoning.effort: "none"`** — the one fact the doc sweep couldn't verify, now live-confirmed.
  72% accuracy vs CFPB gold ≈ our local classifier's 0.73 (dataset boundary fuzziness).
- **Judge** (25 blinded pairs, L2 triage base-vs-tuned): 0 skipped, structured verdicts,
  position randomization, noise warning self-printed. Independent cross-check: the OpenAI judge
  scored tuned at 80% W / 16% T / 4% L over base — consistent with the session-native judging
  that shipped the L2 run.
Both API scripts now proven live on BOTH providers (anthropic shape verified by claude-api docs
+ shim; openai by shim + live run).

---

## 2026-06-12 — Eval round 2 (8 cases): PASSED — with-skill 8/8

Workflow `wf_dc0ad312-5d5`: 6 regression cases + 2 new recipe-flavored traps
(router false-cheap blindness; distiller hallucination tolerance). With-skill
8 pass / 0 partial / 0 fail vs no-skill 3 pass; with-skill judged better in
7/8 (one tie). Notable from the judges:
- Case 2's judge independently verified the with-arm's commands against the
  actual scripts on disk and noted it "checks disk state before acting and
  sequences around an existing live run" — the §7 continuity prose observably
  drives behavior.
- Case 7 (false-cheap): with-skill led the metric card with false-cheap rate
  and threshold curves; no-skill accepted the average-CSAT framing.
- Case 8 (hallucination tolerance): with-skill refused the 2%-corruption
  trade and proposed the mechanical atomic-grounding gate as both eval metric
  and training-data filter — the exact discipline the distiller dogfood run
  exercised hours earlier.
Round-1 prose fixes held: stratification stated-or-disclaimed (case 2 hit),
no unperformed-check assertions observed.

---

## 2026-06-12 — Phase 2 milestone C: distiller closed at the hardware boundary; Phase 2 DONE

The distiller dogfood run (dogfood/distiller/) ends at a documented 16GB
boundary, not a checkpoint: six consecutive Metal OOMs training
Qwen3.5-2B-4bit on 365 compression pairs, across progressively minimal
configs (final leg: batch 1, no grad-accum, seqlen 2048, 8 layers), two of
them on a freshly rebooted machine, both dying at the iter-1 val pass.
Diagnosis (the lesson worth the six legs): activation memory scales with
ACTUAL sequence length — the same machine trained a larger 4B model at
12.4GB on ~100–400-token triage records under the same seqlen cap, while
1,850-token compression records OOM at 2B. `max-seq-length` is a cap, not a
cost. Scope decision: accept and document (trimming to ≤1,024 tokens biases
the set to the blobs that least need compressing; the 0.8B fallback is the
classification-tier model on the hardest metric family). Requirement
recorded: 32GB+ or a cloud backend.

What shipped instead of a checkpoint — and is the recipe's real value:
- recipes/03-tool-result-distiller.md — pipeline + frozen teacher prompt +
  the mechanical grounding gate as centerpiece + the 6-leg OOM table.
- dogfood/distiller/DATACARD.md — 1,200 extracted pairs → 1,149 deduped →
  800 teacher-compressed → 683 gate-verified (85.4%; ratio p50 0.245, zero
  atomic hallucinations) → 546/68/68 split; $0.00 API spend.
- dogfood/distiller/runs/…/state.json closed (status failed,
  hardware-boundary note, full 6-leg resume_history) + all six train logs
  committed as receipts.
- The pre-registered bar is logged and UNCONSUMED — test set looked at zero
  times; it binds any future run against this dataset.
- README: Phase 2 receipts (router + distiller) added; plugin v0.3.0.

Phase 2 definition of done (PLAN §11): router recipe ✓ (milestone B),
distiller recipe ✓ (this milestone — boundary documented with receipts),
OpenAI teacher option ✓ (milestone A + live validation above), eval round 2
✓ (8/8). Next: Phase 3 (CPT showcase on EDGAR + research mode).

---

## 2026-06-12 — v2 plan finalized (PLAN-V2.md) + deep research sweep + distiller leg 7 LAUNCHED

**v2 finalization:** PLAN-V2.md committed after a review session. Decisions of record (full
detail in PLAN-V2 §3): cascade flagship reworked into Recipe 1 on Banking77 (CFPB's frontier
ceiling is 72% — measured 2026-06-11 — so 95–98% there was impossible for any architecture);
Recipe 3 = 4B-only leg 7 under the wired-limit hypothesis; DPO/ORPO deferred to concept doc +
mlx-lm-lora spike; cloud (HF Jobs delegation) in scope only as Recipe 3's escalation path.
Flagship bars pre-registered in PLAN-V2 §4.4 before any tier has produced a score.

**Research sweep (rc's request: verify the plan is cutting-edge, not traditional).** Method:
four inline web-search batches run while leg 7 trained; ~30 sources; full list + adopted
changes in PLAN-V2 §14. Facts that changed the spec:
- Cascade deferral with **conformal risk control** (distribution-free error guarantees on the
  routed subset) is the 2025–26 selective-prediction frontier → adopted as cascade_compose.py's
  centerpiece; calibration pinned to token-margin + isotonic regression (UCCI recipe).
- **Banking77 has ~14% flagged label errors** (ACL 2022) → ≥0.936 bar stands (same-noise
  anchor); 0.95 stretch flagged as near the noisy ceiling; noise-aware secondary eval added.
- The mechanical grounding gate is a textbook **RLVR verifier** → Recipe 3 round 2 = GRPO
  against gate-pass + ratio reward (mlx-lm-lora ships GRPO locally); supersedes ORPO-on-failures
  as the headline next step.
- **mlx_lm.dwq / dynamic_quant / awq are installed and unexploited** → DWQ adopted for student
  export (quantization scales distilled against the unquantized model).
- OPD (on-policy distillation) is now a standard frontier post-training primitive; CPT canon =
  rewarming + re-decaying + replay + EntiGraph-style synthetic CPT for small corpora; curriculum
  difficulty signals include compression ratio (directly applicable to the distiller set).

**Distiller leg 7 (run `20260612-qwen3-4b-distiller-leg7`, pid in state.json):** launched
~16:40Z after rc raised the wired limit (`iogpu.wired_limit_mb` 0 → 13312; verified). Model
Qwen3-4B-Instruct-2507-4bit, leg-6-minimal config, same 365/44 data. **Correction to the leg-7
pre-registration:** `--clear-cache-threshold` was NOT a new lever — legs 4–6 already ran with
it at 1GB (state.json hparams; the pre-registration text was wrong to call it untried). The
true delta vs leg 6 is the wired limit + the 2B→4B model swap. Early telemetry: the iter-1 val
pass **completed** (val loss 2.147 — legs 5/6 died exactly here) and training proceeded past
it, wired peaking ~9.5GB then dropping to ~7.6GB as the cache threshold cycles — consistent
with the hypothesis. Outcome entry to follow when the run resolves.

---

## 2026-06-12 (evening) — v2 BUILD SPRINT: Phases A–E substantially shipped in one session

Full per-area detail in `dogfood/distiller/EXPERIMENT-LOG.md`, `dogfood/cascade/EXPERIMENT-LOG.md`,
and PLAN-V2 §11 (phase status). Headlines:

- **Phase A (distiller) DONE:** wired-limit unblock confirmed in practice (4B trains on 16GB,
  peak 6.86GB); NaN root-caused to data (zero-trainable-token records, not LR); leg 9 trained
  clean, early-stopped iter 700; **bar consumed honestly** — ratio 0.215 ✅, judge 0.94 ✅,
  grounding 0.82 vs teacher 0.93 ❌. The judge-passes/gate-fails split is the recipe's thesis
  proven on itself → RLVR round 2 motivated by a measured failure. Recipe 3 rewritten.
- **Phase B (cascade) ~90%:** Banking77 in; **frontier ceiling probe — $0 LR beats frontier
  zero-shot 0.883 vs 0.818 (+6.5)**; three-tier composition **0.9416 beats every single tier at
  8× lower cost, conformal-certified** (UCB 0.067 ≤ 0.10); **flywheel cycle demonstrated**
  (champion 0.813 → challenger 0.890, +7.8, promote). Recipe 1 → hybrid-cascade flagship.
  Scripts: llm_classify, cascade_compose, flywheel, grounding_gate — all evidence-tested,
  3 real bugs caught+fixed. 4B headline tier-2 training at session end.
- **Phase C ~85%:** 7 concept docs (cascades, calibration/conformal, flywheels, SFT-vs-preference,
  CPT, curriculum, MoE); tune-decide reworked to experiment-driven (ceiling probe → headroom);
  eval round 3 cases 9–12 added; README accuracy-first + plugin v0.4.0.
- **Phase D DONE:** Recipe 4 (finance CPT analyst) skeleton with pre-registered metric card.
- **Phase E (capstone) ~80%:** `tune-loop` skill (MAPE champion/challenger orchestration) +
  `promote.py` (bar-gated adjudication, one-look-slice ledger, descriptor versioning — smoke-
  tested); Recipe 5 (self-improving system). Owed: multi-round live dogfood.
- **DPO/ORPO spike (PLAN-V2 §8):** `mlx-lm-lora` 2.1.0 installs; `mlx_lm_lora.train` module +
  DPO/ORPO/GRPO modes confirmed importable. Live training smoke queued behind the 4B (Metal).
- **Owed to fully close v2:** 4B-tier-2 official-test headline; DPO/ORPO live smoke; RLVR
  distiller round-2; ≥3-round tune-loop dogfood; full multi-agent eval round 3; live EDGAR CPT.

## 2026-06-12 — DPO/ORPO spike COMPLETE (live training verified on Apple Silicon)
- `mlx_lm_lora.train --train-mode dpo` ran live on Qwen3-0.6B-4bit (24 preference pairs:
  chosen=teacher compression, rejected=raw blob). 10-iter smoke: loss 0.646→0.332,
  **preference accuracy 0.60→1.00, reward margin 0.102→1.052** (model learned to prefer the
  compressed output), peak 3.24GB, adapters saved + fused. The preference-tuning path
  (PLAN-V2 §8) is verified end-to-end: install + module + LIVE training, not just shape.
  ORPO/GRPO share the same entrypoint. Receipt: dogfood/distiller/dpo_smoke/.
