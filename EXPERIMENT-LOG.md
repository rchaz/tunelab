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
