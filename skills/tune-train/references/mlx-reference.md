# MLX-LM fine-tuning CLI reference (condensed)

Verified against installed **mlx-lm 0.31.3** on 2026-06-10 (Apple Silicon, `uv tool install mlx-lm`; the installed `--help` output is authoritative). Source of truth: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md — fetch it when a flag here misbehaves, but know upstream docs can be stale too: as of this writing LORA.md still documents `mlx_lm.fuse --hf-path`, which does not exist (argparse rejects it).

## Commands

| Command | Purpose |
|---|---|
| `mlx_lm.lora` | Fine-tune (LoRA / DoRA / full), evaluate test perplexity |
| `mlx_lm.fuse` | Merge adapters into the base model; optional HF upload / GGUF export |
| `mlx_lm.generate` | Generate with base model + optional `--adapter-path` |
| `mlx_lm.convert` | Quantize / convert HF models to MLX format |

`mlx_lm <subcommand>` (e.g. `mlx_lm lora --train …`) is the identical parser; `python -m mlx_lm.lora` still runs but prints a deprecation notice.

## mlx_lm.lora flags (verified defaults)

| Flag | Default | Notes |
|---|---|---|
| `--model <id-or-path>` | `Qwen/Qwen3-0.6b` | HF repo or local path. Quantized model → QLoRA automatically |
| `--train` | off | enable training |
| `--data <dir-or-hf-id>` | `mlx-community/WikiSQL` | dir containing `train.jsonl`/`valid.jsonl`/`test.jsonl`, or a HF dataset id directly |
| `--fine-tune-type` | `lora` | `lora` \| `dora` \| `full` |
| `--optimizer` | `adam` | `adam` \| `adamw` \| `muon` \| `sgd` \| `adafactor` (its help text forgets to list muon, but the choice is accepted); per-optimizer kwargs are config-only (`optimizer_config`) |
| `--iters <N>` | 1000 | training steps, not epochs |
| `--batch-size <N>` | 4 | |
| `--learning-rate <f>` | 1e-5 | LR schedules are config-only (`lr_schedule`) |
| `--num-layers <N>` | 16 | layers to adapt, from the top; `-1` = all |
| `--mask-prompt` | off | loss on completion tokens only. Chat + completions datasets ONLY — raises `ValueError` on text datasets |
| `--grad-accumulation-steps <N>` | 1 | effective batch = batch-size × N |
| `--grad-checkpoint` | off | trade compute for memory (bigger models on less RAM) |
| `--adapter-path <dir>` | `adapters` | checkpoints + `adapter_config.json` land here |
| `--save-every <N>` | 100 | checkpoint cadence — see Checkpoints & resume below |
| `--resume-adapter-file <path>` | none | **weights-only** resume — see Checkpoints & resume below |
| `--steps-per-report` | 10 | |
| `--steps-per-eval` | 200 | validation also runs at iter 1 (before training) and at the final iter |
| `--val-batches` | 25 | `-1` = entire validation set |
| `--test` | off | test-set perplexity (with `--adapter-path` to test a tuned model) |
| `--test-batches` | 500 | `-1` = entire test set |
| `--max-seq-length` | 2048 | longer examples are truncated — raise this for long records |
| `--seed` | 0 | PRNG seed |
| `--clear-cache-threshold` | 0 | clear the MLX allocator cache between steps if it grows too large — useful on long memory-tight runs |
| `--report-to` | none | `wandb`, `swanlab`, or `wandb,swanlab` (each needs its package installed) |
| `--project-name` | root dir name | logging project name for wandb/swanlab |
| `-c` / `--config <yaml>` | none | YAML config file; explicit CLI flags override config values |

## Config-only YAML keys (no CLI flags exist)

These can only be set via `-c config.yaml` — any LoRA rank/scale/dropout/target-key change (e.g. a rank sweep) must generate a config file:

```yaml
lora_parameters:        # verified defaults: rank/scale/dropout only — there is NO default `keys`
  rank: 8
  scale: 20.0
  dropout: 0.0
  # `keys` is OPTIONAL. Omitted (the default, and what every flag-form run
  # gets): mlx-lm adapts EVERY convertible module in the last --num-layers
  # blocks — all nn.Linear/QuantizedLinear/Switch/Embedding, i.e. q/k/v/o
  # plus gate/up/down. Setting keys RESTRICTS the adapter, e.g.:
  # keys: ["self_attn.q_proj", "self_attn.v_proj"]
lr_schedule:            # optional; default none (constant LR)
  name: cosine_decay
  warmup: 100
  warmup_init: 1e-7
  arguments: [1e-5, 1000, 1e-7]
optimizer_config:       # per-optimizer kwargs; default {}
  adamw:
    weight_decay: 0.01
```

Comparison gotcha: a flag-form baseline (no config → no `keys` → all projections adapted) vs a YAML run that pins `keys` to q/v changes **two** variables at once — target modules and whatever else the YAML sets. Keep `keys` identical across runs you intend to compare (rank sweeps especially).

## Data formats (one JSON object per line, auto-detected — FOUR formats)

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [...assistant turns may carry "tool_calls"...], "tools": [...]}
{"prompt": "...", "completion": "..."}
{"text": "..."}
```

- `train.jsonl` required for `--train`. **`valid.jsonl` is optional** in 0.31.3: missing/empty just prints a warning and trains without validation (older versions errored). `test.jsonl` only needed for `--test`.
- chat/tools/completions records are rendered through the model's HF chat template before tokenization.
- Completions keys are remappable via config keys `prompt_feature` / `completion_feature`; `hf_dataset` YAML blocks (single or list) support `{train,valid,test}_split` for HF-hosted data.
- `--mask-prompt` with a text dataset raises `ValueError` ("Prompt masking not supported for text dataset.").
- Unknown keys per line are ignored.

## Checkpoints & resume (this drives run continuity — read carefully)

- Every `--save-every` N iters (default 100) the trainer writes **two** files into `--adapter-path`: `adapters.safetensors` (overwritten each save — always the latest) and `{iter:07d}_adapters.safetensors` (e.g. `0000100_adapters.safetensors` — kept per save point). `adapter_config.json` is written at training start; a final `adapters.safetensors` at training end.
- Progress between save points is lost on crash.
- `--resume-adapter-file <checkpoint.safetensors>` is **weights-only**: it does `model.load_weights(..., strict=False)` before training. Optimizer state (Adam moments), LR-schedule position, and the iteration counter are NOT restored — the run restarts at iter 1 and runs the full `--iters` again.
- Correct resume recipe: track total/completed iters externally (e.g. `runs/<id>/state.json`), then relaunch with `--iters <total - completed>` and `--resume-adapter-file <latest numbered checkpoint>`. Expect a brief loss bump from the cold optimizer state.

## Fusing & export

```bash
mlx_lm.fuse --model <base> --adapter-path adapters/                 # → fused_model/ (underscore — the real default)
mlx_lm.fuse --model <base> --adapter-path adapters/ --upload-repo user/repo
mlx_lm.fuse --model <base> --adapter-path adapters/ --export-gguf --gguf-path model.gguf
mlx_lm.fuse --model <quantized-base> --adapter-path adapters/ --dequantize   # QLoRA fuse → fp16 output
```

- `--hf-path` does NOT exist on `mlx_lm.fuse` in 0.31.3 (`--upload-repo` alone uploads); upstream LORA.md is stale here.
- `--gguf-path` is resolved **inside** `--save-path`: the default output is `fused_model/ggml-model-f16.gguf`, not a cwd-relative file.
- GGUF export supports fp16 Llama/Mistral/Mixtral-style architectures only. A quantized (QLoRA) fused model raises `NotImplementedError` on GGUF export — pass `--dequantize` at fuse time, or fuse to safetensors and use llama.cpp's own converter. bfloat16 weights auto-cast to float16 during export.

## mlx_lm.generate (current flags)

```bash
mlx_lm.generate --model <id> --adapter-path adapters/ \
  --prompt "..." --max-tokens 256 --temp 0.0
```

`--prompt/-p` (`-` reads stdin), `--max-tokens/-m` (default 100), `--temp` (default 0.0), `--top-p`, `--top-k`, `--min-p`, `--seed`, `--system-prompt`, `--prefill-response`, `--ignore-chat-template`, `--extra-eos-token`; speculative decoding via `--draft-model` / `--num-draft-tokens`.

## Python API (verified signatures, 0.31.3)

```python
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("mlx-community/...", adapter_path="adapters/")  # adapter_path= confirmed
text = generate(model, tokenizer, prompt, max_tokens=512)               # max_tokens is a direct kwarg
# temperature/top-p are NOT direct kwargs — pass a sampler:
text = generate(model, tokenizer, prompt, max_tokens=512,
                sampler=make_sampler(temp=0.7, top_p=0.9))
```

- `load(path_or_hf_repo, tokenizer_config=None, model_config=None, adapter_path=None, lazy=False, return_config=False, revision=None)`
- `generate(model, tokenizer, prompt, verbose=False, **kwargs) -> str` — kwargs forward to `stream_generate` (Python default `max_tokens=256`; the CLI default is 100).
- `stream_generate(model, tokenizer, prompt, max_tokens=256, draft_model=None, **kwargs)` yields `GenerationResponse` objects.

## Verify checkpoints before downloading

mlx-community repo names churn, and instruct checkpoints often have NO base twin (most Qwen3/Qwen3.5 Base 4-bit quants do not exist). Always check before pointing a multi-GB download at a name:

```bash
curl -s -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/mlx-community/<repo>
# 200 = exists; 401 = missing or gated — the HF API returns 401, NOT 404, for absent repos when unauthenticated
```

Suffix traps: `-MLX-4bit` and `-4bit` are duplicate uploads (same weights); `OptiQ` is mixed-precision and much larger than plain 4-bit — never treat it as interchangeable.
