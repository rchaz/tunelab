# tunelab — fine-tuning & distillation lab for Claude Code

**Distill, train, evaluate — and learn the craft as you go.** A plugin that takes you from "should I even fine-tune?" through data preparation, distillation, local training, and rigorous evaluation — teaching you the why at every step, and honest enough to tell you when a 30-line classifier beats a fine-tuned model.

## Why another fine-tuning skill?

Existing fine-tuning skills assume fine-tuning is the answer and start at the training step. In practice, most "I want to fine-tune a model" requests are better served by something cheaper:

```
Level -1  Better prompt / cheaper API model        zero training, minutes
Level 0   Embedding centroids (no training)        ~20 examples, 30 lines of code
Level 1   Embeddings + classifier (distilled)      few hundred labels, trains in seconds
Level 2   LoRA SFT on a small model (MLX, local)   1k–10k examples, minutes–hours on a Mac
Level 3   Continued pretraining + SFT              100M+ domain tokens, rarely what you want
```

tunelab starts at the top of that ladder and only walks down as far as your task actually requires. When fine-tuning *is* the right call, it runs the whole pipeline locally on Apple Silicon via [MLX-LM](https://github.com/ml-explore/mlx-lm) — no GPU rental required.

## Skills

| Skill | What it does |
|---|---|
| `tune-decide` | Interviews you about the task, picks the right level on the ladder, and can ship Level 0/1 (embeddings + classifier) on the spot |
| `tune-data` | Builds training sets: ingest logs/CSV/JSONL, distill labels or synthetic examples from a teacher model (Claude), dedupe, stratified splits, format validation |
| `tune-train` | LoRA / QLoRA / DoRA / full fine-tuning and continued pretraining with MLX-LM: model selection, hyperparameter recommendations, training, monitoring, fusing, GGUF export |
| `tune-eval` | Held-out test evaluation: classification metrics + confusion matrix, LLM-as-judge for generative tasks, base-vs-tuned-vs-teacher comparison, ship/iterate decision |

Each skill bundles runnable scripts (PEP 723 inline dependencies — run them with `uv run`).

## Install

```
/plugin marketplace add rahulcsekaran/tunelab
/plugin install tunelab@tunelab
```

Or clone and copy `skills/*` into `.claude/skills/` in any project.

## Requirements

- **Training** (`tune-train`): Apple Silicon Mac (M1+), [uv](https://docs.astral.sh/uv/), ~8GB free RAM for 1–4B models
- **Embeddings** (`tune-decide` Levels 0/1): nothing — local by default (a static embedding model via `model2vec`, no torch, no API key). `OPENAI_API_KEY` unlocks the optional quality upgrade (`text-embedding-3-small`).
- **Distillation & LLM-as-judge** (`tune-data`, `tune-eval`): no key needed for small datasets — tunelab runs inside Claude Code, so the session itself labels and judges. `ANTHROPIC_API_KEY` is the scale path (pinned model, structured outputs, resumability).

## A typical run

> "I have 5,000 support emails that GPT-4o has been bucketing into 6 categories at $40/day. Make this cheaper."

1. `tune-decide` interviews you, concludes: fixed buckets → **Level 1**, not LoRA. Embeds your logged decisions, trains a logistic regression, reports it recovers 96% of teacher accuracy at ~zero marginal cost. Done in one session.

> "Fine-tune a small model to draft replies in our support voice."

1. `tune-decide` → open-ended generation in a fixed style → **Level 2** (LoRA SFT)
2. `tune-data` → converts 2k logged reply pairs to MLX chat format, dedupes, splits 80/10/10, validates
3. `tune-train` → recommends Qwen3-1.7B-4bit + LoRA hyperparameters scaled to your dataset, runs `mlx_lm.lora`, watches validation loss
4. `tune-eval` → LLM-as-judge compares tuned model vs base vs teacher on the untouched test split, reports win rates, recommends ship or iterate

## Roadmap

- NVIDIA/Unsloth and cloud (HF Jobs) training backends
- Preference tuning (DPO) once SFT path is solid
- Anthropic Batches API for large distillation jobs (50% cost reduction)
- Drift monitoring recipes for deployed distilled models

## License

MIT
