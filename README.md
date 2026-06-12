# tunelab — fine-tuning & distillation lab for Claude Code

**Distill, train, evaluate — and learn the craft as you go.** A plugin that takes you from "should I even fine-tune?" through data preparation, distillation, local training, and rigorous evaluation — teaching you the why at every step, and honest enough to tell you when a 30-line classifier beats a fine-tuned model.

## Why another fine-tuning skill?

Existing fine-tuning skills assume fine-tuning is the answer and start at the training step. In practice, most "I want to fine-tune a model" requests are better served by something cheaper:

```
Level -1  Better prompt / cheaper API model        zero training, minutes
Level 0   Embedding centroids (no training)        ~20 examples, 30 lines of code
Level 1   Embeddings + classifier (distilled)      few hundred labels, trains in seconds
Level 2   LoRA SFT on a small model (MLX, local)   500–10k examples, minutes–hours on a Mac
Level 3   Continued pretraining + SFT              ~10M+ domain tokens, rarely what you want
```

tunelab starts at the top of that ladder and only walks down as far as your task actually requires. When fine-tuning *is* the right call, it runs the whole pipeline locally on Apple Silicon via [MLX-LM](https://github.com/ml-explore/mlx-lm) — no GPU rental required.

## Skills

| Skill | What it does |
|---|---|
| `tune-decide` | Interviews you about the task, picks the right level on the ladder, and can ship Level 0/1 (embeddings + classifier) on the spot |
| `tune-data` | Builds training sets: ingest logs/CSV/JSONL, distill labels or synthetic examples from a teacher model (Claude or OpenAI, or the Claude Code session itself — no key), dedupe, stratified splits, format validation |
| `tune-train` | LoRA / QLoRA / DoRA / full fine-tuning and continued pretraining with MLX-LM: model selection, hyperparameter recommendations, training, monitoring, fusing, GGUF export |
| `tune-eval` | Held-out test evaluation: classification metrics + confusion matrix, LLM-as-judge for generative tasks, base-vs-tuned-vs-teacher comparison, ship/iterate decision |

Each skill bundles runnable scripts (PEP 723 inline dependencies — run them with `uv run`), teaches with a What/Why/Expect/Read protocol backed by short [concepts/](concepts/) explainers, keeps an append-only `EXPERIMENT-LOG.md` in your project, and survives dead sessions mid-training-run (`runs/<id>/state.json` + checkpointed resume). A research mode runs predict-then-run experiments (watch overfitting happen, sweep LoRA ranks) when the goal is understanding rather than shipping.

## Real receipts (June 2026)

Every number from live runs on an M1 Pro 16GB, **$0.00 in API spend** (session-native teacher/judge):

- **Level 1** — 3,000 real CFPB consumer complaints, 10 classes, local no-torch embeddings: **0.730 held-out macro-F1** against a pre-registered 0.70 bar; ≥0.6-confidence routing keeps 90% of traffic local. Tutorial: [recipes/01-ticket-triage.md](recipes/01-ticket-triage.md).
- **Level 2** — 825 synthetic support tickets → strict-JSON triage on Qwen3-4B (QLoRA, early-stopped at the val bottom): **100% format validity, 92.2% category accuracy vs teacher, 68.9% judged equivalent-or-better** — all three pre-registered bars passed on a fresh test set. The tuned student beat its own teacher on 5 of 90 blinded pairs.
- **LLM auto-router** — trained on the author's own logged agent traffic (7,794 events): **39.0% of events routed cheap at zero observed false-cheap** on a fresh 60-family holdout, an estimated **26.4% bill reduction**, ~0.6ms router latency. Including the round-1 bar *failure* (a template-twin false-cheap) and the fix: [recipes/02-llm-auto-router.md](recipes/02-llm-auto-router.md).
- **Tool-result distiller** — 683 verified training pairs distilled free from real agent logs: teacher compression **p50 ratio 0.245 with zero atomic hallucinations**, enforced by a mechanical grounding gate (85.4% of teacher outputs passed; the rest were dropped, not trained on). Training then hit a real wall — six consecutive Metal OOMs at 16GB — and the recipe documents the boundary instead of hiding it: [recipes/03-tool-result-distiller.md](recipes/03-tool-result-distiller.md).
- Full logs, including the failures (a wired-memory freeze, a discarded full-LR resume leg, six OOM legs): [dogfood/](dogfood/).

## Install

```
/plugin marketplace add rchaz/tunelab
/plugin install tunelab@tunelab
```

Or clone and copy `skills/*` into `.claude/skills/` in any project.

## Requirements

- **Training** (`tune-train`): Apple Silicon Mac (M1+), [uv](https://docs.astral.sh/uv/), ~8GB free RAM for 1–4B models
- **Embeddings** (`tune-decide` Levels 0/1): nothing — local by default (a static embedding model via `model2vec`, no torch, no API key). `OPENAI_API_KEY` unlocks the optional quality upgrade (`text-embedding-3-small`).
- **Distillation & LLM-as-judge** (`tune-data`, `tune-eval`): no key needed for small datasets — tunelab runs inside Claude Code, so the session itself labels and judges. At scale: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (`--provider anthropic|openai` — pinned model, structured outputs, resumability).

## A typical run

> "I have 5,000 support emails that GPT-4o has been bucketing into 6 categories at $40/day. Make this cheaper."

1. `tune-decide` interviews you, concludes: fixed buckets → **Level 1**, not LoRA. Embeds your logged decisions locally, trains a logistic regression in seconds, reports honest held-out metrics against the bar you set *before* seeing the score, and ships with confidence routing back to the frontier model. Done in one session.

> "Fine-tune a small model to draft replies in our support voice."

1. `tune-decide` → open-ended generation in a fixed style → **Level 2** (LoRA SFT)
2. `tune-data` → converts 2k logged reply pairs to MLX chat format, dedupes, splits 80/10/10, validates
3. `tune-train` → recommends a current verified 4-bit checkpoint + hyperparameters scaled to your dataset, launches `mlx_lm.lora` detached with checkpointed resume, monitors validation loss for the early-stopping point
4. `tune-eval` → LLM-as-judge compares tuned model vs base vs teacher on the untouched test split, reports win rates, recommends ship or iterate

## Roadmap

- **Next (Phase 3):** CPT showcase on SEC EDGAR (finance filings analyst) + research-mode experiment pack
- NVIDIA/Unsloth and cloud (HF Jobs) training backends — also the unblock for recipe 3's 16GB training boundary
- Anthropic Batches API for large distillation jobs (50% cost reduction); DPO once the SFT path is battle-tested
- Drift monitoring recipes for deployed distilled models

## License

MIT
