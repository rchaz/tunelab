# tunelab — fine-tuning & distillation lab for Claude Code

**Build AI systems that beat any single model — then keep getting better.** A plugin that takes
you from "should I even fine-tune?" through data preparation, distillation, local training,
rigorous evaluation, and **hybrid cascades with a data flywheel** — teaching you the why at every
step. The headline isn't "same accuracy, cheaper." It's *higher accuracy than any single
approach can reach* — and then a feedback loop that improves it over time.

Measured on Banking77 (public, external gold): a three-tier cascade hit **0.94 accuracy — beating
both a $0 classifier (0.88) and a frontier model used zero-shot (0.82) — at 8× lower cost than
frontier-only, with a conformal-certified error guarantee.** Not asserted; composed from real
per-tier runs. ([recipes/01-hybrid-cascade.md](recipes/01-hybrid-cascade.md))

## Why another fine-tuning skill?

Existing fine-tuning skills assume fine-tuning is the answer and start at the training step. In
practice, the right answer is rarely one model — it's the *right tier for each input*, composed
into a cascade and improved by a flywheel. tunelab finds that composition by measurement. And
most "I want to fine-tune" requests are better served by something cheaper first:

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
| `tune-decide` | Interviews you, then **runs experiments on your data** — cheap probes plus a frontier *ceiling probe* (headroom = ceiling − floor) — and recommends an architecture with evidence, not a static ladder pick. Ships Level 0/1 on the spot |
| `tune-data` | Builds training sets: ingest logs/CSV/JSONL, distill labels or synthetic examples from a teacher model (Claude or OpenAI, or the Claude Code session itself — no key), dedupe, stratified splits, format validation, and the **flywheel** (prediction log → drift/triggers → retrain manifest) |
| `tune-train` | LoRA / QLoRA / DoRA / full fine-tuning and continued pretraining with MLX-LM: model selection, hyperparameter recommendations, training, monitoring, fusing, GGUF export |
| `tune-eval` | Held-out evaluation: classification metrics, LLM-as-judge, base-vs-tuned-vs-teacher — plus **cascade composition** (`cascade_compose`: all architectures × thresholds, isotonic calibration, conformal-certified operating points) and a **mechanical grounding gate** |
| `tune-loop` | **The capstone**: drives champion/challenger rounds (Monitor→Analyze→Plan→Execute) using the other four skills as tools — the self-improving system |

Each skill bundles runnable scripts (PEP 723 inline dependencies — run them with `uv run`), teaches with a What/Why/Expect/Read protocol backed by short [concepts/](concepts/) explainers, keeps an append-only `EXPERIMENT-LOG.md` in your project, and survives dead sessions mid-training-run (`runs/<id>/state.json` + checkpointed resume). A research mode runs predict-then-run experiments (watch overfitting happen, sweep LoRA ranks) when the goal is understanding rather than shipping.

## Real receipts (June 2026)

Every number from live runs on an M1 Pro 16GB, **$0.00 in API spend** (session-native teacher/judge):

- **Hybrid cascade (the flagship)** — Banking77, 77 fine-grained intents, three tiers (LR →
  fine-tuned SLM → frontier w/ kNN few-shot): the composed cascade hit **0.9416 — beating the $0
  classifier (0.883), the frontier zero-shot (0.818), and the tiny fine-tune (0.513) — at $0.25
  vs $2.00 per 1k (8× cheaper), 87.7% of traffic local, conformal-certified** (kept-set error ≤
  0.10 at 95% confidence). The killer finding: *the $0 classifier beats the frontier model
  zero-shot by 6.5 points* on fine-grained classification. [recipes/01-hybrid-cascade.md](recipes/01-hybrid-cascade.md).
- **Data flywheel** — the cascade improves itself: a champion trained on a starved 2,000-record
  slice (0.813) was beaten by a challenger retrained after one feedback cycle (**0.890, +7.8
  points**) → promote. Bias-aware (audit slice vs feedback pile), trigger-gated, champion/challenger.
- **LLM auto-router** — trained on the author's own logged agent traffic (7,794 events): **39.0% of events routed cheap at zero observed false-cheap** on a fresh 60-family holdout, an estimated **26.4% bill reduction**, ~0.6ms router latency. Including the round-1 bar *failure* (a template-twin false-cheap) and the fix: [recipes/02-llm-auto-router.md](recipes/02-llm-auto-router.md).
- **Tool-result distiller** — 683 verified pairs distilled free from real agent logs; the SFT
  student compresses well (**ratio 0.215**) and a blinded frontier judge rates it **equivalent to
  its teacher (0.94)** — but the mechanical grounding gate catches it inventing identifiers ~18%
  of the time (**0.82 vs teacher 0.93**), so it fails the non-negotiable grounding bar. *The gate
  catches what the judge's eye misses* — which is the whole point, and it defines the RLVR round
  that fixes it. (The six-OOM "boundary" fell to one `sysctl`: 4B now trains on 16GB.)
  [recipes/03-tool-result-distiller.md](recipes/03-tool-result-distiller.md).
- Full logs, including every failure and root-cause (NaN-from-data, six OOM legs, the bar that failed honestly): [dogfood/](dogfood/).

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

> "Our intent classifier is stuck at 79% and we want higher accuracy, not just cheaper."

1. `tune-decide` runs the probes on *your* data: a $0 classifier (the floor) and a frontier **ceiling probe** (what any model can reach on your gold). It reads the headroom — and if the frontier itself only reaches ~80%, it tells you the *labels* are the ceiling, not the model. If there's real headroom, it builds a **cascade**: `cascade_compose` simulates ML-only, fine-tuned-only, frontier-only, and every cascade, then recommends the accuracy/cost/latency winner with a **conformal-certified** operating point. Then the **flywheel** keeps it improving from feedback. Higher accuracy than any single approach — with the evidence.

> "I have 5,000 support emails that GPT-4o has been bucketing into 6 categories at $40/day. Make this cheaper."

1. `tune-decide` interviews you, concludes: fixed buckets → **Level 1**, not LoRA. Embeds your logged decisions locally, trains a logistic regression in seconds, reports honest held-out metrics against the bar you set *before* seeing the score, and ships with confidence routing back to the frontier model. Done in one session.

> "Fine-tune a small model to draft replies in our support voice."

1. `tune-decide` → open-ended generation in a fixed style → **Level 2** (LoRA SFT)
2. `tune-data` → converts 2k logged reply pairs to MLX chat format, dedupes, splits 80/10/10, validates
3. `tune-train` → recommends a current verified 4-bit checkpoint + hyperparameters scaled to your dataset, launches `mlx_lm.lora` detached with checkpointed resume, monitors validation loss for the early-stopping point
4. `tune-eval` → LLM-as-judge compares tuned model vs base vs teacher on the untouched test split, reports win rates, recommends ship or iterate

## Roadmap

- **The capstone (Recipe 5 + `tune-loop`):** the self-improving system — champion/challenger
  architecture search driven by the data flywheel, with pre-registered promotion bars and
  one-look eval slices (the discipline that separates it from AutoML slop). Designed; in build.
- **RLVR/GRPO round for the distiller** — the grounding gate as a verifiable reward, to teach the
  grounding discipline SFT didn't transfer (`mlx-lm-lora`, local).
- **CPT showcase on SEC EDGAR** (finance filings analyst) as a *maintained system* — corpus
  refresh → incremental CPT → SFT restore → eval gates.
- NVIDIA/Unsloth and cloud (HF Jobs) training backends; Anthropic Batches API for large
  distillation jobs; DPO/ORPO once the spike verifies.

## License

MIT
