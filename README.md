# tunelab

Replace expensive API calls with cheap local models — classifiers, fine-tunes, cascades — and prove they work on your data before you ship them.

```
You: We classify support tickets with Claude Opus 4.8. It costs $40/day. Can we do this cheaper?

tunelab: Trained a free local classifier on your 8k tickets.
         Free model: 88.5%  —  Claude Opus 4.8: 81.8%
         The free one is more accurate. No fine-tuning needed.
```

```
You: The model outputs random text. I need it to return exact label names like "card_arrival".

tunelab: Fine-tuned Qwen2.5-0.5B on your labels. 300 steps, 2 minutes on a Mac.
         Before: "Request Information", "Loss of cash"
         After:  "card_arrival" ✓, "wrong_amount_of_cash_received" ✓
         All outputs now match your label format.
```

```
You: We send everything to Claude. Most queries are easy. Can we only send the hard ones?

tunelab: Built a 3-tier cascade: free classifier → small fine-tune → frontier.
         88% of traffic handled locally for $0. Only 12% hits the frontier model.
         Accuracy: 94% — higher than any single model alone. 8× cheaper overall.
```

---

## Quick Start

```bash
git clone https://github.com/rchaz/tunelab && cd tunelab

uv run quickstart.py cost       # free local classifier vs frontier model       ($0, any computer)
uv run quickstart.py loop       # champion/challenger with pre-registered bar   ($0, any computer)
uv run quickstart.py finetune   # LoRA on Apple Silicon                         (~2 min)
```

All demos use [Banking77](https://huggingface.co/datasets/banking77) (77-class intent classification). Only prerequisite: [uv](https://docs.astral.sh/uv/). First run downloads a ~500MB embedding model.

**As a Claude Code plugin:**
```
/plugin marketplace add rchaz/tunelab
/plugin install tunelab@tunelab
```
Then describe your problem — the right skill activates automatically.

**With another agent (Gemini CLI, Codex, any tool that reads skills):** the skills follow the [Agent Skills](https://agentskills.io/) convention. Point your agent at [`AGENTS.md`](AGENTS.md) and it can drive the same workflow.

---

## How it works

tunelab walks a ladder from cheapest to most expensive, stopping as soon as your accuracy bar is met:

| Level | What | Cost | Use case |
|---|---|---|---|
| **-1** Better prompt | Cheaper model tier, better prompt | $0 | Low volume, task still changing |
| **0** Centroids | Group by embedding similarity | ~20 examples/class | Well-separated categories |
| **1** Small classifier | Train on labeled examples | hundreds of labels, seconds | Routers, triage, high volume |
| **2** LoRA fine-tune | Teach a local model your task | 500–10k examples, minutes on a Mac | Structured output, house style |
| **3** Continued pretraining | Domain-specific language | millions of tokens | Rare — new domain vocabulary |

These are the same Level -1 … 3 the skills refer to throughout. The best systems combine levels into a **cascade** — cheap model first, escalate hard cases. A 3-tier cascade hit **94% accuracy** on Banking77 while keeping 88% of traffic local and running 8× cheaper than frontier-only.

---

## Skills

| Skill | What it does |
|---|---|
| **tune-decide** | Interviews you, runs experiments on your data, recommends the cheapest approach that clears your bar |
| **tune-data** | Turns logs/CSV/JSONL into clean training data; generates labels via teacher model if needed |
| **tune-train** | LoRA / QLoRA / full fine-tune / continued pretraining, locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx-lm) |
| **tune-eval** | Accuracy on held-out data, LLM-as-judge, cascade composition — bar set before scores are seen |
| **tune-loop** | Champion/challenger: promotes a new model only when it beats the incumbent by a pre-registered margin |

Each skill keeps an `EXPERIMENT-LOG.md` so any future session picks up where you left off.

---

## Recipes

End-to-end worked examples with real numbers:

1. **[Hybrid cascade](recipes/01-hybrid-cascade.md)** — 3-tier system (classifier → fine-tune → frontier) hits 94%, beats any single model alone
2. **[LLM auto-router](recipes/02-llm-auto-router.md)** — route each request to the cheapest viable model, ~26% cost cut
3. **[Tool-result distiller](recipes/03-tool-result-distiller.md)** — shrink tool output before it re-enters agent context (includes an honest post-mortem on why it didn't ship)
4. **[Cheap ticket triage](recipes/04-cheap-ticket-triage.md)** — free local classifier replaces frontier, with data/eval traps documented

Full run logs including failures: [`dogfood/`](dogfood/)

---

## Requirements

| Task | Hardware | Notes |
|---|---|---|
| Decide & evaluate | Any computer | No API key. Embeddings run locally |
| Train | Apple Silicon (M1+), ~8GB free RAM | Via MLX, no GPU rental |
| Generate labels at scale | Any + API key | Optional — small datasets use the Claude Code session |

**Prerequisites:** [uv](https://docs.astral.sh/uv/), Python 3.10+. Dependencies are declared inline in each script — no `pip install`, no virtualenv to manage.

**Not on a Mac?** Most of tunelab is backend-agnostic. Deciding, building data, and evaluating run on any computer (Linux, Windows, Intel Mac) — only the local *training* step uses MLX, which is Apple Silicon only. NVIDIA/Linux users still start at `tune-decide`; if a run reaches the training level, point it at your own trainer (e.g. Unsloth/PEFT) and the rest of the pipeline carries on unchanged.

---

## Concepts

Short explainers in [`concepts/`](concepts/): [cascades](concepts/why-cascades-work.md), [ML vs LLM vs SLM](concepts/ml-vs-llm-vs-slm.md), [LoRA vs QLoRA](concepts/lora-vs-qlora.md), [fine-tuning vs RAG](concepts/cpt-vs-rag.md), [overfitting](concepts/epochs-and-overfitting.md), and more.

---

## Project layout

```
quickstart.py        One-command demos (cost / loop / finetune) on Banking77
skills/              The five Claude Code skills — each is a SKILL.md + scripts/
  tune-decide/         Front door: interviews, runs experiments, recommends a level
  tune-data/           Builds, cleans, labels, and splits training data
  tune-train/          Local LoRA / QLoRA / full / continued-pretraining via MLX
  tune-eval/           Held-out accuracy, LLM-as-judge, cascade composition
  tune-loop/           Champion/challenger promotion with a pre-registered bar
concepts/            Plain-English explainers for every idea tunelab uses
recipes/             Worked end-to-end examples with real numbers
examples/banking77/  The dataset all the demos and tests run on
dogfood/             Full internal run logs, including failures
tests/               Script-level tests that invoke the real bundled scripts
AGENTS.md            Entry point for non-Claude agents (Gemini, Codex, etc.)
```

The scripts under each skill are plain, runnable Python — you can call them directly without an agent. Run any with `--help`, or use `quickstart.py --verbose` to see the exact commands a skill issues.

## Contributing

Issues and PRs welcome, including AI-assisted ones. Development needs only Python 3.10+ and [uv](https://docs.astral.sh/uv/) (Apple Silicon only for the training tests). Run the suite with:

```bash
bash tests/run_all.sh
```

First run downloads ~500MB of models into `~/.cache/huggingface`; later runs are cache-fast. See [CONTRIBUTING.md](CONTRIBUTING.md) for PR guidelines and [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

MIT
