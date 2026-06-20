# tunelab

**A Claude Code plugin that helps you replace expensive AI calls with something cheaper, faster, or more accurate — and explains every step so you actually learn how.**

If you have an app that calls a big model (Claude, GPT, etc.) for the same kind of task over and over — classifying tickets, routing requests, extracting fields, drafting replies — you're probably overpaying. Most of that work can run on a tiny model on your own machine for ~$0, and often *more accurately*. tunelab helps you find out which work that is, build the cheaper version, and prove it actually works.

You don't run anything by hand. You install the plugin, describe your problem to Claude Code in plain English, and it walks you through the rest.

---

## The one idea behind tunelab

Most people who say "I want to fine-tune a model" don't actually need to. There's a ladder of options, cheapest first:

| Option | What it is | What it costs | Good for |
|---|---|---|---|
| **Better prompt** | Improve the prompt, use a cheaper model tier | nothing | Low volume, task still changing |
| **Centroids** | Group examples by meaning, no training | ~20 examples per category | Clean, well-separated categories |
| **A small classifier** | Train a tiny model on your labeled examples | a few hundred labels, trains in seconds | Fixed categories, high volume (routers, triage) |
| **Fine-tuning (LoRA)** | Teach a small local model your task | 500–10,000 examples, minutes–hours on a Mac | Structured output, a specific writing style |
| **Continued pretraining** | Teach a model a whole domain's language | millions of words of text | Rare — a domain the model barely understands |

tunelab **starts at the top and only goes down as far as your task actually needs.** Talking you *out* of fine-tuning — by showing a cheaper rung already clears your bar — is a success, not a failure.

And here's the surprising part: you usually shouldn't pick just one rung. The best systems send each input to the *right-sized* tool — a **cascade**: try the cheap model first, and only escalate the hard cases to a bigger one. A cascade can be **cheaper *and* more accurate** than always using the biggest model, because each input is handled by the tool that's actually best at it.

---

## Why tunelab is different

- **It decides by experiment, not opinion.** Instead of guessing, it runs quick tests *on your own data* and shows you the numbers before recommending anything.
- **It runs on your Mac.** Training happens locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx-lm) — no GPU rental, no data leaving your machine. (The decision and evaluation steps work on any hardware.)
- **It's honest.** Every accuracy number comes from a real test run. The bar for "good enough" is written down *before* the test, so results can't be massaged after the fact.
- **It teaches you.** Each step explains what it's doing, why, and how to read the result. By the end you understand the process well enough to do it without tunelab. Short [concept explainers](concepts/) back up every term.

---

## Install

```bash
git clone https://github.com/rchaz/tunelab.git
cd your-project
mkdir -p .claude/skills
cp -r /path/to/tunelab/skills/* .claude/skills/
```

That's it — the skills activate automatically next time you start Claude Code in that project.

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (scripts declare their own dependencies inline, so there's nothing else to install). For training (`tune-train`), you need an Apple Silicon Mac with ~8GB free RAM.

## How to use it

Just tell Claude Code what you're trying to do. For example:

> "Our support tickets get sorted into 6 buckets by GPT-4o and it costs $40/day. Can we make this cheaper?"

> "I want to fine-tune a small model to draft replies in our company's voice."

> "Our intent classifier is stuck at 79% accuracy — can we do better?"

The right skill activates automatically and guides you from there. You'll be asked a few questions, then it starts running experiments and showing results.

---

## What's inside (the five skills)

You never need to invoke these directly — Claude Code picks the right one — but here's what each does:

| Skill | What it does |
|---|---|
| **tune-decide** | The front door. Interviews you, then runs quick experiments on your data to recommend the cheapest approach that meets your goal — with evidence. Handles the simple cases on the spot. |
| **tune-data** | Builds your training set: turns logs/CSV/JSONL into clean data, generates labels or examples using a teacher model, removes duplicates, and splits it correctly. |
| **tune-train** | Trains the model locally on your Mac (LoRA / QLoRA / full fine-tuning / continued pretraining), recommends settings, and survives interrupted runs. |
| **tune-eval** | Grades the result honestly on data it never trained on — accuracy, an LLM-as-judge, and cascade composition (finds the best cheap+accurate combination). |
| **tune-loop** | The capstone: keeps the system improving over time by automatically testing better versions against the current one and promoting only real winners. |

Each skill bundles small, runnable Python scripts (run with [uv](https://docs.astral.sh/uv/) — dependencies are declared inline), and keeps an `EXPERIMENT-LOG.md` in your project so any future session can pick up where you left off.

---

## Examples (recipes)

Worked, end-to-end examples — each one is a real run with real numbers:

1. **[Hybrid cascade](recipes/01-hybrid-cascade.md)** — the flagship. A 3-tier system (tiny classifier → small fine-tuned model → frontier model) that beats every single model alone, at a fraction of the cost.
2. **[LLM auto-router](recipes/02-llm-auto-router.md)** — send each request to the cheapest model that can handle it, cutting the bill ~26%.
3. **[Tool-result distiller](recipes/03-tool-result-distiller.md)** — shrink bulky tool output before it re-enters an agent's context (and an honest story about why this one *didn't* ship).
4. **[Finance filings analyst](recipes/04-finance-cpt-analyst.md)** — teach a model the language of SEC filings with continued pretraining.
5. **[Self-improving system](recipes/05-self-improving-system.md)** — wire it all into a loop that keeps getting better from feedback.

## Does it actually work?

Yes — and here's the headline, measured on [Banking77](https://huggingface.co/datasets/banking77) (a public dataset of 77 fine-grained banking questions), on an M1 Pro laptop with **$0 in API spend**:

> A 3-tier cascade reached **94% accuracy** — beating both a free local classifier (88%) and a frontier model used on its own (82%) — while keeping **88% of traffic local** and running **8× cheaper** than frontier-only.

The most surprising finding: on this task, the **free classifier beat the frontier model by 6.5 points.** Bigger isn't always better — the right tool for each input is. Full run logs, including every failure and dead end, live in [`dogfood/`](dogfood/). Recipe 1 walks through the whole build.

## What you need

- **To train** (`tune-train`): an Apple Silicon Mac (M1 or newer), [uv](https://docs.astral.sh/uv/), and ~8GB free RAM for 1–4B models.
- **To decide and evaluate**: nothing special — these work on any computer, no API key required. Embeddings run locally by default.
- **To generate labels at scale**: optional. For small datasets, the Claude Code session itself does the labeling for free. For large jobs, set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

## Want to understand the concepts?

The [`concepts/`](concepts/) folder has short, plain-English explainers for every idea tunelab uses — [what a cascade is and when it helps](concepts/why-cascades-work.md), [classic ML vs LLM vs small models](concepts/ml-vs-llm-vs-slm.md), [LoRA vs QLoRA](concepts/lora-vs-qlora.md), [fine-tuning vs RAG](concepts/cpt-vs-rag.md), [overfitting](concepts/epochs-and-overfitting.md), and more.

## License

MIT
