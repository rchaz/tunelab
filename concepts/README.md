# Concepts — the "why" behind tunelab

Short, plain-English explainers for every idea tunelab uses. You don't need to read these to use the plugin — Claude Code explains things as it goes — but they're here when you want to actually understand what's happening.

## Start here

- [Classic ML vs LLM vs SLM](ml-vs-llm-vs-slm.md) — three different tools, and which one fits which job.
- [Why cascades work](why-cascades-work.md) — why "try the cheap model first, escalate the rest" beats using one model for everything.
- [Distillation](distillation.md) — using a big "teacher" model to train a small "student."
- [Fine-tuning vs RAG](cpt-vs-rag.md) — which one actually stores facts reliably? (Spoiler: retrieval.)

## How models get trained

- [LoRA vs QLoRA](lora-vs-qlora.md) — cheap ways to fine-tune large models on small hardware.
- [SFT vs preference tuning vs RLVR](sft-vs-preference-tuning.md) — a decision tree for *how* to fine-tune.
- [Continued pretraining](continuous-pretraining.md) — teaching a model a whole domain's language.
- [Parameters vs hyperparameters](parameters-vs-hyperparameters.md) — what the model learns vs. what you set.
- [Epochs & overfitting](epochs-and-overfitting.md) — why training too long makes a model *worse*.

## Evaluating honestly

- [Validation vs test](validation-vs-test.md) — why you need two held-out sets, and the "look once" rule.
- [Calibration & selective prediction](calibration-and-selective-prediction.md) — can you trust a model's confidence score, and when should it escalate instead of guess?
- [Label ceiling & annotator agreement](label-ceiling-and-annotator-agreement.md) — when two good labelers only agree X%, no model can truly beat X%; how to measure and read it.
- [Synthetic evals & circularity](synthetic-eval-and-circularity.md) — why a teacher-made eval can't fairly grade the teacher, and why you keep a small real-labeled anchor.

## Systems & frontier ideas

- [Data flywheels & active learning](data-flywheels-and-active-learning.md) — how a live system turns its own usage into better training data.
