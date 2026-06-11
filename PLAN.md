# tunelab — Product & Implementation Plan

**Status:** Planning complete, implementation deferred to a future session.
**Date:** 2026-06-10. Authored from a planning conversation; revised same day after plan review (cloud backends §14.7, QLoRA framing, Level-1 model choice, run continuity, distillation ToS + open-weights teacher note, research-mode non-goal). This document is self-contained — a fresh session should be able to implement from it without that conversation.
**Repo state:** a throwaway prototype scaffold exists in this repo (see §10). Treat it as reference material, not a decision.

---

## 1. Vision

**One-liner:** An open-source Claude Code plugin for the full small-model development lifecycle — distillation, classic ML, LoRA fine-tuning, continued pretraining, validation, and evaluation — that teaches you what it's doing while it does it.

Three identities, all first-class:

1. **A generic pipeline tool** — every stage present: decide → prepare data → train → evaluate. Breadth is the spec ("the whole thing"), not a wedge. No stage is cut; rarely-used stages (CPT) are *sequenced later and gated*, never deleted.
2. **A cookbook of trend-relevant recipes** — the examples are marketing and on-ramps, not the product. The recipe set targets people building LLM apps (routers, agent token-diet, finance analysis) because that audience feels token costs daily.
3. **A teaching/research product** — the skills handhold the user through the entire process and explain the *why* at every step. A user who finishes a tunelab run should understand fine-tuning well enough to do it without tunelab. Research mode supports deliberately small experiments run for understanding, not production.

**What tunelab is NOT:** a training framework (it delegates to MLX-LM and friends), a vendor wrapper, or an advice-only skill — every recommendation comes with runnable evidence.

## 2. Audience

| Persona | What they come for | What they leave with |
|---|---|---|
| Developer with an LLM app and a cost/latency problem (primary) | "Make my LLM bill smaller" | A router/classifier/distilled SLM in production + the judgment to know which to reach for next time |
| Learner/researcher (first-class, not secondary) | "I want to actually understand fine-tuning" | Completed experiments with predicted-vs-actual outcomes, an experiment log, working vocabulary |
| Apple Silicon owner | "Can my Mac really do this?" | Yes — local LoRA/CPT runs with zero cloud cost |

## 3. Product principles

1. **Decision-first front door.** Validate the level before executing. Talking a user *out* of fine-tuning — by demonstrating a cheaper rung meets their bar — is a success outcome, and the trust engine of the whole product.
2. **Evidence over advice.** Don't say "a classifier would probably work" — train it in 10 minutes and show held-out accuracy next to the projected cost of LoRA.
3. **Teach while doing.** Every step states what/why/expect/how-to-read (§7). Jargon is defined on first use. The user is never a passenger.
4. **Pre-registered evaluation.** The acceptance bar is set before results are seen. The test set is looked at once. Validation steers; test judges.
5. **Local-first, vendor-neutral.** Local embeddings by default (no API key for Levels 0–1); teacher model pluggable (Anthropic + OpenAI at minimum); training local on Apple Silicon first.
6. **Delegate mechanics, own judgment.** MLX-LM runs training; tunelab owns deciding, data quality, hyperparameter reasoning, and evaluation discipline. This keeps the maintenance surface where the differentiation is.

## 4. The capability ladder (the product's spine)

First match wins, walking down:

| Level | Approach | Needs | Poster-child recipes |
|---|---|---|---|
| **-1** | Better prompt / cheaper API tier / prompt caching | nothing | (inline fix, no pipeline) |
| **0** | Embedding centroids — no training | ~10–20 examples/class | Semantic cache ("answered this before?"), duplicate detection, router cold-start |
| **1** | Embeddings + classifier (logistic regression/XGBoost) | 200+ labels (LLM logs count) | **LLM auto-router**, prompt-injection gate, agent intent/tool routing, ticket triage, PII gate, escalation prediction |
| **2** | LoRA SFT on a 1–8B model (local, MLX) | 500–10k pairs | **MCP tool-result distiller**, fast-apply model, context compactor, text-to-SQL (one schema), brand-voice replies, tool-call distillation |
| **3** | Continued pretraining + SFT (+ RAG hybrid) | ~10M+ domain tokens (relaxed in research mode) | **Finance filings analyst** (SEC EDGAR corpus) |

Escapes that are not rungs:
- **Knowledge tasks → RAG first.** Fine-tuning teaches behavior/style, not reliable facts. The production pattern at Level 3 is *CPT for domain fluency + RAG for fresh facts with citations* — and CPT's legitimate edge is latency/cost (no retrieval round-trip, shorter prompts) plus offline/on-device.
- **Open-ended reasoning → stay on the frontier model** (with Level -1 optimizations). Distillation transfers narrow behavior, not general reasoning.
- **Confidence routing everywhere:** whatever ships, low-confidence inputs route to the frontier model. The hybrid beats either alone, and routed cases are the next training data.

Aspirational unifying pitch (use in README, don't let it narrow scope): *most of what your LLM does every day doesn't need a frontier model — tunelab helps you find that work and move it to models that cost nothing to run.*

## 5. Architecture

**Form factor:** Claude Code plugin (`.claude-plugin/` + `skills/`), four skills + a recipes directory.

```
tunelab/
├── .claude-plugin/{plugin.json, marketplace.json}
├── skills/
│   ├── tune-decide/   SKILL.md + scripts/ (centroid_classify, train_classifier)
│   ├── tune-data/     SKILL.md + scripts/ (distill_generate, dedupe, split_data, validate_dataset)
│   ├── tune-train/    SKILL.md + scripts/ (recommend_hparams) + references/ (mlx-reference)
│   └── tune-eval/     SKILL.md + scripts/ (run_test_set, eval_classifier, judge_eval)
├── recipes/           one .md per cookbook entry (§9): problem → data → level → metrics → bar
├── concepts/          glossary files the skills reference for teaching (§7)
├── evals/             skill-quality test prompts (§12)
├── PLAN.md            this file
└── README.md
```

- **Scripts:** bundled in skills, PEP 723 inline deps, run via `uv run`. **Open question (§14):** extract a `tunelab` PyPI CLI at ~v1.0 once interfaces stabilize — better testing/CI, standalone non-Claude audience — but not before; bundled scripts iterate faster.
- **Pipeline artifacts** (per project, in the user's workdir): `data/{train,valid,test}.jsonl`, `data/DATACARD.md`, `adapters/`, `EXPERIMENT-LOG.md` (§7), eval reports.
- **Training backends:** MLX-LM (Apple Silicon) first. Keep `tune-train` structured so a backend section (NVIDIA/Unsloth, HF Jobs) can be added without touching decide/data/eval — those three are backend-agnostic by construction.
- **Teacher/judge providers:** Anthropic first (current default `claude-opus-4-8`), OpenAI second; provider behind a `--provider` flag in `distill_generate`/`judge_eval`. No more than two in v1.
- **Session-native teacher tier (no API key):** tunelab runs *inside* Claude Code, which is already an authenticated Claude — for small datasets (up to a few hundred items) the session model itself is the teacher: subagent fan-out labels/generates directly to JSONL, no `ANTHROPIC_API_KEY` required. The API script is the scale path (pinned model, structured-output guarantees, resumability, Batches API for >2k); subscription rate/weekly limits and the missing batch discount make session-native labeling wrong at scale. Evaluate `claude -p` headless (Agent SDK) as a scriptable middle tier at implementation time. Net effect: the "no API key" story extends past Levels 0–1 to small Level 2 datasets.
- **Embeddings:** local default (sentence-transformers, e.g. a small static or MiniLM-class model — pick at implementation time), OpenAI `text-embedding-3-small` as the quality upgrade. "Levels 0–1 need no API key" is a core OSS story.

## 6. Skills specification

### tune-decide — the front door
- Batch interview: task shape (+2–3 real I/O examples), volume & current cost, data inventory, motive (cost/latency/privacy/offline/quality), hardware.
- Apply the ladder (first match wins); present recommendation **with economics** and what evidence would change it.
- Execute Levels -1/0/1 inline (scripts bundled here); route 2/3 to tune-data → tune-train → tune-eval, *passing the interview results forward* so no skill re-asks.
- Level-1 model choice: logistic regression is the default over embedding features (fast, interpretable, calibrated probabilities — confidence routing needs them); XGBoost when non-text/tabular features join the embeddings or LR demonstrably underperforms. `train_classifier` encodes the heuristic and prints which it chose and why.
- CPT gate: stable knowledge? ≥~10M tokens? latency/offline motive? — plus an explicit **research-mode bypass**: small-corpus CPT is allowed when the goal is learning, with expectations reset (perplexity/fluency, not QA accuracy).
- Always: tell the user to log inputs/outputs/corrections from day one, and store embeddings alongside.

### tune-data — ingest → (distill) → dedupe → split → validate → datacard
- Path A: labeled data exists → convert to MLX format (chat/completions/text).
- Path B: unlabeled inputs + teacher → `distill_generate` (classify mode uses structured outputs so labels are guaranteed valid; generation mode for rich outputs). Resumable; spot-check 25 before scaling; mention Batches API for >2k items (roadmap). Teacher tier chosen by dataset size (§5): session-native (no API key) for small sets, API script at scale; DATACARD records which tier + model produced the labels.
- Path C: nothing → synthetic generation with explicit diversity instructions (personas/lengths/tones/edge cases), generate ~130%, dedupe eats the excess.
- Path D: raw domain text → chunk to `{"text"}` records for CPT (natural boundaries, 1–4k tokens).
- Dedupe: exact (normalized hash) + near-dup (char 4-gram MinHash, threshold 0.80, user/assistant content only — system prompts are boilerplate). Prints sample dropped pairs; warns on templated-data collapse (>20% flagged → suggest 0.95 + inspection). *Rationale: repetition is the dominant memorization cause.*
- Split: stratified by label, 80/10/10, seeded. Contract: valid steers, test judges once.
- Validate: hard gate (non-zero exit = do not train) — format consistency, role sanity, empties, length warnings, label distribution.
- DATACARD.md: provenance, teacher+prompt, dedupe stats, split sizes, known gaps.

### tune-train — local training on Apple Silicon
- Preflight: arm64 check, mlx-lm install, RAM → model ceiling (8GB→~3B, 16GB→~8B, 32GB+→8B comfortable).
- Model selection: smallest plausibly-capable, 4-bit mlx-community checkpoints (verify repo names live — they churn); instruct variants for SFT, base for CPT.
- Method framing: LoRA adapters over the default 4-bit checkpoints is **QLoRA in practice** (fp16 adapters on a frozen quantized base — the reason 8B trains in 16GB); loading an fp16 model gives plain LoRA. mlx-lm handles both transparently; the distinction gets a `concepts/` entry.
- Hyperparameters via `recommend_hparams` heuristics: iters = epochs×n/batch (epochs 5/3/2 by dataset size); LR 1e-4 (≤4B LoRA) / 5e-5 (7–8B) / 1e-5 (full) / 5e-6–2e-5 (CPT); `--mask-prompt` always for SFT; `--num-layers 8` under 300 examples; grad-checkpoint for big-model-small-RAM.
- Monitor, don't fire-and-forget: healthy / overfitting (val climbs — note the bottom iteration, stop there; *expected* on small data, teach it as such) / not-learning (almost always data format, then LR).
- Long-run continuity (sessions die; runs must not): launch training detached, stdout → `runs/<id>/train.log`; record PID + full command + hparams + status in `runs/<id>/state.json`. Monitoring = polling the log file, never holding the process in conversation context. Checkpoint every N iters (`--save-every`) so an interrupted run resumes weights via `--resume-adapter-file`. On every invocation tune-train first scans for an in-flight/interrupted run and offers to re-attach, re-deriving health from the log tail. The resume contract is disk-only (`state.json` + `train.log` + EXPERIMENT-LOG.md): a fresh session — or one that just compacted — picks up mid-run with zero conversation memory.
- CPT mode: text data, base model, low LR, 1–2 passes, followed by a small SFT pass to restore instruction-following.
- Post: 3–5 eyeball generations (garbage → debug, don't eval), test perplexity, fuse, optional GGUF.

### tune-eval — the honest scoreboard
- Rules: bar pre-registered; one look at test; three-way comparison (tuned vs base = did training work; tuned vs teacher = what did distillation lose; vs gold where it exists).
- Classification: per-class P/R/F1 + confusion matrix read *with* the user against error costs.
- Generative: pairwise LLM-as-judge — blinded, position-randomized, task-specific criteria (generic "which is better" rewards length), structured verdicts; <±10 points on ~100 items is noise.
- CPT: held-out domain perplexity delta + domain QA probes + catastrophic-forgetting spot-check (small general-benchmark slice before/after) + downstream-SFT lift (the real test).
- Decision matrix: ship (+ confidence routing + drift plan) / more data for weak classes / debug training / escalate level.
- Drift: the eval pipeline is the drift monitor — sample fresh production inputs monthly, teacher-label, re-score.

## 7. Teaching & research mode (the differentiator)

**Explain-why protocol** — every pipeline step the skills run is framed as four lines before and one after:
- **What** we're about to do, **Why** it matters (the failure it prevents), **Expect** — what the output should look like if healthy, **Read** — how to interpret what actually came out. After: one line connecting result → next decision.
- Jargon defined on first use, inline (epoch, LoRA rank, perplexity, stratified split…), with a pointer to `concepts/` for depth. The `concepts/` files are short standalone explainers (the curriculum mirrors this project's origin: ML vs LLM vs SLM, parameters vs hyperparameters, validation vs test, epochs/overfitting, distillation levels, CPT vs RAG).

**Pre-registration checkpoints** — the skills *stop and ask* at exactly the judgment points: the acceptance bar (before eval), the labeling prompt (after 25-sample spot-check), the level recommendation (before executing), expensive runs (before launching).

**Research mode (predict-then-run)** — first-class experiments whose purpose is understanding, run small and cheap:
- *Overfit on purpose*: tiny dataset, too many iters — watch val loss bottom and climb; now "early stopping" is felt, not read.
- *LoRA rank/layers sweep*: same data, 3 configs — see capacity vs overfitting.
- *CPT on 500K tokens*: watch domain perplexity fall while QA stays unreliable — the fluency-vs-facts lesson.
- *Dedupe ablation*: train with and without dedupe on a corpus with planted dups — observe memorization (verbatim regurgitation probes).
- Protocol: the skill asks the user to **predict the outcome first**, runs it, compares prediction vs actual in the log.
- Deliberate non-goal: autonomous overnight hill-climbing (à la karpathy/autoresearch — agent mutates config, keeps/discards on a metric, no human in the loop) is out of scope; it inverts teach-while-doing. Roadmap-only mention at most.

**EXPERIMENT-LOG.md** — append-only per-project log the skills maintain: every decision + rationale, every run + config + metrics, predictions vs actuals, lessons. (Pattern credit: unsloth-buddy's `gaslamp.md`.) This is the artifact a learner walks away with, and what makes any run reproducible. It is also the cross-session memory: every skill reads it (plus `runs/*/state.json`) on invocation before asking the user anything, so a fresh session resumes mid-pipeline from disk, not from conversation history.

## 8. Metrics framework

Global discipline lives in §6 tune-eval. Per-family metric sets — recipes pick from these and **must name their metric card before training starts** (pre-registration includes *which metrics*, not just the bar):

| Family | Core metrics | Guardrail metrics |
|---|---|---|
| Classification | accuracy, macro-F1, per-class P/R | confusion-cost weighting (user-defined expensive cells), calibration (does 0.9 confidence mean 90%?), coverage-at-threshold for routing |
| Generative (SFT) | judge win-rate vs base & vs teacher (blinded, criteria-specific) | format validity rate (JSON parse %), faithfulness/grounding, length calibration |
| Extraction/compression | field recall vs what downstream actually needed | **hallucinated-content rate** (output grounded in source — must be ~0), precision/noise (% of output unused downstream) |
| Routing | cost reduction at iso-quality; quality at iso-cost (Pareto curve) | false-cheap rate (hard query → cheap model) vs false-expensive rate; router latency overhead (ms budget) |
| CPT | held-out domain perplexity Δ; domain probe QA | catastrophic forgetting (general slice before/after); downstream SFT lift vs non-CPT base |
| Systems (every recipe) | $ per 1k calls before/after; p50/p95 latency before/after | added pipeline latency; escalation rate |

### Metric cards for the flagships

**LLM auto-router (Level 1).** Primary: *cost reduction at iso-quality* — % of traffic routed cheap while end-task quality stays within a pre-set floor (e.g. ≤1% drop vs always-frontier, measured by judge or task success on a held-out traffic sample). Report the full threshold curve (cost-quality Pareto), not one point — the threshold is a product knob. Guardrails: false-cheap rate on a hard-query probe set; router latency (<10ms target — this is why it's embeddings+LR, not an SLM); calibration of the "cheap suffices" probability. Cold start uses benchmark-informed static category→model tables; the *learned* router trains on your logged outcomes (which model sufficed), because public benchmark scores correlate loosely with your distribution.

**MCP tool-result distiller (Level 2).** Primary: *downstream answer equivalence* — agent's final answer with compressed vs raw tool results, blinded pairwise judge, target ≥98% equivalent-or-better. Plus *compression ratio* (tokens out/in; the cost win) and *field recall* — of the fields the frontier model demonstrably used in logged answers, % preserved (measurable from logs, no judge needed). Guardrails: **hallucinated-field rate ≈ 0** (every output value must be groundable in the source blob — string/value matching; a compressor that invents data is worse than no compressor); latency added vs prefill saved; report net $ per 100 agent steps.

**Fast-apply model (Level 2).** Primary: exact-match or AST-equivalence of applied edit vs ground-truth applied file; syntax-validity rate. Guardrails: latency vs frontier apply (the point is speed); failure detection (does it know when the sketch is inapplicable?).

**Finance filings analyst (Level 3).** Primary: domain perplexity Δ on held-out filings + task accuracy on the SFT layer (hawkish/dovish classification, guidance-change extraction) vs non-CPT baseline — the *lift from CPT* is the headline number. Guardrails: forgetting check; numeric-fact policy (the model never sources figures — extraction must cite spans; spot-audit). Probe sets exist off-the-shelf (FinanceBench, FinQA).

## 9. Recipe cookbook (v1 set)

Each recipe is one markdown file: problem → why this level → data source → pipeline walkthrough → metric card → pre-registered bar → cost receipts from a real run.

| # | Recipe | Level | Hook | Data source |
|---|---|---|---|---|
| 1 | Ticket/email triage (the tutorial) | 0→1 | The canonical walkthrough; simplest end-to-end | user's logs or synthetic |
| 2 | **LLM auto-router** | 1 | Route every query to the cheapest capable model | logged queries + outcomes |
| 3 | **MCP tool-result distiller** | 2 | "Stop paying frontier prices to read JSON" | agent logs (free distillation) |
| 4 | Fast-apply model | 2 | "Cursor's apply model, on your Mac" | coding-agent edit logs |
| 5 | Injection gate + semantic cache | 0–1 | Two quick wins, near-zero effort | traffic logs |
| 6 | **Finance filings analyst** | 3 | CPT showcase; reproducible by anyone (SEC EDGAR is free) | EDGAR 10-K/10-Q/transcripts |

Cluster identity: recipes 2–5 are *infrastructure for LLM apps* — router, gate, cache, distill — one coherent story (every LLM builder has these costs) rather than scattered verticals.

## 10. Current prototype state (reference, not decision)

Built in this repo during the planning session, uncommitted, treat as a sketch to mine:
- 4 SKILL.md drafts matching §6 in spirit (no teaching protocol yet — that's new).
- 10 scripts, all compile-checked; stdlib pipeline (dedupe → split → validate → hparams → eval-report) smoke-tested end-to-end on fixtures. API scripts (distill_generate, judge_eval, embeddings×2) **never run live**.
- Bugs already found & fixed by testing: word-3-gram near-dup detection misses one-word edits on short records (→ char 4-grams @ 0.80); system prompts inflate similarity (→ excluded); templated data collapses hard (→ sample-pair printing + warning).
- Verified API facts baked in: teacher/judge default `claude-opus-4-8`; structured outputs via `output_config.format` (json_schema); no sampling params on Opus 4.7+; judge uses adaptive thinking; mlx_lm CLI flags verified against ml-explore/mlx-lm LORA.md (June 2026).
- Missing vs this plan: teaching protocol (§7), concepts/, recipes/, EXPERIMENT-LOG convention, research mode, local embeddings, OpenAI teacher option, router/distiller scripts, CPT data-prep path (chunking), CPT eval (perplexity probes/forgetting check).

## 11. Roadmap

**Phase 1 — Core, breadth-first.** All four skills at spec (§6) including CPT *paths* (data chunking, train dials, eval probes); teaching protocol + concepts/ + EXPERIMENT-LOG woven into every skill; local embeddings default; recipes 1 (tutorial) written. Definition of done: full Level 1 and Level 2 runs executed for real on a live dataset; skill-creator eval loop round 1 (see §12) passed.
**Phase 2 — Flagship recipes.** Router (2) and distiller (3) end-to-end with real cost receipts in the README; OpenAI teacher option; eval loop round 2.
**Phase 3 — CPT showcase + research mode.** Finance recipe (6) on EDGAR; predict-then-run experiments shipped; forgetting/perplexity tooling hardened; cloud-backend spike (§14.7 — HF Jobs/Together or delegation to HF's skills).
**Phase 4 — Distribution.** Recipes 4–5; marketplace polish; decide CLI extraction (§14); announce (README cost receipts are the launch asset).

Each phase ends with a skill-creator eval iteration — the plugin is itself developed eval-first, which is also a credibility story.

## 12. Evaluating tunelab itself

With-skill vs no-skill subagent runs on trap prompts (drafts exist in `evals/evals.json`):
1. *Classification trap* — "fine-tune a local model for my 6-category email logs" → must route to Level 1, not LoRA.
2. *Legitimate LoRA* — "2k reply pairs, our voice, M3 24GB" → full pipeline, sane model/hparams, pre-registered bar.
3. *RAG trap* — "fine-tune to know our API docs via CPT" → RAG primary; CPT only with the research-mode framing.
Add: 4. *research-mode case* — "I want to learn what overfitting looks like" → predict-then-run experiment, not production pipeline. 5. *no-data case* — synthetic generation path with diversity instructions. 6. *hardware boundary* — NVIDIA user → honest "MLX-first today, here's what still applies" (decide/data/eval are backend-agnostic).
Plus: **dogfood case study** — one real workload end-to-end with actual dollar numbers; this is the launch README's centerpiece.

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| mlx-lm CLI/model-name churn | Skills teach *how to verify* (live repo checks) rather than hardcoding; references/ file cites source-of-truth URL |
| Maintenance gravity (backends × providers × methods) | Hard caps in v1 (1 backend, 2 providers, SFT+CPT only); decide/data/eval kept backend-agnostic |
| Teaching verbosity annoys expert users | Explain-why protocol is calibrated (one-liners, expandable via concepts/); experts can say "skip the teaching" — skills honor it |
| Templated/short-text dedupe false positives | Already mitigated: sample-pair printing + collapse warning + tunable threshold |
| Judge metrics gamed by length/polish | Criteria-specific judging, blinding, position randomization, noise thresholds — already in spec |
| Scope re-explosion (DPO, RFT, vision, …) | Roadmap-only mentions; the ladder is the scope contract |
| Distillation ToS: providers restrict using outputs to train competing models | tune-data + distillation recipes state the constraint plainly (a narrow internal task model on your own logged traffic is a different posture than a competing general model — but the user owns the call; tunelab informs, doesn't lawyer); DATACARD records teacher + intended use |

## 14. Open questions for review

1. **CLI extraction timing — RESOLVED (2026-06-10):** plugin-first confirmed by the user. The Claude Code plugin (bundled scripts as internal plumbing) is the v1 deliverable; a standalone PyPI CLI is Phase 4 and demand-gated (wait for: stable script interfaces, real non-Claude-user demand, CI/versioning need). Scripts keep clean argparse interfaces so extraction stays repackaging, not rewriting. Secondary channel: the same skills (minus plugin manifest) upload to Claude.ai/Cowork as Agent Skills.
2. **Local embedding model choice** — pick at Phase 1 (small static-embedding vs MiniLM-class; criteria: CPU speed, quality on short texts, install weight).
3. **Teacher providers** — Anthropic + OpenAI confirmed? OpenRouter as a cheap third later? Also note for the distillation docs: an open-weights teacher (Qwen Apache-2.0 / DeepSeek MIT class) is the ToS-clean option for users worried about the competing-models clause — Llama doesn't qualify (its license attaches a naming requirement to models trained on Llama outputs).
4. **Recipe 1 dataset — RESOLVED (2026-06-10):** decision rule, not a fixed dataset — use real logs when the chosen use case has them, synthetic otherwise; user preference at run time. Phase 1's definition of done still requires at least one live end-to-end run on whichever the user picks.
5. **Name — RESOLVED (2026-06-10): `tunelab`** — tune (the category searchers actually type) + lab (the teaching/research identity). Collision record from June 2026 searches: `tunesmith` rejected (songwriter SEO; names the product after the stage it often advises against); `understudy` rejected by user (too obscure); `whittle` taken (whittle-org LLM compression lib); `modelsmith` taken twice (cisco-open compression toolkit + PyPI structured-outputs lib); all `distill*` roots crowded (DistillKit, EasyDistill, LLM-Distillery, DistiLLM, and Distil Labs the company); `tinytune` taken twice. `tunelab` and `slimtune` (runner-up) both came back clean. Direct verification (2026-06-10): **PyPI available** (404), **npm available** (404), **github.com/tunelab taken** — but by an inactive user account with zero public repos and no activity; the org name is unavailable, which is harmless if publishing under a personal account (per the metadata note below, discoverability lives in repo metadata, not the URL) — only matters if a `tunelab` GitHub org was wanted (§14.6).

   **Repo metadata (apply at repo creation — discoverability lives here, not in the name):**
   - Display name: "Tunelab — fine-tuning & distillation for Claude Code"
   - GitHub description: "Claude Code plugin for LLM fine-tuning, distillation, and evaluation — decide whether you need fine-tuning at all, distill your LLM logs into small local models (MLX/LoRA), evaluate with held-out discipline, and learn the why at every step."
   - Topics: `claude-code`, `claude-code-plugin`, `fine-tuning`, `distillation`, `lora`, `mlx`, `slm`, `llm-evaluation`, `model-distillation`, `machine-learning`
   - README H1 carries the literal search phrase "fine-tuning & distillation … for Claude Code"
   - Tagline: "The fine-tuning lab for Claude Code — distill, train, evaluate, and learn the craft as you go."
6. **Repo home & license — RESOLVED (2026-06-10):** MIT (already in repo); publishes under the user's personal GitHub account (the `tunelab` org name is held by an inactive account anyway — see §14.5; discoverability lives in repo metadata, not the URL).
7. **Cloud training backend (Phase 2/3)** — candidates: HF Jobs and Together (selection criterion: **weight export** — closed vendor fine-tune APIs that keep the weights defeat the cost/privacy/ownership motives and stay out; OpenAI's FT endpoint fails this). Cheapest integration may be delegation, not ownership: Hugging Face ships official Claude Code skills that drive TRL on HF Jobs, and tune-data's JSONL chat format is what TRL consumes — tune-train could hand off to those skills rather than grow a backend section. Verify the HF skills' current state at implementation time. v1's one-backend cap (§13) holds; until then the supported cloud story is eval case 6: decide/data/eval here, train wherever, return for eval.
