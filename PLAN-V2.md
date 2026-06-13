# tunelab v2 — Plan: hybrid cascades, the data flywheel, experiment-driven decisions

**Status:** Finalized 2026-06-12 from a review session (brief → receipts-check → four user decisions
→ two follow-up adjustments). Implementation phased A→D below. This document is self-contained —
a fresh session implements from it plus PLAN.md (v1, still authoritative for everything v2 doesn't
change).
**Prime directive carried over from v1:** every number from a real run, every bar pre-registered,
the test set looked at once. v2 changes *what* tunelab builds, not how it earns trust.

---

## 1. What v2 changes

1. **Accuracy-first positioning.** The business case is "exceeds what any single approach can do,"
   not "matches frontier at lower cost." Priority order: accuracy → cost (without sacrificing
   accuracy) → latency → novelty. README and recipes reframe accordingly — without rewriting any
   honest historical receipt.
2. **The hybrid cascade becomes a first-class capability** (Recipe 1, reworked in place):
   ML classifier → fine-tuned SLM → frontier LLM, each tier handling what the previous tier isn't
   confident about, with the architecture *chosen by measurement, not belief*.
3. **The data flywheel becomes a concrete artifact**: a prediction/feedback log schema, retrain
   triggers, and a demonstrated retrain cycle with before/after numbers — not a "log your data"
   paragraph.
4. **tune-decide runs experiments, not just interviews.** The interview survives (it gates what's
   worth compute), but its output becomes an experiment plan; the recommendation comes from
   measured probes on the user's data.
5. **Concepts catch up to 2025–26 practice**: cascades/selective prediction, calibration,
   flywheels/active learning, preference tuning (SFT vs DPO/ORPO), MoE routing, curriculum
   training, continuous pretraining as a *system*.

## 2. Corrections to the v2 brief (logged so nobody re-litigates them)

The original v2 brief contained three claims the repo's own receipts contradict. Decisions below
already account for these.

1. **The Recipe 3 OOM was on Qwen3.5-2B-4bit, not Qwen3-4B** — and the six-leg diagnosis stands:
   activation memory scales with *actual* record token length (1,850-token compression records),
   not the seqlen cap and not model size alone. The same 16GB machine trained Qwen3-4B fine on
   100–400-token records (12.4GB peak). "Stick with 4B" therefore *raises* the memory bar on the
   task that OOM'd at 2B. The user decided 4B-only anyway (§5) — viable only because the six legs
   never touched the most promising lever (the Metal wired-memory limit, §5.1).
2. **95–98% accuracy against CFPB gold is impossible for any architecture.** Live-measured
   2026-06-11: gpt-5.5 scores **72%** vs CFPB gold — statistically the same as the $0 logistic
   regression (73%). That's label fuzziness, not model weakness; a cascade cannot exceed its
   strongest tier on noisy gold. Hence the flagship dataset swap (§4.1).
3. **"A fine-tuned LLM beats a general LLM on the domain" is a hypothesis, not a premise.**
   Published Banking77 numbers support it (fine-tuned ~93.7% vs zero-shot frontier trailing by
   ~19–26 points — see §4.1 sources), but v2 measures it on our own runs and lets the
   architecture comparison decide the cascade's composition. If a tier doesn't earn its slot,
   the system says so — that's the feature working.

## 3. Decisions of record (2026-06-12)

| # | Decision | Choice | Notes |
|---|---|---|---|
| 1 | Flagship dataset/metric | **Clean public benchmark (Banking77)** — external gold, real tier separation | CFPB stays as the dogfood/level1 historical receipt |
| 2 | Recipe 1 fate | **Rework in place** — Recipe 1 becomes the cascade flagship | The Level-0→1 ladder narrative survives *inside* the cascade story (tier 1 IS Level 1) |
| 3 | Recipe 3 OOM strategy | **Qwen3-4B only**, wired-limit + cache levers, pre-registered leg 7 | If it fails: cloud escalation (HF Jobs delegation), not another boundary narrative |
| 4 | DPO/ORPO scope | **Concept doc + verification spike now; full tune-train integration next phase** | Downgraded from "full capability now" in follow-up discussion |
| 5 | Cloud fine-tuning | **In scope as Recipe 3's escalation path + tune-train escape-hatch section** | Local-first stays the brand; weight-export criterion from PLAN.md §14.7 governs |

## 4. Recipe 1 rework — the hybrid cascade flagship

### 4.1 Dataset and verified anchors

**Banking77** (`PolyAI/banking77`, Hugging Face): 13,083 train / 3,080 test customer-service
queries, 77 fine-grained banking intents. Published anchors (verified 2026-06-12 via web search;
re-verify exact splits/license at implementation):

- Fine-tuned BERT 93.66% / RoBERTa-base 93.86% accuracy (the "best single fine-tuned model" anchor).
- Zero-shot frontier LLMs trail fine-tuned models by ~19–26 points on this dataset.
- Retrieval-augmented few-shot prompting lifts GPT-4o to slightly *above* the fine-tuned baseline —
  which is why tier 3 must be implemented well (§4.2), not as a bare zero-shot call.

Why this dataset: enough headroom between an embeddings+LR floor and a ~94% ceiling for every
tier to demonstrably earn or lose its slot, against external gold nobody here controls.

**Live measurement (2026-06-12, `dogfood/cascade/`), which reframes the flagship narrative:**
on identical stratified validation records, tier-1 LR = **0.8831**, frontier zero-shot
(session-native) = **0.8182** — the **$0 classifier beats the frontier by 6.5 points**. This is
the headline (adopted, replacing the brief's "95–98%"): *most everyday classification doesn't
need a frontier model; a tiny local fine-tune closes the rest of the gap to the 0.937 anchor.*
Two consequences baked into the design: (a) tier-3 **must** use kNN few-shot — bare zero-shot
sits below the tier-1 floor and would lower cascade accuracy; (b) the "fine-tuned beats general"
hypothesis is already supported pre-training — the real headroom (~+5 pts) lives in the
fine-tuned tier, so the ≥0.936 bar hinges on tier-2 quality.

### 4.2 The three tiers

| Tier | Model | Confidence signal | Expected cost/latency profile |
|---|---|---|---|
| 1 | Static embeddings (model2vec) + logistic regression | Calibrated class probability (native) | ~1ms, ~$0 |
| 2 | Qwen3-4B-Instruct-2507-4bit + QLoRA SFT (short records — the proven memory shape from dogfood/level2) | Label-token logprob margin via mlx-lm Python API, calibrated on validation | ~0.3–1s, ~$0 local |
| 3 | Frontier (session-native default; OpenAI optional) with **kNN few-shot prompting** — reuse tier 1's embedding index to pull the k nearest labeled examples into the prompt | n/a (terminal tier) | seconds, $ |

Tier-2 confidence is the genuine research bit: an SLM gives nothing calibrated for free. The
implementation scores generated label tokens (margin between top candidates), then calibrates the
threshold on validation. This machinery gets its own script and its own concept doc (§7).

### 4.3 Experiment design — offline counterfactual composition

**Run each tier once over the same evaluation sets; compose architectures offline.** Every tier
produces `{record_id, prediction, confidence, latency, cost}` on identical validation and test
sets. A single analysis script then simulates *every* architecture — ML-only, FT-only,
frontier-only, ML+FT, ML+frontier, ML+FT+frontier — across the full grid of confidence thresholds,
producing the accuracy/cost/latency Pareto surface from three prediction files. Architecture
comparison becomes arithmetic, not five live systems. Threshold/architecture selection happens on
**validation**; the untouched test set is consumed once, by the selected architecture and the
single-tier baselines, together.

Sub-experiments (pre-registered alongside the main bar):
- Tier 2 trained on the **full train set vs the low-confidence residual slice** — residual-only
  risks data starvation and boundary mismatch; let the data decide.
- Tier 3 bare zero-shot vs kNN few-shot (quantifies what the published RAP result claims, on our
  tiers).

### 4.4 Pre-registered bars (binding once rc confirms; veto window per the standing "continue, don't wait" delegation)

- **Primary:** on the untouched Banking77 test set, the validation-selected architecture achieves
  accuracy **≥ the best single-tier baseline measured in the same run** AND **≥ 0.936** (the
  published fine-tuned-BERT anchor). Beating every single approach is the headline claim — it must
  be true on our own numbers.
- **Stretch (reported, not pass/fail):** ≥ 0.95.
- **Guardrails:** frontier-tier share ≤ 15% of test traffic at the selected thresholds; tier-1+2
  (local) share ≥ 85%; full cost+latency accounting per 1k queries vs frontier-only; calibration
  check on both confidence signals (does 0.9 mean ~90%?).
- **Metric card family:** classification + routing (PLAN.md §8) — accuracy, macro-F1, per-tier
  coverage/accuracy, Pareto curve, false-escalation/false-local rates.

API budget: $0 target via session-native tier 3 (fan-out batches, the dogfood pattern); hard cap
$5 (gpt-5.4-mini fallback for scale) — whichever, the DATACARD records tier/model provenance.

### 4.5 The data flywheel (concrete artifact)

- **Prediction log schema** (JSONL, append-only; SQLite optional later):
  `{ts, input_hash, text_ref, tier, prediction, confidence, latency_ms, cost_usd, feedback,
  feedback_source, model_version}` — every prediction from every tier, including tier-3 outputs
  (which are free training labels by construction).
- **`flywheel.py`**: ingests the log + new feedback; computes drift stats (confidence-distribution
  shift, accuracy on the feedback window); evaluates retrain triggers (≥N new labels · drift past
  threshold · scheduled); emits a retrain manifest (which tier, what data) rather than silently
  retraining — the user stays in the loop, the skill executes.
- **One demonstrated retrain cycle in the dogfood run:** round 1 ships, routed/feedback cases
  accumulate (simulated via held-back gold, *labeled as simulated*), trigger fires, tier 1 and/or
  tier 2 retrain, before/after accuracy reported. The delta is the receipt that the flywheel turns.
- Test-set hygiene across rounds is part of the design: retraining consumes feedback data, never
  test data; each round's eval uses fresh held-out slices (the dogfood/level2 "one-look rule"
  precedent).

### 4.6 New scripts (PEP 723, `uv run`, evidence-tested like milestone 2)

| Script | Home | Job |
|---|---|---|
| `llm_classify.py` | tune-eval | MLX model classification over JSONL with label-logprob confidence + latency per record; works for base and tuned (adapter) models |
| `cascade_compose.py` | tune-eval | Offline composition: tier prediction files → all architectures × threshold grid → Pareto table + validation-selected recommendation |
| `flywheel.py` | tune-data | Log schema init, feedback ingest, drift stats, retrain-trigger evaluation, retrain manifest |
| (reuse) `distill_generate.py --mode classify` | tune-data | Tier-3 frontier predictions + the ceiling probe (§6) |

## 5. Recipe 3 — leg 7: the wired-limit hypothesis, 4B-only

### 5.1 Hypothesis and levers

**Hypothesis:** the six Metal OOMs hit the **default Metal wired-memory limit** (~⅔ of RAM ≈
10–11GB on a 16GB machine), not the machine's total memory. Evidence consistent: the error class
(`kIOGPUCommandBufferCallbackErrorOutOfMemory`) is the wired-limit error; legs died at the iter-1
val pass with the system otherwise healthy; the 4B triage run peaked at 12.4GB *reported* with
system freeze symptoms (wired pressure) rather than process OOM. The six legs varied
batch/accum/seqlen/layers — all the wrong knobs if the wired limit binds.

Levers never tried (verified available 2026-06-12):
1. `sudo sysctl iogpu.wired_limit_mb=13312` — raise the GPU wired limit to ~13GB, reserving
   ~2.7GB for macOS. Runtime-settable, **reverts on reboot** (low-risk), needs rc present for sudo.
   Community-documented unblock for exactly this error class on 16GB Apple Silicon.
2. `--clear-cache-threshold` (mlx-lm flag, flagged "matters" in the milestone-1 verification but
   absent from all six legs) + MLX cache-limit controls — Metal buffer-cache fragmentation is a
   known OOM contributor distinct from true allocation pressure.
3. Before training: re-check the installed mlx-lm version against upstream for memory fixes since
   0.31.3 (same live-verification method as milestone 1).

### 5.2 Protocol (pre-register before launching)

- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (user decision: 4B only). Data: the frozen
  683-pair verified dataset, 546/68/68 split, **test untouched — the 2026-06-11 pre-registered
  bar is logged and UNCONSUMED and binds this run as-is** (hallucination ≈ 0 via the same gate,
  field recall ≥ 90%, ratio ≤ 0.35, judge ≥ 60% vs teacher).
- Leg 7 config: start from leg-6 minimalism (batch 1, no accum, seqlen 2048, 8 layers,
  `--grad-checkpoint`) + raised wired limit + cache threshold; record `iogpu.wired_limit_mb`,
  wired-memory telemetry, and peak mem in state.json. Predict-then-run: log the predicted outcome
  first (the §7 research-mode discipline, applied to ourselves).
- Pass → scale config upward only as memory headroom proves out (layers 8→16, then seqlen), full
  training to the val bottom, eval against the unconsumed bar. The recipe's OOM table gains a leg-7
  row and the "hardware boundary" section becomes "the boundary was a sysctl — here's how to find
  yours" (the lesson *upgrades*, it doesn't vanish).
- Fail (4B at this token profile doesn't fit even at ~13GB wired) → **cloud escalation, not a
  boundary narrative**: hand the (already TRL-format) JSONL to HF Jobs — delegation to Hugging
  Face's official Claude Code skills if their current state verifies live, else a minimal job
  submission — train there (weight export = the §14.7 criterion), return to local tune-eval
  against the same unconsumed bar. Either way Recipe 3 ends with a student model and a consumed
  bar this time.
- Cleanup: restore the wired limit (reboot or sysctl back) and document the revert in the recipe.

## 6. tune-decide rework — interview in, experiment plan out

- **Keep the interview** (volume, economics, data inventory, hardware, motive — no experiment
  answers those). Its output becomes an **experiment plan**, not a verdict.
- **Always-run cheap probes** on the user's actual data: Level 0 centroid probe, Level 1 LR
  (minutes, $0) — *plus the new frontier ceiling probe*: frontier model on ~150 validation items.
  The ceiling bounds what ANY architecture can achieve on the user's gold labels; LR sets the
  floor; **headroom = ceiling − floor** is the budget that justifies (or kills) the fine-tuned
  tier before hours of training are spent.
- The label-noise conversation becomes mechanical: "the frontier only reaches 78% on your gold —
  your labels, not your model, are the constraint" (the CFPB lesson, productized).
- The ladder survives as the **experiment schedule** (cheap rungs first, escalate on measured
  headroom); "classification never starts at Level 2" survives as "Level 2 earns its place on the
  residual." Brand intact: tunelab still talks users out of fine-tuning — now with their own data
  as the argument.
- Output artifact: experiment-plan entry in EXPERIMENT-LOG.md (probes → results → architecture
  recommendation with the Pareto table → what evidence would change it).

## 7. Concepts — seven new/updated docs

| Doc | Core teaching |
|---|---|
| `why-cascades-work.md` | Selective prediction, cascade theory, speculative decoding as the same cheap-first pattern; when cascades beat single models and when they can't (noisy-gold ceiling) |
| `calibration-and-selective-prediction.md` | Why LR probabilities route and raw LLM logits don't; calibration checks; threshold selection on validation — the load-bearing doc for v2 |
| `data-flywheels-and-active-learning.md` | Feedback loops, routed-cases-as-labels, retrain triggers, drift; uncertainty sampling as free active learning |
| `sft-vs-preference-tuning.md` | SFT vs RLHF vs DPO/ORPO: what each optimizes, data each needs, when SFT suffices (most tunelab tasks), when preference pairs pay |
| `continuous-pretraining.md` | CPT as a *system* (corpus refresh → incremental CPT → SFT restore → eval gates), not a one-shot; extends cpt-vs-rag.md |
| `curriculum-and-progressive-training.md` | Ordering/staging strategies beyond vanilla SFT; honest about thin evidence at small scale |
| `moe-routing.md` | Short: experts-inside-a-model vs models-inside-a-system; why the cascade is "MoE at the system level" only as analogy |

## 8. Preference tuning (deferred capability, grounded now)

- **This cycle:** the concept doc (§7) + a timeboxed **verification spike**: install `mlx-lm-lora`
  (v2.1.0, 2026-04; `mlx_lm_lora.train --train-mode dpo|orpo`, `{prompt, chosen, rejected}`
  format), smoke a tiny DPO and ORPO run on Apple Silicon, log verified facts to
  EXPERIMENT-LOG.md exactly as milestone 1 did for mlx-lm. No tune-train integration yet.
- **Next phase:** wire as a tune-train path if the spike verifies. Natural dogfood dataset already
  exists: the distiller's 683 gate-passed (chosen) vs 117 gate-failed (rejected) teacher outputs —
  ORPO-for-grounding with zero new data collection.

## 9. Continuous pretraining capability (plan only — Recipe 4 direction)

Extends the Phase 3 EDGAR showcase from one-shot to a system: scheduled corpus refresh → delta
chunking (tune-data Path D) → incremental CPT at low LR → small SFT restore pass → eval gates
(domain perplexity Δ, forgetting slice, downstream-SFT lift) → flywheel integration (the
prediction log feeds the next corpus round). Deliverable in v2: design doc + Recipe 4 skeleton
with pre-registered metric card; live runs are Phase 3 scope as before.

## 10. Eval round 3 (skill-quality, §12 discipline)

Regression cases from rounds 1–2, plus new traps:
1. *Noisy-gold trap* — "get me 99% on this dataset" where the ceiling probe reveals ~80% frontier
   accuracy → must surface label noise, not promise training.
2. *Skip-the-ML-tier trap* — "LLMs are smarter, just fine-tune" → must run the probes and show the
   cascade math.
3. *Flywheel hygiene trap* — retraining on feedback that includes test rows → must catch the
   leakage.
4. *Wired-limit trap* — Metal OOM report on 16GB → must reach for sysctl/cache levers and actual
   token-length analysis before declaring hardware boundaries.

## 11. Phasing — STATUS (updated 2026-06-12)

| Phase | Scope | Status |
|---|---|---|
| **A** | Recipe 3 distiller: train Qwen3-4B, consume the bar, rewrite | ✅ **DONE** — wired-limit unblocked 4B; leg 9 trained clean; bar consumed (ratio✅ judge✅ grounding❌→RLVR); recipe rewritten |
| **B** | Cascade flagship: Banking77, 3 tiers, composition, flywheel, Recipe 1 | ✅ **~90%** — all tiers, composition table (0.9416 beats every tier, conformal-certified), ceiling finding, flywheel cycle (+7.8pts), Recipe 1 rewritten. Owed: 4B-tier-2 official-test headline number |
| **C** | tune-decide rework + 7 concepts + DPO/ORPO spike + eval round 3 + README | ✅ **~85%** — 7 concept docs✅, tune-decide experiment-rework✅, eval round 3 cases (9–12) added✅, README accuracy-first✅. Owed: DPO/ORPO live spike (Metal), full multi-agent eval-round-3 run |
| **D** | CPT continuous-capability design doc (Recipe 4 skeleton) | ✅ **DONE** — Recipe 4 skeleton with pre-registered metric card |
| **E** | Capstone: Recipe 5 + tune-loop + champion/challenger | ✅ **~80%** — tune-loop skill✅, promote.py✅ (smoke-tested), Recipe 5✅, flywheel cycle demonstrated✅. Owed: ≥3 autonomous rounds on the replay stream + 1 on real router traffic |

Remaining to fully close v2: the 4B-tier-2 official-test headline number; the DPO/ORPO live
spike; multi-round `tune-loop` dogfood (≥3 rounds); the full multi-agent eval-round-3 run; the
RLVR distiller round-2; and the live EDGAR CPT run (Recipe 4). Everything else shipped 2026-06-12.

## 12. Constraints (v1 constraints hold, plus)

- Apple Silicon M1 Pro 16GB, MLX-LM, `uv run`/PEP 723, session-native teacher/judge preferred.
- API spend: $0 target, $5 hard cap for v2 (tier-3 scale fallback) — renegotiate before exceeding.
- `iogpu.wired_limit_mb` changes: user-run sudo, never set within ~2.5GB of total RAM, always
  reverted after the run, always recorded in state.json.
- Research-sweep-before-build (2025–26 cascade/routing/calibration literature; mlx-lm changelog;
  MLX memory advances) at the start of Phases A and B, logged like milestone 1.

## 13. Sources for the 2026-06-12 verification pass

- mlx-lm-lora: [GitHub](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) · [PyPI](https://pypi.org/project/mlx-lm-lora/) — 12 training algorithms incl. DPO/CPO/ORPO/GRPO; v2.1.0.
- Wired limit: [override notes](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap) · [Peddals guide](https://blog.peddals.com/en/fine-tune-vram-size-of-mac-for-llm/) · [mlx-lm OOM issue #1015](https://github.com/ml-explore/mlx-lm/issues/1015) (cache fragmentation note).
- Banking77: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) · fine-tuned ~93.7%/93.9% and the zero-shot gap + RAP result: [ASRJETS RAP study](https://asrjetsjournal.org/American_Scientific_Journal/article/view/12048) · [Loukas et al. 2023](https://arxiv.org/pdf/2308.14634).

---

## 14. Addendum (2026-06-12, post-finalization deep research sweep) — upgrades ADOPTED into the spec

Run the same day as finalization, while distiller leg 7 trained (rc's request: verify the whole
plan is cutting-edge, not traditional). Four search batches; sources in §14.6. Verdict first:
the finalized architecture (cascade + calibrated deferral + flywheel + experiment-driven decide)
matches 2025–26 research practice — production frontier systems are themselves cascades/routers
now, and the cascade-routing literature is an active 2026 survey-grade field. The sweep found
five places to go from "current" to "ahead," adopted below as spec changes.

### 14.1 Cascade (Recipe 1) — three upgrades

1. **Calibration method pinned (replaces "calibrate on validation" hand-wave):** tier-2
   confidence = token-margin aggregation + **isotonic regression**, the exact recipe of
   cost-optimal cascade work (UCCI, 2026). Tier 1's LR probabilities get the same calibration
   check. A Self-REF-style confidence-token fine-tune is noted as roadmap, not v2.
2. **Conformal risk control is the flagship's research centerpiece:** `cascade_compose.py`
   selects deferral thresholds with **distribution-free, finite-sample guarantees** ("local-tier
   error ≤ ε at confidence 1−δ" on the routed subset), alongside the descriptive Pareto sweep.
   This upgrades the recipe's product knob from "a threshold you eyeball on validation" to "a
   certified operating point" — the 2025–26 selective-prediction literature does exactly this
   for LLM cascades, and nobody has shipped it as a laptop-scale teaching recipe.
3. **Banking77 label noise is quantified and the bars account for it:** ~14% of train
   utterances are flagged as potential label errors (ACL 2022); filtering them lifted F1 ~4.5
   points in the original study. Adopted: (a) the ≥0.936 primary bar STANDS — the anchor was
   measured under the same noise; (b) the 0.95 stretch is explicitly flagged as possibly above
   the noisy-gold ceiling; (c) the eval adds a **noise-aware secondary read** (confident-learning
   filter over test; report raw AND filtered accuracy) — the §6 "noisy-gold ceiling" lesson,
   productized. (d) **Tier-1 embedding upgrade sub-experiment:** Qwen3-Embedding-0.6B (the 2026
   small-classification leader) vs the static default — static stays the latency default; the
   experiment prices what accuracy-first buys at 0.6B.

### 14.2 Recipe 3 — the post-SFT roadmap becomes RLVR

1. **Round 2 = GRPO/RLVR with the mechanical grounding gate as the verifiable reward.** The
   gate (atomic verbatim grounding + length budget) is a textbook programmatic verifier;
   SFT → RL-against-verifier is the standard 2025–26 two-stage pipeline, and `mlx-lm-lora`
   ships GRPO locally. Reward = gate-pass + ratio-budget terms. This supersedes
   "ORPO on gate failures" as the headline next step (ORPO stays as the cheaper alternative).
   RL rollout memory at 4B/16GB gets a feasibility check in the §8 spike.
2. **DWQ export:** the fused student gets `mlx_lm.dwq` treatment (already installed,
   currently unexploited by tunelab) — distilled weight quantization tunes quantization
   scales/biases against the unquantized model as teacher; "compression accuracy" extends to
   the weights themselves. `mlx_lm.dynamic_quant` (sensitivity-ranked mixed precision) noted
   for the same path.
3. **On-policy distillation context:** OPD is now a standard post-training primitive in
   frontier recipes (GKD lineage). Token-level teacher logprobs aren't available session-native,
   so tunelab's practical on-policy step IS the RLVR round above — the concept doc says so.
4. **Curriculum A/B (cheap, pre-registered):** short→long record ordering for the distiller —
   the published difficulty signals literally include compression ratio, and ordering softens
   early-iteration memory spikes on exactly this dataset shape.

### 14.3 Flywheel — adopt the MAPE control-loop formalism

`flywheel.py`'s trigger logic is specified as a **Monitor–Analyze–Plan–Execute loop** (the 2025
adaptive-data-flywheel formulation; same architecture as NVIDIA's NeMo data-flywheel
microservices, the current production standard — which independently validates §4.5). The NeMo
case-study scale ("~685 curated points moved a small model materially") matches tunelab's
dataset sizes — worth quoting in the recipe.

### 14.4 CPT (Recipe 4 design inputs)

LR **rewarming + re-decaying** with **general-data replay** (the continual-pretraining
canon); **LoRA-CPT** for small corpora (forgetting control by construction);
**EntiGraph-style synthetic CPT** (ICLR 2025) as the modern answer when the domain corpus is
under the ~10M-token gate — synthetic corpus amplification slots into tune-data Path C/D.
These four go into `continuous-pretraining.md` and the Recipe 4 skeleton.

### 14.5 Concepts — currency requirements

- `why-cascades-work.md`: routing-vs-cascading taxonomy per the 2026 survey; EAGLE-3/3.1 as the
  speculative-decoding exemplar (token-level + multi-layer feature fusion; production-default).
- `calibration-and-selective-prediction.md`: isotonic regression + conformal risk control as
  the two practical recipes (matching what cascade_compose.py actually does).
- `curriculum-and-progressive-training.md`: difficulty signals with evidence (compression
  ratio, MTLD, readability) + the "difficulty is not enough — sample utility matters" caveat.
- `sft-vs-preference-tuning.md`: RLVR/GRPO as the 2025–26 default for *verifiable* tasks;
  DPO/ORPO for preference-shaped tasks without programmatic verifiers; the decision rule is
  "do you have a verifier?" first, not "SFT vs RLHF."

### 14.6 Sources (research sweep, all fetched 2026-06-12)

Cascades/routing: [UCCI](https://arxiv.org/html/2605.18796) · [routing/cascading survey](https://arxiv.org/pdf/2603.04445) · [confidence tuning for cascades](https://openreview.net/pdf?id=qYI4fw3g4v) · [Self-REF confidence tokens](https://arxiv.org/pdf/2410.13284) · [agreement-based cascading](https://arxiv.org/pdf/2407.02348).
Conformal/risk control: [Conformal Arbitrage](https://arxiv.org/pdf/2506.00911) · [conformal selective prediction with general risk control](https://arxiv.org/html/2603.24704) · [bootstrapped conformal risk control for LLMs](https://arxiv.org/pdf/2509.23007) · [COIN](https://arxiv.org/pdf/2506.20178).
Distillation/RL: [on-policy distillation collection](https://github.com/chrisliu298/awesome-on-policy-distillation) · [GKD](https://arxiv.org/abs/2306.13649) · [rethinking OPD](https://arxiv.org/html/2604.13016v1) · [RLVR overview](https://www.emergentmind.com/topics/reinforcement-learning-with-verified-reward-rlvr) · [RL for large models survey](https://arxiv.org/pdf/2508.08189).
Banking77 noise: [Label Errors in BANKING77 (ACL 2022)](https://aclanthology.org/2022.insights-1.19/) · [cleanlab confident learning](https://docs.cleanlab.ai/stable/tutorials/clean_learning/text.html).
PEFT: [LoRA variant landscape](https://arxiv.org/pdf/2502.16894) · [HF PEFT LoRA guide (rsLoRA/PiSSA/LoftQ)](https://huggingface.co/docs/peft/developer_guides/lora) · [LoRA-as-knowledge-memory empirical analysis](https://arxiv.org/pdf/2603.01097).
Embeddings: [2026 open-source embedding guide](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models) · [2026 comparison](https://milvus.io/blog/choose-embedding-model-rag-2026.md).
Flywheels: [MAPE adaptive data flywheel](https://arxiv.org/pdf/2510.27051) · [NeMo flywheel (MLRun)](https://www.mlrun.org/blog/mlrun-nvidia-nemo-building-observable-ai-data-flywheels-in-production/) · [NVIDIA flywheel case study](https://www.zenml.io/llmops-database/data-flywheels-for-cost-effective-ai-agent-optimization).
CPT: [Synthetic continued pretraining / EntiGraph (ICLR 2025)](https://arxiv.org/pdf/2409.07431) · [how to (re)warm your model](https://www.semanticscholar.org/paper/193955704f66923ac20a664bd184ed4663b2bdf9) · [continual learning survey](https://github.com/Wang-ML-Lab/llm-continual-learning-survey).
Curriculum: [curriculum pretraining](https://arxiv.org/abs/2506.11300) · [Difficulty Is Not Enough (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/40400/44361).
Speculative decoding: [EAGLE-3 in practice](https://huggingface.co/blog/lujangusface/tw-eagle3-gpu) · [EAGLE 3.1](https://www.marktechpost.com/2026/05/27/meet-eagle-3-1-the-speculative-decoding-algorithm-that-fixes-attention-drift-in-llm-inference/).
MLX: [LEARNED_QUANTS.md (DWQ/AWQ/dynamic)](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md) · [mlx-lm](https://github.com/ml-explore/mlx-lm) · [Apple M5 MLX benchmarks](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).

---

## 15. The capstone (added 2026-06-12): the self-improving system — Recipe 5 + `tune-loop` (Phase E)

rc's framing, adopted: **the flywheel is not a feature of one recipe — it IS the AI system.**
The system boundary covers data preparation, training, evaluation, serving, and feedback.
What sits inside may be a single model, a cascade, an agent, or a deterministic multi-model
workflow — the system doesn't care; it experiments with architectures × fine-tuning
methodologies and adopts whatever measurably improves accuracy from the current state, on the
user's own data. This is the champion/challenger pattern from classical MLOps, generalized to
compound-AI architecture search, driven by the §14.3 MAPE loop.

### 15.1 The pieces

1. **System descriptor** (small versioned YAML — makes architectures enumerable, comparable,
   reproducible): components (model + training method + calibration), routing/composition
   logic, thresholds. Examples:
   `single: FT-SLM(qwen3-4b, sft)` ·
   `cascade: LR(static-emb) →[p<.70] FT-SLM →[margin<.20] frontier(knn-fewshot)` ·
   `workflow: injection-gate → router → {cheap, frontier}`.
2. **Runtime + prediction log** (§4.5 schema): every prediction from every component, plus
   feedback — explicit (user corrections), automated (programmatic verifiers like the grounding
   gate; downstream success signals), and structural (escalation-tier outputs are free labels
   for the tiers below).
3. **Curation:** dedupe → conflict resolution → confident-learning noise filter → time-stamped
   snapshots. Feedback is never trained on raw.
4. **Experiment engine** (Analyze/Plan): on trigger (≥N curated labels · drift · schedule),
   assemble snapshot D_t, generate a challenger set from the **portfolio** (the ladder,
   generalized: architecture families first, then per-family method refinements — LoRA vs DoRA,
   rank, RLVR round, embedding upgrade — one factor per round), score cheap proxies first
   (offline counterfactual composition, §4.3), train only survivors within a declared
   compute/$ budget.
5. **Promotion** (Execute): champion vs challengers on a *fresh, never-used* eval slice against
   a pre-registered promotion bar (≥X accuracy at ≤Y cost at iso-latency; certified operating
   point re-derived per §14.1). Promote or retain; either way the round is appended to
   EXPERIMENT-LOG.md. Repeated no-promotions lengthen the trigger interval — the system
   self-paces toward stability instead of churning.

### 15.2 The three failure modes the design must own (else this is AutoML slop)

1. **Feedback bias:** feedback over-samples escalated/uncertain cases — the logged distribution
   is not the serving distribution. Owned by: a small always-on **uniform random audit slice**
   (x% of traffic gets gold feedback regardless of confidence), stratified-by-tier reporting,
   importance weighting where feasible.
2. **Eval burn:** continuous improvement eats test sets. Owned by: time-based **rolling frozen
   slices, each consumed exactly once** (the one-look rule, systematized); challenger selection
   happens on validation windows; adjudication only on the newest untouched slice.
3. **Search explosion:** architectures × methods is unbounded. Owned by: staged search (family
   → refinement), per-round declared budget, and a minimum-improvement bar so rounds converge.

### 15.3 Form factor and dogfood

- **Recipe 5 (the capstone recipe)** + a **fifth skill, `tune-loop`**, that drives MAPE rounds
  using the existing four skills as its tools — decide is Plan, data is curation, train is
  Execute, eval is Analyze/adjudicate. The four skills are one revolution; tune-loop is the
  crank. No serving infrastructure (no gateway/deploy product): artifacts stay local scripts,
  logs, and descriptors.
- **Dogfood, two stages:** (a) build the machinery on a **Banking77 replay stream** (Phase B
  assets; gold labels revealed over simulated time, labeled as simulated) — cheap, controlled,
  reproducible; (b) showcase on **the router's real traffic** (Recipe 2 — rc's own logs accrue
  daily; automated feedback = blinded judge over cheap-tier outputs), which makes the capstone
  a story about a system that already exists in this repo getting measurably better on live data.
- **Definition of done (Phase E):** ≥3 autonomous rounds on the replay stream with receipts,
  including at least one challenger promotion *earned* on the pre-registered bar AND at least
  one promotion correctly *rejected*; zero eval-slice reuse; then ≥1 round on real router
  traffic.

### 15.4 A deliberate v1 reversal, logged

PLAN.md §7 named autonomous config-mutation hill-climbing a non-goal ("inverts
teach-while-doing"). The capstone deliberately reverses that — with the disciplines that make
it teaching-grade rather than slop: pre-registered promotion bars, one-look slices, declared
budgets, append-only logs, and human checkpoints at spend and promotion (delegable, like the
standing "continue, don't wait" arrangement). The EXPERIMENT-LOG a learner reads afterwards is
still the product.

**Operating rule (rc, 2026-06-12): when in doubt, escalate.** The system — and its builders —
run in supervised mode: verify mechanically first, then present findings with a recommendation
and let the human pick; never silently execute judgment calls (dataset-composition changes,
spend, anything that alters a pre-registration). Complete autonomy is the destination, not the
mode.

### 15.5 Sequencing (maps onto existing phases — nothing is thrown away)

- **Phase B already builds one hand-cranked revolution:** tier components, offline composition,
  prediction log, one demonstrated retrain cycle (§4.5).
- **Phase C's tune-decide rework IS the experiment engine's Plan stage** (§6).
- **Phase E (new, after D):** system descriptor + `tune-loop` + champion/challenger promotion +
  the §15.2 bias/burn machinery + the two-stage dogfood. Recipe 5 written from those receipts.

---

## 16. Development strategy (added 2026-06-12): tiny-first, system-first

1. **Tiny-first.** All system machinery — cascade scripts, flywheel, conformal composition,
   `tune-loop` — is developed and debugged with **Qwen3-0.6B** (rc's pick; mlx-community 4-bit,
   0.34GB, already live-verified in this repo's milestone-2 smoke tests, same family/tokenizer
   as the Qwen3-4B target so configs transfer 1:1) standing in for the fine-tuned tier. Full
   pipeline iterations take minutes. Big models run **once, at the end**, for a recipe's
   headline numbers. The receipt that motivated this: the leg-7/8 NaN was a data bug that
   burned 2×33 minutes at 4B and would have surfaced in ~3 minutes at 0.6B.
   (Gemma 3 270M noted as the browser/on-device demo alternative for a later cascade-deployment
   showcase.)
2. **System-first critical path.** Multi-hour headline training runs launch detached with
   watchers and never block development; attention goes to Phase B/E machinery. Recipe 3's
   leg 9 is the template: data rebuilt and verified in minutes, run launched, work continues.
3. **Diagnose offline before re-running.** No blind retry legs: a failed run gets a mechanical
   diagnosis (tokenizer scans, log forensics) before any relaunch — a 2-minute scan beats a
   33-minute leg.
