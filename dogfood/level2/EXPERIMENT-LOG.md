# taskloop-triage-json — EXPERIMENT-LOG

## 2026-06-10 — tune-decide: level decision
- Decision: **Level 2** (LoRA SFT) — output is a structured JSON object (category + urgency +
  free-text summary), not a bare label: the summary field makes this generation, not
  classification, so Level 1 is genuinely ruled out (the category *alone* would be Level 1 — per
  the multi-output rule, that decomposition was considered; here the JSON object is the product).
- Interview: input = customer support ticket (free text, fictional Taskloop SaaS); output =
  strict JSON {category ∈ 8 labels, urgency ∈ 4 levels, summary ≤ 15 words}; data = none (Path C
  synthetic, user decision 2026-06-10); motive = dogfood the full Level-2 pipeline with zero API
  spend; hardware = M1 Pro 16GB.
- Pre-registered (2026-06-10, user rc, before any data existed): **bar = format-validity ≥ 98% on
  test ∧ category accuracy ≥ 0.85 vs teacher labels ∧ blinded session-native judge: tuned ≥ 60%
  equivalent-or-better vs teacher outputs.** Metrics = the generative family card (format
  validity, field accuracy, judge win-rate) per PLAN §8. Training launch pre-approved by rc at
  the same checkpoint.
- Model plan: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (verified id; non-thinking template —
  output starts at the schema's first byte).
- Teacher tier: **session-native** (no API key) — model unpinned: Claude Code session
  (Claude Fable 5), 2026-06-10. Recorded per the tier-disclosure convention.

## 2026-06-10 — tune-data: Path C spot-check + prompt freeze
- Run (config): 25 synthetic inputs authored across the diversity axes (terse/verbose/angry/
  typos/multi-issue/vague/wrong-product/security-report), labeled per the draft teacher prompt;
  read with user rc.
- Decision: **prompts frozen 2026-06-10** (user approval at the prompt-freeze checkpoint):
  teacher prompt = the strict-JSON triage rubric (recorded verbatim in data/DATACARD.md);
  input-generation prompt = Taskloop product facts + per-batch diversity slices (personas ×
  lengths × edge quotas × rare-category quotas).
- Result: 25/25 valid JSON; category spread across all 8 labels; judgments accepted unchanged.
- Scale plan: 16 session-native batches × 50 → ~800 + 25 spot-check, dedupe toward ~600,
  split 80/10/10 stratified by category.

## 2026-06-10 — tune-data: dataset built
- Run (config): 800 generated + 25 spot-check; dedupe --threshold 0.80 → **825→825, zero dups**
  (the per-batch diversity slicing prevented synthetic collapse outright — the 130% overgeneration
  budget went unspent); split --seed 42 --label-key label → 659/83/83; validate: OK.
- Lesson: diversity-sliced generation beats overgenerate-and-dedupe — engineer spread up front.

## 2026-06-10/11 — tune-train: run 20260610-qwen3-4b-2507-triage (3 legs, early-stopped)
- Leg 1 (lr 1e-4, batch 4, full memory): iters 1→~120; val 4.733→**0.702 @68**. Killed at ~120 —
  the run wired 11 GB of Metal memory and froze the user's machine (Peak mem 12.4/16 GB);
  ~52 post-checkpoint iters lost, bounded by --save-every per design.
- Interim eval @ckpt-68 on test.jsonl (n=83): validity 100%, category 71.1%, urgency 57.8%
  vs base 100/65.1/32.5 → format learned, policy under-trained; keep training.
  **test.jsonl spent by this look** → fresh test generated (90 records, 2 never-used slices,
  0 text collisions with training data).
- Leg 2 (resume @ lr 1e-4, batch 2 + grad-accum 2, --grad-checkpoint): **DISCARDED** — cold-Adam
  shock: train 0.585→0.737, val 0.718→0.848 over 68 iters.
- Leg 3 (same resume point, **lr halved to 5e-5**): val 0.718→0.705→0.563→0.536→**0.509 (global
  ~340)**→0.513→0.666 → early-stopped on two non-improving evals; best checkpoint = global ~340,
  staged in adapters-best/.
- Predicted-vs-actual: predicted the val bottom would arrive well before the planned 823 iters
  on narrow synthetic data → actual bottom at ~41% of plan. Predicted "brief" resume loss bump →
  actually destabilizing at full LR; halving LR fixed it.
- Lesson (propagated to tune-train SKILL.md): weights-only resume at the original LR destabilizes
  long remainders — halve LR on resume. Also: 4B training at batch 4 without --grad-checkpoint
  is not desktop-compatible on 16 GB.

## 2026-06-11 — tune-eval: final scoreboard (fresh test, n=90) — ALL BARS PASSED
- Bar (pre-registered 2026-06-10): validity ≥98% ∧ category ≥85% ∧ judge ≥60% equiv-or-better.
- Result: **validity 100%** ✓ · **category 92.2%** ✓ · **judge 68.9%** ✓ (5 tuned wins / 57 ties /
  28 teacher wins, blinded pairs with flips committed to disk pre-judging, 5 subagent batches).
- Base control on the same fresh test: category 70.0%, urgency 44.4% → training lift +22.2 / +40.0
  points. Tuned urgency 84.4%.
- Notable: the student beat its teacher on 5 pairs — judges flagged the teacher's own >15-word
  summaries and urgency misratings. Distillation onto a frozen rubric can out-discipline the
  rubric's author.
- Judge tier: session-native (model unpinned — Claude Code session, Claude Fable 5, 2026-06-11);
  verdicts in verdicts_unblinded.jsonl.
- Remaining error mass: billing/cancellation→how_to tail (4) + the bug↔data_integration teacher-
  drift boundary the DATACARD flagged.
- Verdict: **bars met — dogfood run complete.** (Fictional product: no production routing/drift
  plan; the pipeline that produced these numbers is the deliverable.)
