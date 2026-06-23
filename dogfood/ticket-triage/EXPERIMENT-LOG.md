# ticket-triage (Recipe 4) — EXPERIMENT-LOG

A 5-bucket support-ticket triage (account_access, billing, bug_report,
feature_request, shipping) replacing an expensive frontier classifier. Motive =
cost; hardware = Apple M1 Pro, 16 GB. The real decision trail, dead ends kept in.

## 2026-06-22 — tune-decide: Level 1 attempted, data leakage found
- **Decision:** Level 1 (local embeddings + logistic-regression classifier) —
  fixed-label classification, never fine-tune for this shape. User also asked to
  see Level 2 (LoRA) as a comparison.
- **Pre-registered bar:** frontier parity ("cheaper without losing accuracy").
- **Data:** `tickets.jsonl` = 150 rows, balanced 30/class — BUT only **25 unique
  texts** (5/class), each repeated ~6× on average (multiplicities 4–10×). No
  conflicting labels among the duplicates.
- **Result:**
  - Single 20% holdout: acc 1.000 / macro-F1 1.000 — FALSE (train/test leakage
    from duplicates).
  - 5-fold CV on the raw 150: 1.000 ± 0.000 — still leaked (dups span folds).
  - **Deduped (25 unique), 5-fold CV: acc 0.560 ± 0.233, macro-F1 0.480 ± 0.268 —
    the honest number.** Folds swing 0.40–1.00 → estimate unstable on 5/class.
- **Reading:** the 100% was leakage, not skill. The level/model choice is fine;
  the blocker is **data quantity** (guardrail wants 20+ unique/class).
- **Lesson:** always dedup before trusting a held-out score; a perfect score is a
  leakage alarm.

## 2026-06-22 — tune-data: generation recipe frozen (Path C, synthetic)
- **Teacher:** session-native Opus 4.8 (unpinned), no API key. Path C (synthetic
  from scratch) — no usable production log existed yet.
- **Spot-check:** 25 samples, 2 rounds. Added 4 realism axes (real-world mess,
  multi-intent, identifiers/context, wider length+emotion) — all user-selected.
- **FROZEN recipe:** buckets {account_access, billing, bug_report,
  feature_request, shipping}; vary length/persona/mess/identifiers; ~20–25%
  multi-intent labeled by **primary actionable intent** (tie-break rule ratified
  on 5 borderline cases); avoid near-dups.
- **Test-set decision:** human-verified GOLD set from REAL tickets, NOT synthetic
  — avoids "Opus grading its own homework" (circular eval). Plan a 4-way compare:
  classifier vs LoRA vs Opus 4.8 vs GPT-5.5 (OpenAI key available).

## 2026-06-22 — tune-data done + Option B (classifier) first gold result
- Generated 628 synthetic (15 Opus subagents) → dedup 625 → stratified split
  train 530 / valid 95 (seed 42). Gold test = 25 real tickets, human-verified.
- Clean-split AUDIT: 0 exact + 0 near leakage (max cross Jaccard 0.46), 0 label
  conflicts, 0 real texts in train/valid, balanced. `validate_dataset.py`: OK.
- **Option B** (Level 1: logistic regression on static embeddings), train 530
  synthetic → eval 25 real gold: **accuracy 0.920, macro-F1 0.920.** Perfect
  billing/feature_request/shipping; 2 misses both on the
  account_access↔bug_report boundary.
- Reading: the free local classifier already ~matches frontier quality on real
  tickets; errors confined to an inherently fuzzy boundary. Strong ship
  candidate. Next: LoRA (Option C) + 4-way eval.

## 2026-06-22 — tune-train: LoRA (Option C, research-mode comparison)
- Decision: Qwen2.5-1.5B-Instruct-4bit (QLoRA, non-thinking → clean label
  output), classification SFT.
- Run: `mlx_lm.lora --iters 662 --batch-size 4 --lr 1e-4 --num-layers 16
  --mask-prompt --seed 42` (5 epochs over 530).
- Result: val loss ~0.038 by iter 110 (train ~0.015) → task solved fast;
  early-stopped. Latest adapter = iter 275, plateaued at the floor, no climb.

## 2026-06-22 — tune-eval: 4-way scoreboard on real gold (n=25)
- Bar pre-registered = frontier parity. Classification, exact-match, identical
  blind system prompt for all.
- Result (accuracy / macro-F1):
  - base Qwen2.5-1.5B (control): 0.680 / 0.678 → training added +0.32, clearly took
  - Option B classifier (~$0): 0.920 / 0.920 (2 misses: SSO→bug, dashboard-freezes→account)
  - LoRA (Option C): 1.000 / 1.000 (got BOTH boundary cases)
  - Opus 4.8 (teacher): 1.000 / 1.000
  - GPT-5.5: 1.000 / 1.000
- Caveat (logged at the time): n=25 → 0.92 vs 1.00 = a 2-ticket gap, within noise
  (binomial CIs overlap heavily). **This caveat is the whole reason the eval was
  later scaled** — see the flip below.
- Verdict (interim): SHIP Option B + confidence-route low-prob tickets. LoRA
  proves local frontier-parity is achievable but adds model-hosting overhead for
  a within-noise gain.

## 2026-06-22 — Option B improvement: better embeddings + confidence routing
- Distilled a larger embedding model: static → text-embedding-3-small. Gold
  0.920 → 0.960 (fixed "dashboard freezes after login"); v1 synthetic held-out
  0.943 → 0.991. Cost ~$0.002/1k.
- Remaining gold miss: "SSO stopped working" → confidence 0.55 (near-tie; the
  model self-flags the ambiguity → a routing candidate, not a label to force).

## 2026-06-22 — tune-data v2: large Opus dataset + independent audited eval
- 25 Opus 4.8 subagents: 15 train (~760) + 10 eval (~465, deliberately different
  distribution). Folded in the v1 pool.
- TRAIN 1387 → dedup 1381 (6 exact, 0 near) → stratified 85/15 → **train 1174 /
  valid 207** (balanced).
- EVAL **465** (balanced), cross-deduped vs train+gold (max Jaccard 0.74, p99
  0.42, 0 dropped) → independent generalization test.
- Real gold 25 kept as the cross-model anchor.
- **LABEL AUDIT** (GPT-5.5 2nd pass, all 465): **agreement Opus vs GPT-5.5 =
  0.951 — the LABEL CEILING.** Per-class: feature_request 1.00, shipping 0.99,
  billing 0.97, bug_report 0.96, **account_access 0.84**. Dominant disagreement:
  account_access↔bug_report = 15 of 23 — the irreducible login-vs-malfunction
  boundary.
- Read: ~95% is the achievable ceiling on the synth-eval; account_access is the
  hard class. Densification won't fix irreducible ambiguity → resolve via routing.

## 2026-06-22 — tune-train v2 + full scoreboard (synth-eval n=465 + real-gold n=25)
- LoRA retrained on data2 (1174 train): Qwen2.5-1.5B-Instruct-4bit, iters 880,
  batch 4, lr 1e-4, num-layers 16, mask-prompt, seed 42.
- Label ceiling (Opus vs GPT-5.5) = 0.951. Synth-eval = teacher-mimicry;
  real-gold = anchor.
- **Synth-eval (n=465) acc:** static 0.800; text-embedding-3-small 0.927; LoRA v2
  0.912; GPT-5.5 0.951 (== agreement, circular); Opus 1.000 (definitional, excluded).
- **Real-gold (n=25) acc:** static 0.960; 3-small 0.960; LoRA 0.960; GPT-5.5
  1.000; Opus 1.000.
- **The flip:** at n=25 the LoRA's 1.000 had beaten the classifier's 0.920; at
  n=465 the ranking REVERSED — classifier 0.927 > LoRA 0.912. Small test sets lie.
- Verdict: SHIP the OpenAI-embedding classifier (~$0.002/1k, 0.927 vs 0.951
  ceiling) + routing. **LoRA does NOT beat it at scale and costs more to run;
  fine-tuning not justified.** account_access residual irreducible.

## 2026-06-22 — embedding-scaling + ensemble test
- **text-embedding-3-LARGE classifier** (same train 1174, same eval): synth-eval
  **0.942** (up from 3-small 0.927), real-gold **1.000**, held-out 0.979,
  account_access-F1 0.92. ~$0.013/1k. Now ~0.9pt under the 0.951 ceiling.
- Confirms the **EMBEDDING is the lever** (not model size): a bigger embedding
  (0.942) beats the fine-tuned 1.5B LoRA (0.912), still ~free.
- Disagreement (3-small clf vs LoRA, n=465): agree 90.8%; of 43 disagreements clf
  right 23, LoRA right 16, both wrong 4. Ensemble ORACLE (perfect picker) = 0.961
  — a realistic ensemble can't beat the single 3-large classifier (0.942).
  Ensembling small+LoRA is pointless; the 4/465 both-wrong are irreducible.
- **FINAL RECOMMENDATION:** ship the text-embedding-3-large classifier (0.942
  eval / 1.000 gold, ~$0.013/1k) + confidence routing. LoRA & bigger SLMs
  unnecessary.

## 2026-06-22 — verification pass (recomputed from artifacts for Recipe 4)
Reproduced the headline numbers directly from the saved prediction files so the
recipe cites measured, not remembered, values:
- Scoreboard, independent eval n=465 (acc / macro-F1):
  static **0.800 / 0.798**; 3-small **0.927 / 0.927**; 3-large **0.942 / 0.942**;
  LoRA **0.912 / 0.911**.
- Real gold n=25: static 0.960; 3-small 0.960; 3-large **1.000**; LoRA 0.960.
- Label ceiling (`eval_audited.jsonl`): Opus vs GPT-5.5 = **0.9505 (442/465)**.
  Per-class agreement: account_access 0.839, billing 0.967, bug_report 0.957,
  feature_request 1.000, shipping 0.989. account_access↔bug_report = **15** of the
  23 disagreements (earlier notes said 13; the artifact says 15 — corrected here).
- Generalization gap, recomputed within v2 (static classifier on `valid` via the
  repo's own predict path): **holdout 0.932 → independent eval 0.800 (−13.2)**;
  3-small **holdout 0.966 → eval 0.927 (−3.9)**. The bigger embedding generalizes;
  the static one had leaned on surface phrasing that didn't transfer.
- Leakage hook: `tickets.jsonl` confirmed 150 rows / 25 unique / 5 per class.

Source datasets, models, and run logs live in the originating project
(`tunelab-trial`): `data2/{train,valid,test}.jsonl`, `gold_test.jsonl`,
`eval_synth.jsonl`, `eval_audited.jsonl`, `clf2_*.joblib`, `runs/`.
