# SFT vs preference tuning (DPO/ORPO) vs RLVR

Once you've decided to fine-tune, *how* you fine-tune depends on what kind of
signal you have. The 2025–26 decision rule isn't "SFT vs RLHF" — it's a
question tree.

## The first question: do you have a verifier?

A **verifier** is a program that can mechanically check whether an output is
correct — a unit test, a JSON-schema validator, a symbolic math checker, the
distiller's atomic-grounding gate. If you have one, you have the strongest and
cheapest training signal there is.

- **Yes, a verifier exists → RLVR** (reinforcement learning with verifiable
  rewards, usually **GRPO**). The model generates candidates, the verifier
  scores them, and the policy is pushed toward the ones that pass. No human
  preferences, no reward model to train — the reward is just "did it pass?"
  This is how frontier reasoning/code/math models are post-trained, and it runs
  locally on Apple Silicon via `mlx-lm-lora`. tunelab's distiller is a natural
  fit: SFT first, then a GRPO round with the grounding gate as the reward.

## The second question: preferences or demonstrations?

If there's no programmatic verifier:

- **You have input→ideal-output pairs → SFT** (supervised fine-tuning). The
  workhorse. Teach behavior, style, one schema, one voice by imitation. Most
  tunelab tasks are here and *stop* here — SFT is usually enough, and the
  honest move is not to reach for RL when imitation suffices.
- **You have "A is better than B" pairs → preference tuning.** When quality is
  a matter of judgment (tone, helpfulness, which of two replies is better) and
  can't be reduced to a verifier:
  - **DPO** (Direct Preference Optimization) — trains directly on
    chosen/rejected pairs against a reference model; no separate reward model,
    no RL loop. The practical default for preference data.
  - **ORPO** — folds preference and SFT into one stage with no reference model
    (lighter memory, one pass). Good when you're training from scratch on
    preference data rather than refining an SFT checkpoint.
  - **RLHF/PPO** — the classic (train a reward model, then RL). More moving
    parts; DPO/ORPO get most of the benefit with far less machinery, which is
    why they dominate in 2025–26.

## What each optimizes

- **SFT** maximizes likelihood of the reference outputs — it can only be as
  good as your demonstrations, and it can't learn "avoid this."
- **DPO/ORPO** raise the gap between chosen and rejected — they can learn what
  *not* to do, which SFT can't, but they need preference pairs.
- **RLVR/GRPO** maximizes a real correctness signal — the strongest, but only
  available when correctness is checkable.

## tunelab's stance

Default to SFT; reach further only with a reason. If you have a verifier (and
many "extraction/compression/format" tasks secretly do), an RLVR round on top
of SFT is the highest-leverage upgrade. Preference tuning is for genuinely
judgment-shaped quality. Verify the MLX ecosystem support live before relying
on it — `mlx-lm-lora` ships DPO/ORPO/GRPO, but versions move fast.
