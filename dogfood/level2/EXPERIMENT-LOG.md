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
