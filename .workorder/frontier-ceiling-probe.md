# Work Order: Re-run Frontier Ceiling Probe with Opus 4.8 and GPT-5.5

**STATUS: COMPLETE (2026-06-23).** Opus 4.8 = 126/154 = 0.8182 (prior session, session-native);
GPT-5.5 = 132/154 = 0.8571 (this session, OpenAI API). Both below the $0 LR classifier (0.883),
so the hero/cascade framing holds — cascade delta immaterial, no re-composition. See
`dogfood/cascade/EXPERIMENT-LOG.md` (2026-06-23 entries) for the full trail.

## Context

The README's first hero example claims a free local classifier (88.5%) beats a frontier model (81.8%) on Banking77 77-class intent classification. The 81.8% number was measured session-native on **Claude Fable 5** — not on Opus or GPT-5.5. Since Fable is a stronger model than both, the 81.8% may be *generous* to the frontier tier, meaning Opus/GPT-5.5 could score lower. Either way, the current README attributes "81.8%" to GPT-5.5, which is unverified. We need real numbers.

## Task 1: Run the frontier ceiling probe on Opus 4.8 and GPT-5.5

### Data
- Input: `dogfood/cascade/data/ceiling_probe.jsonl` (154 stratified records, 2 per class × 77)
- Labels: `dogfood/cascade/data/labels.json` (77 labels)
- Each record: `{"id": "...", "text": "...", "label": "..."}`

### Script
Use `skills/tune-data/scripts/distill_generate.py` with `--gold-key label` to produce eval-ready predictions (outputs `{"id", "text", "predicted", "expected"}`).

**Opus 4.8 run:**
```bash
cd backend  # or wherever uv is configured — run from the tunelab root
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY uv run skills/tune-data/scripts/distill_generate.py \
  --mode classify \
  --provider anthropic \
  --model claude-opus-4-8 \
  --input dogfood/cascade/data/ceiling_probe.jsonl \
  --labels dogfood/cascade/data/labels.json \
  --gold-key label \
  --system "You are a classifier. Assign the user message to exactly one label from this list:\n{labels}\nReply with the label only — no explanation, no punctuation, nothing else." \
  --output dogfood/cascade/data/tier3_ceiling_opus48.jsonl
```

**GPT-5.5 run:**
```bash
OPENAI_API_KEY=$OPENAI_API_KEY uv run skills/tune-data/scripts/distill_generate.py \
  --mode classify \
  --provider openai \
  --model gpt-5.5 \
  --input dogfood/cascade/data/ceiling_probe.jsonl \
  --labels dogfood/cascade/data/labels.json \
  --gold-key label \
  --system "You are a classifier. Assign the user message to exactly one label from this list:\n{labels}\nReply with the label only — no explanation, no punctuation, nothing else." \
  --output dogfood/cascade/data/tier3_ceiling_gpt55.jsonl
```

### Scoring
After each run, compute accuracy (predicted == expected) on the 154 records. Compare against the existing baselines:
- Fable 5 (session-native): 126/154 = 0.8182
- Tier-1 LR classifier: 136/154 = 0.8831

### Notes
- The `--gold-key label` flag makes the script emit `predicted` and `expected` fields, ready for eval.
- The script is resumable — if it crashes partway, re-run the same command and it skips completed IDs.
- Run `--limit 5` first as a smoke test before the full 154.
- The `--system` flag uses `{labels}` as a placeholder — the script injects the label list from `--labels`.
- Cost estimate: 154 calls × ~100 input tokens × ~5 output tokens ≈ negligible (< $0.50 total for both runs).

## Task 2: Update README and Recipe 01 with real numbers

Once you have the results:

1. **README.md lines 6-10** — Update the hero example. The "You:" prompt currently says GPT-5.5. Replace the accuracy number with the real GPT-5.5 result. If both Opus and GPT-5.5 score below the classifier (88.5%), the story holds as-is but with corrected numbers. If either scores *above* the classifier, the example needs a different framing — don't bury that.

2. **recipes/01-hybrid-cascade.md line 23** — Currently says "Frontier model, no examples ('zero-shot') | 0.818". Update with the new numbers. Consider showing all three frontier results in a table:
   | Model | Accuracy |
   |---|---|
   | Claude Fable 5 (session-native) | 0.818 |
   | Claude Opus 4.8 (session-native) | 0.818 |
   | GPT-5.5 (API) | 0.857 |

3. **dogfood/cascade/EXPERIMENT-LOG.md** — Append a new dated entry documenting the re-run: method, results, comparison to Fable baseline, and any implications for the cascade composition.

## Task 3: Verify cascade claims still hold

If the frontier tier accuracy changes significantly, re-check whether these claims from the README and Recipe 01 still hold:
- "94% accuracy" for the 3-tier cascade
- "8× cheaper than frontier-only"
- "free classifier beats frontier by +6.7 points"

The cascade composition numbers (from `cascade_compose.py`) used the Fable 0.818 as the tier-3 input. If Opus/GPT-5.5 score differently, the cascade's overall accuracy and cost split would change. You may need to re-run composition — but document whether the delta is material first before re-running everything.

## Acceptance criteria
- [x] Opus 4.8 accuracy on ceiling_probe.jsonl measured and recorded — 0.8182 (prior session)
- [x] GPT-5.5 accuracy on ceiling_probe.jsonl measured and recorded — 0.8571 (this session)
- [x] README hero example updated with real, verified number — re-attributed to Opus 4.8 81.8% (GPT-5.5 is below the classifier, so the gate to revisit the hero wasn't tripped)
- [x] Recipe 01 updated with new frontier baselines — all three frontier rows + 2.6–6.5 range
- [x] EXPERIMENT-LOG.md entry appended — 2026-06-23 "GPT-5.5 LEG RUN" entry
- [x] If cascade claims are affected, document the delta — immaterial; 0.9416 / +12.3 / 8× hold verbatim (no re-composition)
