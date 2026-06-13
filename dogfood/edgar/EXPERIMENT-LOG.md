# EDGAR CPT (Recipe 4) — EXPERIMENT-LOG

## 2026-06-13 — research-mode CPT on SEC EDGAR filings (the fluency lesson, live)
- Corpus: 3 real 10-K filings (AAPL/MSFT/NVDA), fetched from EDGAR (public, no auth), HTML
  stripped → 231K tokens → 114 chunks (1-2k tokens) → 103 train / 11 valid. This is a
  RESEARCH-MODE corpus (well under the ~10M-token gate) — the goal is to watch domain
  perplexity fall (fluency), NOT to claim factual QA gains.
- Model: `mlx-community/Qwen3-0.6B-Base-4bit` (base, not instruct — CPT continues next-token
  pretraining). LoRA, lr 1e-5, 8 layers, no --mask-prompt (text data), 200 iters, batch 2.
- **Domain perplexity (held-out EDGAR chunks):**
  | iter | val loss | perplexity |
  |---|---|---|
  | 1 (baseline) | 1.988 | 7.30 |
  | 50 | 1.879 | 6.55 |
  | 100 | 1.859 | 6.42 |
  | 150 | 1.849 | 6.35 |
  | 200 | 1.846 | 6.33 |
  **Baseline 7.30 → CPT 6.33 = 13.3% perplexity reduction** — the base model got measurably more
  fluent in filings-speak. Smooth monotone decline, converging. Peak mem 5.47GB.
- The fluency-vs-facts lesson made visible: perplexity drops (the model speaks the domain idiom
  better) — but this run makes NO QA-accuracy claim, exactly as research-mode CPT should. The
  production pattern stays CPT-for-fluency + RAG-for-facts (concepts/cpt-vs-rag.md).
- Recipe 4 goes from skeleton to a live perplexity demonstration; the full pipeline (SFT restore,
  forgetting check, downstream-lift, maintained-loop) at scale remains Phase-3 work.
