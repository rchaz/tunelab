# Recipe 4 — Finance filings analyst

**Who this is for:** you work in a domain with its own dense, specialized language — SEC filings, legal contracts, medical notes — and a general model only half-understands the idiom.

**The plain idea.** Some domains speak a language the base model barely knows. You can't fix that by retrieving facts (that's what RAG does) — you have to make the model *fluent* in the domain. **Continued pretraining (CPT)** does that: you keep training a base model on a big pile of raw domain text so it gets comfortable with how the domain actually writes.

The production pattern is **CPT for fluency + RAG for facts**: the model reads the domain natively, and the actual numbers come from retrieved, cited source documents — never from the model's memory (see [fine-tuning vs RAG](../concepts/cpt-vs-rag.md)).

> **Jargon, once:** *perplexity* = how surprised the model is by a piece of text. Lower perplexity = the text feels more natural to the model = more fluent.

## Why this is a system, not a one-time job

A model trained on last quarter's filings drifts as the domain moves. So CPT is a loop:

```
new filings come in → train on just the new text → small refresh pass →
check it didn't break → ship or hold → repeat
```

Each step has a non-obvious requirement (full detail in [continued pretraining](../concepts/continuous-pretraining.md)): only train on genuinely new text; use a low, carefully-shaped learning rate; mix in some general data so the model doesn't *forget* its general skills ("catastrophic forgetting"); and run a small instruction-tuning pass afterward, because raw CPT produces a text-completer, not a helpful assistant.

## The live result (research mode)

A small, honest demonstration — **not** a production-scale run:

- **Corpus:** 3 real 10-K filings (Apple, Microsoft, NVIDIA), fetched free from SEC EDGAR (public, no login), cleaned to ~231,000 tokens.
- **Model:** Qwen3-0.6B base, trained locally with LoRA. Peak memory 5.47GB.
- **Result — domain fluency improved:**

  | Training step | Perplexity on held-out filings |
  |---|---|
  | 0 (baseline) | 7.30 |
  | 50 | 6.55 |
  | 100 | 6.42 |
  | 200 | **6.33** |

  **A 13.3% drop in perplexity** — the base model got measurably more fluent in filings-speak, with a smooth, converging curve.

This is the **fluency-vs-facts lesson made visible**: the model speaks the domain better, but this run makes **no claim about answering questions more accurately** — exactly what a small research-mode run should and shouldn't claim. For real factual answers, you still pair CPT with RAG.

## When CPT is actually the right call

CPT is the heaviest rung on the ladder — only reach for it when all three hold:

1. The knowledge is **stable** (not facts that change weekly — those are RAG's job).
2. You have **millions of words** of raw domain text (or you generate synthetic domain text to amplify a smaller corpus).
3. Your motive is **speed, cost, or offline use** — no retrieval round-trip, shorter prompts, runs on-device.

Otherwise: RAG for facts, or a lighter rung on the ladder. (Research mode relaxes the size requirement when the goal is *learning*, with expectations reset to fluency, not accuracy — which is exactly what the run above did.)

## What's done and what's next

Done: a live perplexity demonstration on real filings. Still ahead (larger scale): the instruction-tuning refresh pass, a forgetting check, a downstream task showing the CPT model beats the un-CPT'd one on a real job, and wiring the refresh loop into `tune-loop`.

Full run log: [`dogfood/edgar/EXPERIMENT-LOG.md`](../dogfood/edgar/EXPERIMENT-LOG.md).
