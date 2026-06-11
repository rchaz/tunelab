# Continued pretraining (CPT) vs RAG

Both get domain knowledge into a system. They do it in incompatible ways, and
picking the wrong one is the most expensive mistake on tunelab's ladder.

**CPT** continues a base model's original training objective — predict the
next token — on your domain corpus (SEC filings, your codebase, clinical
notes). The model absorbs the domain's *vocabulary, phrasing, and structure*.
What it does not reliably absorb is *facts*. Weights are a lossy, blurry
store: a model CPT-ed on your API docs will write fluently in your API's
idiom and still hallucinate parameter names. Training teaches behavior and
style, not lookup.

**RAG** (retrieval-augmented generation) keeps facts outside the model: store
the documents, retrieve the relevant passages per query, put them in the
prompt. Facts stay exact, current, and citable — update a doc and the system
knows it immediately, no retraining.

## The decision rule

> "I want a model that *knows our docs*" → that's a retrieval problem. RAG
> first. Always.

CPT earns its place when:

1. The domain's *language itself* is foreign to the base model (dense
   filings-speak, a niche legal sublanguage) — fluency, not facts, is the gap;
2. You have serious corpus volume (~10M+ tokens — below that the model can't
   absorb much and the compute mostly burns); and
3. There's a latency/cost/offline motive: no retrieval round-trip, shorter
   prompts, runs on-device with no document store attached.

The production pattern at Level 3 is **both**: CPT for domain fluency + RAG
for fresh facts with citations. The CPT-ed model reads retrieved passages
*better* because the domain idiom is native to it, and the numbers come from
the source, not the weights.

## Two practical CPT notes

- CPT uses a **base** (non-instruct) model and plain `{"text": ...}` data.
  The result is a text-completer, not an assistant — it needs a small SFT pass
  afterwards to restore instruction-following.
- **Research-mode exception:** tunelab will happily run CPT on a small corpus
  (say 500K tokens) when the goal is *learning* — you'll watch domain
  perplexity drop while factual QA stays unreliable, which is the
  fluency-vs-facts lesson made visible. Expectations are reset accordingly:
  that run is an experiment, not a product.
