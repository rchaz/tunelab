# Recipe 4 — Finance filings analyst (CPT showcase) — SKELETON

**Level 3 · continued pretraining as a *maintained system*, not a one-shot · SEC EDGAR is free
and reproducible by anyone. This recipe is a pre-registered skeleton: the metric card and
pipeline are fixed here; the live run is the next CPT-capable session.**

## The problem

Some domains speak a language the base model only half-knows — dense filings-speak, a niche legal
sublanguage. RAG fetches *facts*; it doesn't fix *fluency*. Continued pretraining (CPT) on a
domain corpus buys the fluency, and the production pattern is **CPT for domain idiom + RAG for
fresh facts with citations** (see [concepts/cpt-vs-rag.md](../concepts/cpt-vs-rag.md)). The
numbers come from retrieved source spans; the *reading* of them comes from a model that speaks the
domain natively.

## Why this is a system, not a run

A model CPT'd on last quarter's filings drifts as the domain moves. So CPT is a loop, the same
flywheel logic applied to a corpus:

```
corpus refresh → delta-chunk the new filings → incremental CPT (low LR, replay) →
small SFT restore pass → eval gates → ship or hold → repeat
```

Each stage has a non-obvious requirement (full detail:
[concepts/continuous-pretraining.md](../concepts/continuous-pretraining.md)): delta-chunk only
new text; LR re-warm then re-decay at ~10% of pretraining LR; mix general-data **replay** (or use
LoRA-CPT) to prevent catastrophic forgetting; an SFT restore pass because CPT yields a
text-completer, not an assistant.

## Data source

SEC EDGAR 10-K / 10-Q / earnings-call transcripts — public-domain, fetchable without auth, and
the canonical "the language itself is foreign" corpus. For a small corpus (under the ~10M-token
gate), **EntiGraph-style synthetic CPT** amplifies it: prompt a strong model to generate diverse
text connecting the corpus's entities, then CPT on the synthetic corpus (trades API cost for the
domain tokens you don't have).

## Pre-registered metric card (logged before any CPT run)

- **Primary — downstream SFT lift:** a task fine-tune (hawkish/dovish classification, or
  guidance-change extraction) on the CPT'd base vs the same fine-tune on the un-CPT'd base. The
  *lift from CPT* is the headline number — necessary because perplexity alone doesn't prove
  usefulness.
- **Domain perplexity Δ** on held-out filings (did fluency improve?).
- **Guardrail — catastrophic forgetting:** a general-benchmark slice before/after (did we break
  the base?). Numeric-fact policy: the model never sources figures from weights — extraction must
  cite retrieved spans; spot-audit.
- Probe sets exist off-the-shelf (FinanceBench, FinQA).

## CPT gate (all three must hold — or research-mode bypass)

(a) the knowledge is stable (not weekly-changing facts — those are RAG's job); (b) ~10M+ tokens
of raw domain text (or synthetic amplification); (c) a latency/cost/offline motive. Research-mode
bypass: small-corpus CPT is allowed for *learning*, with expectations reset to perplexity/fluency,
not QA accuracy.

## Status

Skeleton only — metric card and pipeline pre-registered here. The live EDGAR run, the synthetic-
amplification path, and the maintained-loop integration with `tune-loop` are the next CPT-capable
session's deliverables.
