# Continuous pretraining as a system

[cpt-vs-rag.md](cpt-vs-rag.md) covers *whether* to continue-pretrain (CPT) at
all: CPT buys domain *fluency*, RAG buys *facts*, and the Level-3 pattern is
both. This doc covers the part most treatments skip — CPT done **continuously**,
as a maintained system rather than a one-shot run.

## Why one-shot CPT decays

A model CPT'd on last quarter's filings drifts as the domain moves: new
vocabulary, new conventions, new entities. The corpus is a moving target, so
the training has to be a loop, not an event — the same flywheel logic as the
[data flywheel](data-flywheels-and-active-learning.md), applied to a domain
corpus instead of prediction logs.

## The loop

```
corpus refresh → delta-chunk the new text → incremental CPT (low LR, with
replay) → small SFT restore pass → eval gates → ship or hold → repeat
```

Each stage has a non-obvious requirement:

- **Delta-chunking** — only the *new* text since the last round, chunked on
  natural boundaries into `{"text": ...}` records. Re-training on the whole
  corpus every round wastes compute and over-fits the old material.
- **Low LR + re-warming** — continued pretraining uses a fraction (~10%) of the
  original pretraining LR; a brief **re-warm then re-decay** schedule stabilizes
  the transition onto new data. Same max-LR as the original run is a known way
  to wreck the model.
- **Replay** — mix a slice of general-domain data back in (experience replay)
  so the model doesn't catastrophically forget how to be a general model while
  it specializes. For small corpora, doing CPT through **LoRA adapters** instead
  of full weights prevents forgetting by construction — the base never moves.
- **SFT restore pass** — CPT on a base model yields a text-completer, not an
  assistant. A small SFT pass afterward restores instruction-following. (CPT on
  an instruct model and skipping this is possible but tends to degrade the
  chat behavior.)
- **Eval gates** — three numbers decide ship/hold: held-out **domain perplexity
  Δ** (did fluency improve?), a **catastrophic-forgetting** slice on a general
  benchmark (did we break the base?), and the real one — **downstream SFT lift**
  (does a task fine-tune on top of the CPT'd base beat the same fine-tune on the
  un-CPT'd base?). Perplexity falling while forgetting stays flat is necessary;
  downstream lift is sufficient.

## When the corpus is too small

CPT wants ~10M+ domain tokens; many real domains don't have them. **Synthetic
continued pretraining** (e.g. EntiGraph) amplifies a small corpus: prompt a
strong model to generate diverse text that connects the entities and facts in
your documents, then CPT on the synthetic corpus. It trades API cost for the
domain tokens you don't have, and it slots into tunelab's synthetic-data path.

## tunelab's stance

CPT is the highest-effort, lowest-frequency rung — gated hard (stable
knowledge, real corpus volume, a latency/offline motive) and sequenced last.
But when it applies, treating it as a *system* (refresh → incremental CPT →
restore → gate) rather than a heroic one-time run is what makes it maintainable.
