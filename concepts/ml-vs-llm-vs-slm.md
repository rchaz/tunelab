# Classic ML vs LLM vs SLM

Three different tools that tunelab's capability ladder moves between. Knowing
which is which explains why "fine-tune a model" is often the wrong first move.

**Classic ML** (Levels 0–1): logistic regression, XGBoost, centroids. These
models don't understand language at all — they learn a *decision boundary* in
some numeric feature space. tunelab gets the language understanding from an
embedding model (which turns text into vectors that put similar meanings near
each other), then lets a classic model draw lines between the regions. Training
takes seconds on a laptop CPU, inference is sub-millisecond, and the model
gives you calibrated probabilities you can route on. The catch: the output can
only ever be one of N labels you defined.

**LLM** (large language model): a frontier model like the one you're talking
to. General-purpose reasoning, open-ended generation, follows novel
instructions. You rent it per token. Most pipelines that "need an LLM"
actually use it as a very expensive classifier or formatter — that's the
work tunelab tries to move down the ladder.

**SLM** (small language model, Level 2–3): a 0.5–8B parameter model that runs
on your own hardware. It *is* a language model — it generates open text, follows
a chat format, handles inputs it's never seen — but it has far less general
knowledge and reasoning than a frontier model. The play is specialization:
fine-tuned on a few thousand examples of one narrow task, an SLM can match a
frontier model *on that task* while costing nothing per call.

## The rule of thumb

- Output is one of N fixed labels → classic ML on embeddings. An SLM is
  overkill and usually *less* accurate per unit of effort.
- Output is structured or styled text in a narrow domain → SLM (LoRA).
- Output needs open-ended reasoning on novel problems → stay on the LLM.

The mistake tunelab exists to prevent: reaching for an SLM (days of work,
GPU-hours, eval discipline) when a classifier (minutes, free) clears the bar —
or reaching for either when the task genuinely needs frontier reasoning, which
distillation does not transfer (see [distillation](distillation.md)).
