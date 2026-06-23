# Distillation

Distillation = using a strong **teacher** model's outputs as training data for
a small **student** model. Instead of paying someone to label 5,000 emails,
the teacher labels them; instead of writing 2,000 ideal support replies, the
teacher drafts them. The student then learns to imitate the teacher *on that
task*.

## What transfers, and what doesn't

The student doesn't inherit the teacher's intelligence. It inherits the
teacher's *behavior on the distribution you sampled*:

- **Transfers well:** narrow, repeated behavior — fixed-label decisions,
  one schema of structured extraction, one voice of reply, one compression
  format. The teacher's outputs are consistent, so a small model can learn
  the mapping.
- **Transfers badly:** general reasoning, knowledge breadth, handling of
  genuinely novel inputs. The student saw a few thousand examples; the
  teacher's capabilities came from a few trillion tokens. Distill an
  open-ended reasoning task and you get a model that mimics the *format* of
  reasoning without the substance.

This is why tunelab's ladder routes "open-ended reasoning" back to the
frontier model, and why every shipped distillation gets **confidence
routing**: inputs the student is unsure about go to the teacher, and those
routed cases become the next round of training data.

## The levels of distillation in tunelab

- **Level 1** is distillation in its cheapest form: the teacher's logged
  *labels* train a classifier on embeddings. No language model is trained.
- **Level 2** distills the teacher's *outputs*: input → teacher-quality output
  pairs fine-tune a small language model (LoRA).
- In both cases the eval compares **student vs teacher** directly — "how much
  did distillation lose" is a measured number, not a vibe.

## Quality hygiene

A bad teacher prompt poisons the whole dataset, which is why tunelab forces a
25-example spot-check before scaling any labeling run, and why the DATACARD
records the teacher model and the exact prompt used — provenance you'll want
when an eval result looks weird.

## Terms-of-service note

Most frontier-model providers restrict using their outputs to train *competing
models*. A narrow internal task model trained on your own logged traffic is a
different posture from a competing general-purpose model — but that call is
yours, and tunelab's job is to surface it, not lawyer it. If the clause
worries you, open-weights teachers (Qwen's Apache-2.0 models, DeepSeek's MIT
models) are the clean option. Llama does not fully qualify: its license
attaches a naming requirement to models trained on Llama outputs. Whatever you
choose, the DATACARD records teacher + intended use.
