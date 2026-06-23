# Banking77 — tunelab quickstart sample

The dataset every tunelab quickstart runs on. One familiar problem, three lenses
(classify it cheaply, fine-tune a small model for it, keep it improving).

## What it is

**Banking77** — real online-banking customer messages, each labeled with one of
**77 fine-grained intents** (e.g. `card_arrival`, `cancel_transfer`,
`balance_not_updated_after_bank_transfer`). Fine-grained, heavily overlapping
classes make it a genuinely hard classification task — which is exactly why it's
a good teaching example.

## Files

| File | Rows | Shape | Used by |
|---|---|---|---|
| `train.jsonl` | 8,005 | `{id, text, label}` | `quickstart cost`, `quickstart loop` |
| `test.jsonl` | 3,080 | `{id, text, label}` | `quickstart loop` (held-out adjudication slice) |
| `labels.json` | 77 | list of label strings | reference |
| `chat/train.jsonl` | 600 | `{messages:[system,user,assistant]}` | `quickstart finetune` |
| `chat/valid.jsonl` | 150 | `{messages:[...]}` | `quickstart finetune` |

The `chat/` files are derived from `train.jsonl` (seeded shuffle, seed 42),
reframed as supervised chat turns so a small model learns to *emit* the label —
the labels live in the model's weights, not in the prompt.

## Provenance & license

- Source: **Banking77**, Casanueva et al., *Efficient Intent Detection with Dual
  Sentence Encoders* (PolyAI, 2020) — <https://huggingface.co/datasets/banking77>
- License: **CC-BY-4.0**. Attribution above satisfies the license; if you
  redistribute, keep it.
- This copy is a fixed train/test split used across tunelab's recipes and dogfood
  runs, so quickstart numbers reproduce exactly.

## Point it at your own data

Every quickstart takes a `--data` / `--train` path. Swap in your own JSONL with
the same shape (`{text, label}` for classify, `{messages:[...]}` for fine-tune)
and the same command works on your problem.
