# Epochs, iterations, and overfitting

**One epoch** = the model has seen every training example once. mlx-lm counts
**iterations** (batches) instead, so the conversion is:

```
iterations = epochs × n_examples / batch_size
# 800 examples, batch 4, 5 epochs → 1000 iterations
```

Small datasets need several epochs — one pass isn't enough signal. Large
datasets need fewer; the same repetition that helps a small dataset becomes
memorization on a large one.

## Overfitting: memorize vs generalize

Early in training the model learns the *task* — patterns that transfer to new
inputs. Train loss falls, validation loss falls with it. Keep going and the
model starts learning the *training set* — the specific phrasings, the quirks
of individual examples. Train loss keeps falling; validation loss bottoms out
and climbs. That U-turn in val loss is overfitting, observed live.

```
loss
 │ ╲ train
 │  ╲_____
 │   ╲    ‾‾──____
 │    ╲ valid
 │     ╲____
 │          ╲___╱‾‾   ← val loss turns: stop here
 └──────────────────── iterations
```

**Early stopping** is the whole remedy: note the iteration where validation
loss was lowest and use the checkpoint from there (mlx-lm saves one every
`--save-every` iterations). On small datasets the U-turn is *expected*, not a
failure — you deliberately over-shoot iterations a little so you can see the
bottom, then keep the checkpoint from the bottom.

## Two ways to overfit without noticing

- **Duplicates are hidden epochs.** A record appearing 50 times trains ~50
  epochs on that one example while everything else gets the nominal count —
  the model learns to recite it verbatim. This is why dedupe is a gate, not a
  suggestion, and why it matters most for continued-pretraining corpora.
- **Capacity oversized for the data.** More trainable parameters memorize
  faster. Under ~300 examples, tunelab drops `--num-layers` from 16 to 8 to
  shrink the adapter's capacity to something the dataset can constrain.

Want to *feel* this instead of reading it? tunelab's research mode runs an
overfit-on-purpose experiment: tiny dataset, too many iterations, predict the
curve before you run it, then watch.
