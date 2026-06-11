# Parameters vs hyperparameters

**Parameters** are what training *learns*: the weights inside the model. You
never set them by hand. In a LoRA run they're the numbers in the adapter
matrices; in a logistic regression they're the coefficients. A 1.7B-parameter
model has 1.7 billion of these.

**Hyperparameters** are what *you* choose before training starts: learning
rate, number of iterations, batch size, LoRA rank, how many layers to adapt.
They control *how* the parameters get learned.

## Why the distinction matters in practice

When a run goes wrong, the fix is almost always a hyperparameter (or the
data) — never the parameters themselves:

- Loss spikes or oscillates → learning rate too high; halve it.
- Loss flat from the start → learning rate too low, or (more often) a data
  format problem.
- Validation loss bottoms out then climbs → too many iterations for this
  dataset; stop earlier (see [epochs and overfitting](epochs-and-overfitting.md)).

## Where the validation set comes in

You can't pick hyperparameters by looking at training loss — a model can drive
training loss to zero by memorizing. You pick them by what they do to
**validation** performance: try a setting, watch val loss, adjust. That's why
the validation set "steers" and is allowed to be looked at repeatedly, while
the test set is not (see [validation vs test](validation-vs-test.md)) — every
hyperparameter you tune against a dataset leaks a little information about
that dataset into your choices.

tunelab's `recommend_hparams` script encodes starting-point heuristics
(iterations scaled to dataset size, learning rate scaled to model size) so the
first run lands in a sane region; the validation curve tells you how to move
from there.
