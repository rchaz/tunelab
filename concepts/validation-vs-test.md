# Validation vs test

tunelab splits every dataset three ways (80/10/10 by default):

| Split | Role | How often you look |
|---|---|---|
| **train** | what the model learns from | constantly (it's the input) |
| **valid** | steers decisions: early stopping, hyperparameter picks, checkpoint choice | as often as you like |
| **test** | the final, honest score | **once** |

## Why the test set is looked at once

Every time you make a decision based on a dataset, information about that
dataset leaks into your model — even if the model never trained on it. Watch
validation loss to pick the best checkpoint and the checkpoint is now
*selected for* that validation set; its val score is a little optimistic.
That's an acceptable price for the validation set, because steering is its job.

The test set's only job is to predict how the model behaves on data it has
never influenced — production traffic, in other words. The first peek spends
that. If you look at test results, go "hmm", change something, retrain, and
re-measure on the same test set, you are now doing slow-motion validation on
it, and the number it reports will be optimistic in ways you can't quantify.

## The contract tunelab enforces

1. **Pre-register the bar.** Decide what score means "ship" *before* running
   the test set (e.g. "≥95% of teacher accuracy", "judge prefers tuned over
   base ≥70%"). A bar chosen after seeing the score is a rationalization.
2. **One look.** Run test, report, decide.
3. **A spent test set stays spent.** If the result sends you back to training,
   carve a fresh test split from new data for the next round when you can.
   Re-using the old one for a second judgment is comparing against a number
   the process has already touched.

This is also why `split_data.py` seeds its shuffle: the same command always
produces the same split, so "untouched" is verifiable, not a memory.
