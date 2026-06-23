# Synthetic evals & circularity

[Distillation](distillation.md) generates your *training* data from a teacher
model. It's tempting to generate the *eval* set the same way — same prompt, same
teacher, a few hundred fresh examples. Don't, at least not for every question. A
teacher-made eval can fairly rank the *students*, but it cannot fairly judge the
*teacher*. It's grading its own homework.

## Why the teacher can't be graded on its own eval

If the teacher wrote the answer key, "accuracy" just means "agreement with the
teacher." The teacher agrees with itself ~100% by construction, so it tops the
board definitionally — that number is meaningless. Worse, *every* model is
rewarded for matching the teacher's style and idiosyncrasies, not for being
correct. A model that's genuinely right but draws a fuzzy boundary differently
than the teacher scores *lower* than one that faithfully copies the teacher's
quirks.

Measured in tunelab on a synthetic 5-way ticket eval: the teacher (Opus 4.8)
scored **1.000** — definitional, it made the labels — and a second frontier model
(GPT-5.5) scored **0.951**, which was *exactly* the independently-measured
Opus-vs-GPT-5.5 agreement. That 0.951 isn't "GPT-5.5's accuracy"; it's "how close
GPT-5.5 stays to Opus." Reading it as a quality ranking would be circular.

## What a synthetic eval is still good for

It isn't useless — it's *partial*. Two jobs it does well, provided it's built
right:

- **Ranking students against each other.** A small classifier, a LoRA, a
  different embedding — none of them authored the labels, so all are equally
  "outside" the teacher, and the eval ranks them fairly. (This is how tunelab
  found a better *embedding* beating a fine-tune on the same task.)
- **Measuring generalization** — *if* the eval is a genuinely different
  distribution from train and hard-deduped against it. tunelab's ticket eval used
  separate generators, a different scenario mix, and a cross-dedup (max Jaccard
  0.74, 0 dropped), so a high score means "handles unseen phrasing," not
  "memorized the training set." A synthetic eval drawn from the *same* generator
  as train measures memorization and little else.

## Always keep a real anchor

Reserve a small set of **real, human-labeled** examples that no model wrote —
tunelab keeps 25 real tickets alongside the 465 synthetic ones. It's the only set
on which you can fairly compare the teacher to a challenger, because the labels
come from neither of them. Keep it small if you must (real labels are expensive),
but respect what small buys you: wide error bars. A 25-item anchor is a sanity
check and a cross-model fairness check — not a certificate you can hang a
ship/no-ship decision on by itself (see [validation vs
test](validation-vs-test.md), and the small-test-set trap, where a 25-item gold
ranked a fine-tune *above* a classifier that a 465-item eval then ranked *below*
it).

## The rule

Generate training data from a teacher freely. Generate an eval from the teacher
only to rank *students*, and only when it's a different, deduped distribution.
For any comparison that includes the teacher or a sibling model — and for the
final honesty check — use a real-labeled anchor and read it against the [label
ceiling](label-ceiling-and-annotator-agreement.md).
