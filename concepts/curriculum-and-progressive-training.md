# Curriculum & progressive training

Vanilla SFT shuffles the training set and shows it to the model in random
order. **Curriculum learning** asks whether *order* matters — easy examples
first, hard ones later, the way humans learn. Sometimes it helps; tunelab is
honest about when it's worth the trouble.

## The idea

Order training data by difficulty and present it easy→hard. The intuition: a
model that has mastered simple cases has better representations to learn the
hard ones from, so it converges faster and generalizes better — and it avoids
early-training instability from being slammed with the hardest examples while
its weights are still random-ish.

## Difficulty signals that actually work

You need a cheap proxy for "hard." The ones with empirical support in 2025–26:

- **Compression ratio** — how compressible the text is (less compressible ≈
  more information-dense ≈ harder). Directly available for tunelab's distiller:
  the compression ratio *is* a difficulty signal.
- **Lexical diversity** (MTLD) and **readability** (Flesch) — proxies for
  linguistic complexity.
- **Sequence length** — longer records are usually harder *and* more
  memory-hungry, so a short→long curriculum doubles as a way to keep early
  iterations cheap (the long records that blow the activation budget come after
  the optimizer has stabilized).

## The honest caveat

The evidence is mixed, especially at small scale. The finding that matters:
**difficulty alone isn't enough — sample *utility* matters too.** Ordering by
difficulty while ignoring whether a sample actually teaches anything new can
waste capacity on examples the model already handles. And on small datasets the
gains are often within noise; curriculum is not a reliable free lunch.

## tunelab's stance

Curriculum is a **research-mode experiment and a cheap A/B**, not a default.
Two places it earns a look:

1. **Memory-shaped curricula** — short→long ordering on variable-length data
   (like the distiller's compression records) softens early-iteration memory
   spikes regardless of whether it helps accuracy. A practical win even if the
   accuracy delta is zero.
2. **Pre-registered A/B** — order by a difficulty signal, train, compare to a
   shuffled-data control on a held-out slice. If it beats the control past the
   noise threshold, keep it; if not, you've learned that shuffling was fine and
   you spent one cheap run to know it.

Predict the outcome first, run both arms, log the delta — the same predict-then-
run discipline as the rest of tunelab's research mode.
