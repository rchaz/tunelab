#!/usr/bin/env python3
"""Split a JSONL dataset into train/valid/test for MLX-LM. Stdlib only.

  python3 split_data.py --input deduped.jsonl --outdir data/ \
      [--ratios 0.8,0.1,0.1] [--seed 42] [--label-key label | --label-from-assistant]

Writes data/train.jsonl, data/valid.jsonl, data/test.jsonl (mlx_lm's expected
names). Pass --label-key (top-level field) or --label-from-assistant (last
assistant message is the label, the distillation chat format) to stratify —
without stratification a rare class can vanish from the test set.

Downstream contract: valid.jsonl steers training; test.jsonl is looked at
exactly once, by tune-eval.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict


def get_label(rec, label_key, from_assistant):
    if label_key:
        return str(rec.get(label_key, "_none"))
    if from_assistant:
        for m in reversed(rec.get("messages", [])):
            if m.get("role") == "assistant":
                return str(m.get("content", "_none"))
    return "_all"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ratios", default="0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=42)
    strat = ap.add_mutually_exclusive_group()
    strat.add_argument("--label-key")
    strat.add_argument("--label-from-assistant", action="store_true")
    args = ap.parse_args()

    try:
        ratios = [float(x) for x in args.ratios.split(",")]
    except ValueError:
        sys.exit("--ratios must be three non-negative numbers summing to 1.0")
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6 or min(ratios) < 0:
        sys.exit("--ratios must be three non-negative numbers summing to 1.0")

    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if len(records) < 10:
        sys.exit(f"only {len(records)} records — too few to split meaningfully")

    rnd = random.Random(args.seed)
    groups = defaultdict(list)
    for rec in records:
        groups[get_label(rec, args.label_key, args.label_from_assistant)].append(rec)

    splits = {"train": [], "valid": [], "test": []}
    for label, recs in sorted(groups.items()):
        rnd.shuffle(recs)
        n = len(recs)
        # Clamp: independent rounding can make train+valid exceed n on tiny
        # classes (e.g. n=3 at 0.5/0.5/0 rounds to 2+2), double-counting records.
        n_train = min(round(n * ratios[0]), n)
        n_valid = min(round(n * ratios[1]), n - n_train)
        splits["train"].extend(recs[:n_train])
        splits["valid"].extend(recs[n_train : n_train + n_valid])
        splits["test"].extend(recs[n_train + n_valid :])
        if len(groups) > 1 and min(n - n_train - n_valid, n_valid) == 0:
            print(f"  warning: class '{label}' (n={n}) has an empty split", file=sys.stderr)

    # Catch the unstratified case too (the per-class warning above only fires
    # when stratifying): an empty test.jsonl silently breaks the tune-eval contract.
    empty = [name for name, part in splits.items() if not part]
    if empty:
        print(
            f"  warning: empty split(s): {', '.join(empty)} — valid steers training and "
            "test judges once; adjust --ratios or add data",
            file=sys.stderr,
        )

    for part in splits.values():
        rnd.shuffle(part)

    os.makedirs(args.outdir, exist_ok=True)
    for name, part in splits.items():
        path = os.path.join(args.outdir, f"{name}.jsonl")
        with open(path, "w") as f:
            for rec in part:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} -> {path}", file=sys.stderr)

    if len(groups) > 1:
        print(f"stratified over {len(groups)} classes", file=sys.stderr)


if __name__ == "__main__":
    main()
