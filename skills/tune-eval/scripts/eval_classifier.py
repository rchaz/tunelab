#!/usr/bin/env python3
"""Score classification predictions against gold labels. Stdlib only.

  python3 eval_classifier.py --predictions preds_tuned.jsonl [--report report.md]

Input lines: {"expected": "...", "predicted": "..."} (extra keys ignored).
Prints accuracy, macro-F1, per-class precision/recall/F1, and a confusion matrix.
Macro-F1 averages over gold classes only — labels that were predicted but never
appear in expected are flagged separately as hallucinated (with small models
these are usually generation artifacts: thinking text, refusals, truncation).
Read the confusion matrix, not just the headline — which confusions are
expensive is a product question the user must answer.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict


def norm(s):
    return str(s).strip().lower()


def show(label, width=60):
    """One-line display form of a label (hallucinated ones can be whole paragraphs)."""
    flat = label.replace("\n", "\\n")
    return repr(flat if len(flat) <= width else flat[: width - 3] + "...")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--report", help="also write a markdown report here")
    args = ap.parse_args()

    pairs = []
    with open(args.predictions) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                pairs.append((norm(rec["expected"]), norm(rec["predicted"])))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                sys.exit(f"{args.predictions}:{lineno}: bad prediction line ({type(e).__name__}: {e})")
    if not pairs:
        sys.exit("no predictions found")

    expected_labels = sorted({e for e, _ in pairs})
    pred_counts = Counter(p for _, p in pairs)
    hallucinated = sorted(set(pred_counts) - set(expected_labels))
    labels = sorted(set(expected_labels) | set(pred_counts))
    correct = sum(1 for e, p in pairs if e == p)
    accuracy = correct / len(pairs)

    confusion = defaultdict(Counter)
    for e, p in pairs:
        confusion[e][p] += 1

    stats = {}  # label -> (prec, recall, f1, support)
    for lb in labels:
        tp = confusion[lb][lb]
        support = sum(confusion[lb].values())
        predicted_as = pred_counts[lb]
        prec = tp / predicted_as if predicted_as else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        stats[lb] = (prec, rec, f1, support)

    # Macro over gold classes only: hallucinated labels already cost recall on
    # the gold class they displaced; counting their zero-F1 rows too would
    # double-penalize and make the metric depend on how many junk strings appeared.
    macro_f1 = sum(stats[lb][2] for lb in expected_labels) / len(expected_labels)

    lines = [
        f"n = {len(pairs)}    classes = {len(expected_labels)}    "
        f"accuracy = {accuracy:.3f}    macro-F1 = {macro_f1:.3f}",
        "",
    ]
    lines.append(f"{'class':<20} {'prec':>6} {'recall':>6} {'f1':>6} {'support':>8}")
    for lb in labels:
        prec, rec, f1, support = stats[lb]
        mark = " *" if lb in hallucinated else ""
        lines.append(f"{lb[:20]:<20} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {support:>8}{mark}")
    if hallucinated:
        lines.append("(* = hallucinated: predicted but never in expected)")

    # Rows are gold classes only — hallucinated labels have no expected row,
    # so they appear as columns (where the misprediction landed).
    lines += ["", "confusion matrix (rows = expected, cols = predicted):", ""]
    short = [lb[:10] for lb in labels]
    lines.append(" " * 12 + " ".join(f"{s:>10}" for s in short))
    for lb in expected_labels:
        row = " ".join(f"{confusion[lb][c]:>10}" for c in labels)
        lines.append(f"{lb[:10]:>12} {row}")

    if hallucinated:
        lines += [
            "",
            f"WARNING: {len(hallucinated)} hallucinated label(s) — predicted but never in expected.",
            "Likely a generation artifact (thinking text, truncation, refusal) rather than",
            "a real class confusion; inspect the raw predictions before reading the metrics.",
        ]
        for lb in hallucinated:
            lines.append(f"  {show(lb)} (x{pred_counts[lb]})")
        lines.append("Note: macro-F1 averages gold classes only; sklearn's union-of-labels")
        lines.append("default (which counts the zero-F1 hallucinated rows) would read lower here.")

    out = "\n".join(lines)
    print(out)

    if args.report:
        with open(args.report, "w") as f:
            f.write("# Classification evaluation\n\n```\n" + out + "\n```\n")
        print(f"\nreport -> {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
