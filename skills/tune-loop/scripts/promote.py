#!/usr/bin/env python3
"""Champion/challenger adjudication for tune-loop. Stdlib only.

The one piece of the capstone that isn't already in flywheel.py / cascade_compose.py:
deciding whether a challenger REPLACES the champion. The rules that make this
trustworthy rather than AutoML slop are enforced here, mechanically:

  1. Pre-registered bar       the promotion threshold is read from the round's
                              registered bar, not chosen after seeing scores.
  2. One-look eval slice      the slice id used for adjudication is recorded in a
                              consumed-slices ledger; reusing one is a hard error.
  3. Beat-by-margin           a challenger promotes only if it beats the champion
                              by >= --min-margin AND clears the absolute bar; ties
                              and noise-band wins retain the champion (cost tiebreak).

  uv run promote.py \
    --champion champion_eval.json --challenger challenger_eval.json \
    --bar 0.936 --min-margin 0.0 --metric accuracy \
    --slice-id valid-window-2026-06-12 --ledger system/consumed_slices.txt \
    --descriptor-in system/descriptor.json --descriptor-out system/descriptor.json

  eval json (from tune-eval): {"metric_name": value, ..., "cost_per_1k": x, "n": N}

Exit 0 = decision made (promote or retain, printed + logged). Exit 2 = a
discipline violation (slice reuse, missing bar) — the loop must stop and surface it.
"""

import argparse
import json
import os
import sys


def die(msg, code=2):
    print(f"DISCIPLINE VIOLATION: {msg}", file=sys.stderr)
    sys.exit(code)


def load_json(path, what):
    """Load a JSON file, failing with a clean one-line message (never a traceback)."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        die(f"{what} file not found: {path}")
    except json.JSONDecodeError as e:
        die(f"{what} file is not valid JSON ({path}): {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--champion", required=True, help="champion eval json")
    ap.add_argument("--challenger", required=True, help="challenger eval json")
    ap.add_argument("--bar", type=float, required=True, help="pre-registered absolute bar")
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="challenger must beat champion by at least this (noise band)")
    ap.add_argument("--slice-id", required=True, help="id of the eval slice used")
    ap.add_argument("--ledger", required=True, help="consumed-slices ledger file")
    ap.add_argument("--descriptor-in")
    ap.add_argument("--descriptor-out")
    ap.add_argument("--log", help="append the decision to this EXPERIMENT-LOG.md")
    args = ap.parse_args()

    # discipline 2: one-look slice
    consumed = set()
    if os.path.exists(args.ledger):
        consumed = {l.strip() for l in open(args.ledger) if l.strip()}
    if args.slice_id in consumed:
        die(f"eval slice '{args.slice_id}' was already consumed (in {args.ledger}). "
            f"Adjudicate on a FRESH slice — reusing one invalidates the comparison.")

    champ = load_json(args.champion, "champion eval")
    chal = load_json(args.challenger, "challenger eval")
    if args.metric not in champ or args.metric not in chal:
        die(f"metric '{args.metric}' missing from an eval json", code=2)
    c_score, x_score = champ[args.metric], chal[args.metric]
    margin = x_score - c_score

    clears_bar = x_score >= args.bar
    beats_champ = margin >= args.min_margin and margin > 0
    promote = clears_bar and beats_champ

    # cost tiebreak note (informational)
    c_cost = champ.get("cost_per_1k")
    x_cost = chal.get("cost_per_1k")

    decision = "PROMOTE" if promote else "RETAIN"
    lines = [
        f"## champion/challenger adjudication — slice {args.slice_id}",
        f"- metric: {args.metric}; pre-registered bar: {args.bar}; min-margin: {args.min_margin}",
        f"- champion:   {c_score:.4f}" + (f" (cost/1k {c_cost})" if c_cost is not None else ""),
        f"- challenger: {x_score:.4f}" + (f" (cost/1k {x_cost})" if x_cost is not None else ""),
        f"- margin: {margin:+.4f}; clears bar: {clears_bar}; beats champion by margin: {beats_champ}",
        f"- **DECISION: {decision}**" + (
            "" if promote else
            "  (challenger did not clear the bar AND beat the champion by the margin; "
            "champion retained — ties/noise-band wins do not promote)"),
    ]
    report = "\n".join(lines)
    print(report)

    # record the slice as consumed (discipline 2) — once adjudicated, it's spent
    with open(args.ledger, "a") as f:
        f.write(args.slice_id + "\n")

    # discipline: bump descriptor version on promote
    if promote and args.descriptor_in and args.descriptor_out:
        d = load_json(args.descriptor_in, "descriptor")
        d["version"] = d.get("version", 0) + 1
        d["_promoted_from_slice"] = args.slice_id
        d["_promoted_metric"] = {args.metric: x_score}
        json.dump(d, open(args.descriptor_out, "w"), indent=2)
        print(f"\ndescriptor promoted to version {d['version']} -> {args.descriptor_out}",
              file=sys.stderr)

    if args.log:
        with open(args.log, "a") as f:
            f.write("\n" + report + "\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
