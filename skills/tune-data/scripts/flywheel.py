#!/usr/bin/env python3
"""The data flywheel: turn served predictions + feedback into the next training
round. Stdlib only.

A deployed model (single, cascade, or workflow) logs every prediction. Feedback
arrives three ways: explicit (a human corrects a label), automated (a
programmatic verifier — the distiller's grounding gate, a downstream success
signal), and structural (in a cascade, an escalation tier's answer is a free
label for the tiers below it). This script is the Analyze + Plan half of a
MAPE control loop (Monitor-Analyze-Plan-Execute): it reads the log, decides
whether a retrain is warranted, and emits a retrain MANIFEST — it never trains
on its own (Execute = tune-train, gated on a human at spend/promotion).

Subcommands:

  init     Create an empty append-only prediction log with the agreed schema.

  log      Append one or more prediction records (used by a serving harness;
           also handy for tests). Schema per line:
             {ts, input_hash, text, tier, prediction, confidence,
              latency_ms, cost_usd, feedback, feedback_source, model_version}
           feedback is null until known; feedback_source in
           {explicit, automated, structural, audit}.

  status   Monitor + Analyze: rolling accuracy on the feedback window,
           confidence-distribution drift vs a reference window (population
           stability index), per-tier coverage, and whether each retrain
           trigger fires. Read-only.

  plan     Plan: if a trigger fires, emit a retrain manifest — which tier(s),
           which curated records (deduped, conflicts resolved, test ids
           excluded), and the pre-registered promotion bar to beat. Writes a
           manifest JSON; does not train.

Triggers (any fires -> retrain candidate):
  --min-new-labels N     >= N new feedback-bearing records since last round
  --drift-psi P          confidence-distribution PSI vs reference >= P
  --accuracy-floor A     rolling feedback-window accuracy < A

Design honesty (see PLAN-V2 §15.2): feedback over-samples hard/escalated cases,
so feedback-window accuracy is NOT the served accuracy. `status` reports the
audit-slice accuracy (feedback_source=audit, a uniform random sample) SEPARATELY
and flags when only biased feedback is available. Curation in `plan` excludes
any id present in --holdout-ids so the flywheel can never train on its own test
set.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict


SCHEMA_FIELDS = ["ts", "input_hash", "text", "tier", "prediction", "confidence",
                 "latency_ms", "cost_usd", "feedback", "feedback_source",
                 "model_version"]


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def read_log(path):
    rows = []
    try:
        for line in open(path):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except FileNotFoundError:
        sys.exit(f"no log at {path} — run `flywheel.py init --log {path}` first")
    return rows


def psi(reference, current, bins=10):
    """Population Stability Index between two score distributions.
    PSI < 0.1 stable; 0.1-0.25 moderate drift; > 0.25 significant."""
    if not reference or not current:
        return 0.0
    lo, hi = 0.0, 1.0
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    edges[-1] += 1e-9

    def hist(xs):
        h = [0] * bins
        for x in xs:
            x = min(max(x, lo), hi)
            for i in range(bins):
                if edges[i] <= x < edges[i + 1]:
                    h[i] += 1
                    break
        n = sum(h) or 1
        return [c / n for c in h]

    r, c = hist(reference), hist(current)
    total = 0.0
    for ri, ci in zip(r, c):
        ri = ri or 1e-6
        ci = ci or 1e-6
        total += (ci - ri) * (json_log(ci / ri))
    return total


def json_log(x):
    from math import log
    return log(x)


def cmd_init(args):
    import os
    if os.path.exists(args.log) and os.path.getsize(args.log) > 0:
        sys.exit(f"{args.log} already exists and is non-empty — refusing to clobber")
    open(args.log, "a").close()
    eprint(f"created append-only prediction log: {args.log}")
    eprint("schema per line: " + ", ".join(SCHEMA_FIELDS))


def cmd_log(args):
    rows = [json.loads(l) for l in open(args.records)] if args.records else []
    if args.json:
        rows.append(json.loads(args.json))
    with open(args.log, "a") as f:
        for r in rows:
            r.setdefault("input_hash", hashlib.sha256(
                str(r.get("text", "")).encode()).hexdigest()[:16])
            for k in SCHEMA_FIELDS:
                r.setdefault(k, None)
            f.write(json.dumps(r) + "\n")
    eprint(f"appended {len(rows)} record(s) to {args.log}")


def _accuracy(rows):
    graded = [r for r in rows if r.get("feedback") is not None]
    if not graded:
        return None, 0
    ok = sum(1 for r in graded if r["prediction"] == r["feedback"])
    return ok / len(graded), len(graded)


def cmd_status(args):
    rows = read_log(args.log)
    eprint(f"{len(rows)} logged predictions")

    # per-tier coverage
    tier_counts = Counter(r.get("tier") for r in rows)
    print("## tier coverage")
    for t, c in sorted(tier_counts.items(), key=lambda kv: str(kv[0])):
        print(f"- {t}: {c} ({c/len(rows):.1%})")

    # feedback accuracy (biased) vs audit accuracy (unbiased)
    feedback_rows = [r for r in rows if r.get("feedback") is not None]
    audit_rows = [r for r in feedback_rows if r.get("feedback_source") == "audit"]
    biased_rows = [r for r in feedback_rows if r.get("feedback_source") != "audit"]
    acc_all, n_all = _accuracy(feedback_rows)
    acc_audit, n_audit = _accuracy(audit_rows)
    acc_biased, n_biased = _accuracy(biased_rows)
    print("\n## accuracy")
    print(f"- all feedback (BIASED toward hard/escalated cases): "
          f"{acc_all:.4f} (n={n_all})" if acc_all is not None else "- no feedback yet")
    if acc_audit is not None:
        print(f"- audit slice (uniform random — the served-accuracy estimate): "
              f"{acc_audit:.4f} (n={n_audit})")
    else:
        print("- audit slice: NONE — served accuracy is unknown; add a uniform "
              "random audit sample before trusting any retrain trigger")

    # drift: PSI on confidence, recent vs reference window
    confs = [r["confidence"] for r in rows if r.get("confidence") is not None]
    if len(confs) >= 2 * args.window:
        ref = confs[: -args.window]
        cur = confs[-args.window:]
        p = psi(ref, cur)
        band = "stable" if p < 0.1 else "moderate" if p < 0.25 else "SIGNIFICANT"
        print(f"\n## drift\n- confidence PSI (last {args.window} vs prior): {p:.3f} ({band})")
    else:
        p = 0.0
        print(f"\n## drift\n- not enough history for PSI (need {2*args.window})")

    # triggers
    new_labeled = len(feedback_rows)  # for a real deployment, since last round
    print("\n## retrain triggers")
    fired = []
    if args.min_new_labels:
        hit = new_labeled >= args.min_new_labels
        fired.append(hit)
        print(f"- min-new-labels {args.min_new_labels}: {new_labeled} -> {'FIRE' if hit else 'hold'}")
    if args.drift_psi:
        hit = p >= args.drift_psi
        fired.append(hit)
        print(f"- drift-psi {args.drift_psi}: {p:.3f} -> {'FIRE' if hit else 'hold'}")
    if args.accuracy_floor and acc_all is not None:
        ref_acc = acc_audit if acc_audit is not None else acc_all
        hit = ref_acc < args.accuracy_floor
        fired.append(hit)
        print(f"- accuracy-floor {args.accuracy_floor}: {ref_acc:.4f} -> {'FIRE' if hit else 'hold'}")
    if fired:
        print(f"\n=> {'RETRAIN CANDIDATE' if any(fired) else 'no trigger — hold'}; "
              f"run `flywheel.py plan` to materialize a manifest" if any(fired) else "")


def cmd_plan(args):
    rows = read_log(args.log)
    holdout = set()
    if args.holdout_ids:
        holdout = {l.strip() for l in open(args.holdout_ids) if l.strip()}

    # curation: feedback-bearing, not in holdout, conflicts resolved (latest wins),
    # deduped on input_hash
    by_hash = {}
    excluded_holdout = 0
    for r in rows:
        if r.get("feedback") is None:
            continue
        h = r.get("input_hash") or hashlib.sha256(
            str(r.get("text", "")).encode()).hexdigest()[:16]
        if h in holdout or r.get("id") in holdout:
            excluded_holdout += 1
            continue
        by_hash[h] = r  # later record wins -> conflict resolution by recency
    curated = list(by_hash.values())

    # structural labels: in a cascade, terminal-tier answers are gold for lower tiers.
    # We surface the count; the actual reuse is the trainer's job per the manifest.
    structural = sum(1 for r in curated if r.get("feedback_source") == "structural")

    manifest = {
        "round": args.round,
        "curated_records": len(curated),
        "excluded_holdout": excluded_holdout,
        "structural_labels": structural,
        "target_tiers": args.target_tiers.split(",") if args.target_tiers else ["tier1"],
        "promotion_bar": args.promotion_bar,
        "promotion_rule": "challenger replaces champion only if it BEATS the "
                          "pre-registered bar on a FRESH never-used eval slice; "
                          "ties retain champion (cost/latency tiebreak)",
        "curated_out": args.curated_out,
    }
    with open(args.curated_out, "w") as f:
        for r in curated:
            f.write(json.dumps({"text": r.get("text"), "label": r.get("feedback"),
                                "input_hash": r.get("input_hash")}) + "\n")
    json.dump(manifest, open(args.manifest_out, "w"), indent=2)
    eprint(f"curated {len(curated)} records -> {args.curated_out} "
           f"({excluded_holdout} holdout-excluded, {structural} structural)")
    eprint(f"manifest -> {args.manifest_out}")
    eprint("NEXT: human reviews the manifest, then tune-train executes; "
           "tune-eval adjudicates against the promotion bar on a fresh slice.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init"); pi.add_argument("--log", required=True)
    pi.set_defaults(func=cmd_init)

    pl = sub.add_parser("log")
    pl.add_argument("--log", required=True)
    pl.add_argument("--records", help="JSONL file of records to append")
    pl.add_argument("--json", help="a single record as a JSON string")
    pl.set_defaults(func=cmd_log)

    ps = sub.add_parser("status")
    ps.add_argument("--log", required=True)
    ps.add_argument("--window", type=int, default=200, help="drift window size")
    ps.add_argument("--min-new-labels", type=int)
    ps.add_argument("--drift-psi", type=float)
    ps.add_argument("--accuracy-floor", type=float)
    ps.set_defaults(func=cmd_status)

    pp = sub.add_parser("plan")
    pp.add_argument("--log", required=True)
    pp.add_argument("--round", type=int, default=1)
    pp.add_argument("--target-tiers", help="comma list, e.g. tier1,tier2")
    pp.add_argument("--holdout-ids", help="file of ids/hashes that must NOT train")
    pp.add_argument("--promotion-bar", default="accuracy > champion on fresh slice")
    pp.add_argument("--curated-out", default="retrain_curated.jsonl")
    pp.add_argument("--manifest-out", default="retrain_manifest.json")
    pp.set_defaults(func=cmd_plan)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
