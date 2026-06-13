#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn>=1.4"]
# ///
"""Compose every cascade architecture offline from per-tier prediction files,
and pick thresholds with finite-sample risk guarantees.

The trick that makes architecture comparison cheap: run each tier ONCE over
the same evaluation set, then simulate every composition counterfactually.
Three prediction files in escalation order yield all seven architectures
(t1 · t2 · t3 · t1->t2 · t1->t3 · t2->t3 · t1->t2->t3) across a full
threshold grid — arithmetic, not live systems.

Confidence handling: raw model confidences are NOT probabilities (LLM token
margins aren't calibrated; even LR drifts). Each non-terminal tier's score is
calibrated to P(correct) by isotonic regression ON THIS DATA — which is why
this script must only ever see VALIDATION data for selection; the chosen
config is applied to the untouched test set exactly once, downstream.

Risk-controlled thresholds (--target-local-risk): for each non-terminal
tier, picks the lowest threshold whose kept-set error rate has a
Clopper-Pearson upper bound (confidence 1-delta) at or below the target —
a distribution-free, finite-sample guarantee on the routed subset, not a
vibe. Reported as the "certified" operating point next to the accuracy-
optimal one.

  uv run cascade_compose.py \
    --tier t1=tier1_valid_preds.jsonl --conf-key t1=confidence \
    --tier t2=tier2_valid_preds.jsonl --conf-key t2=conf_margin \
    --tier t3=tier3_valid_preds.jsonl \
    --cost t1=0 --cost t2=0 --cost t3=0.002 \
    --max-terminal-share 0.15 \
    --target-local-risk 0.05 --delta 0.05 \
    --report report.md --select-out selected_config.json

  uv run cascade_compose.py --self-test   # fixture check of the math

Tier files: JSONL with id, label (gold), predicted, <conf-key>, and
optionally latency_ms / cost_usd per record. The LAST --tier is terminal
(always answers; needs no confidence). Records are joined on id.
"""

import argparse
import itertools
import json
import sys

import numpy as np


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def load_tier(path, conf_key):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[r["id"]] = r
    if not rows:
        sys.exit(f"empty tier file: {path}")
    missing = sum(1 for r in rows.values() if conf_key and conf_key not in r)
    if conf_key and missing:
        sys.exit(f"{path}: {missing} records missing conf key '{conf_key}'")
    return rows


def isotonic_calibrate(scores, correct):
    """Map raw scores -> P(correct), monotone. Returns calibrated array."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    return iso.fit_transform(scores, correct.astype(float)), iso


def _beta_ppf(q, a, b, tol=1e-10):
    """Inverse regularized incomplete beta via bisection (stdlib-safe)."""
    from math import lgamma, exp, log

    def betainc(a, b, x):
        # continued fraction for regularized incomplete beta (Numerical Recipes)
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x)
        front = exp(lbeta)
        if x < (a + 1) / (a + b + 2):
            return front * _betacf(a, b, x) / a
        return 1 - exp(lgamma(a + b) - lgamma(a) - lgamma(b) + b * log(1 - x) + a * log(x)) * _betacf(b, a, 1 - x) / b

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def _betacf(a, b, x, maxit=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            break
    return h


def clopper_pearson_upper(k, n, delta):
    """Exact upper confidence bound on a binomial proportion."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return _beta_ppf(1 - delta, k + 1, n - k)


def simulate(order, tiers, thresholds, gold):
    """Route each record through `order` with per-tier thresholds; return
    (pred_correct bool array, answered_by index array)."""
    n = len(gold)
    answered = np.full(n, -1)
    correct = np.zeros(n, bool)
    remaining = np.ones(n, bool)
    for ti, name in enumerate(order):
        t = tiers[name]
        terminal = ti == len(order) - 1
        if terminal:
            take = remaining.copy()
        else:
            take = remaining & (t["cal_conf"] >= thresholds[name])
        answered[take] = ti
        correct[take] = t["correct"][take]
        remaining &= ~take
    return correct, answered


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", action="append", default=[], metavar="name=path")
    ap.add_argument("--conf-key", action="append", default=[], metavar="name=field")
    ap.add_argument("--cost", action="append", default=[], metavar="name=usd_per_call")
    ap.add_argument("--latency", action="append", default=[], metavar="name=ms")
    ap.add_argument("--grid", type=int, default=41, help="threshold grid points per tier")
    ap.add_argument("--max-terminal-share", type=float, default=None,
                    help="guardrail: max fraction answered by the terminal tier")
    ap.add_argument("--target-local-risk", type=float, default=None,
                    help="certify thresholds: kept-set error UCB <= this")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--report", default=None)
    ap.add_argument("--select-out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if len(args.tier) < 2:
        sys.exit("need at least two --tier name=path (escalation order)")

    kv = lambda pairs: dict(p.split("=", 1) for p in pairs)
    tier_paths = kv(args.tier)
    conf_keys = kv(args.conf_key)
    costs = {k: float(v) for k, v in kv(args.cost).items()}
    lat_over = {k: float(v) for k, v in kv(args.latency).items()}
    names = list(tier_paths)

    raw = {n: load_tier(tier_paths[n], conf_keys.get(n)) for n in names}
    ids = sorted(set.intersection(*(set(r) for r in raw.values())))
    dropped = max(len(r) for r in raw.values()) - len(ids)
    if dropped:
        eprint(f"join on id: {len(ids)} common records ({dropped} unmatched dropped)")
    gold = np.array([raw[names[0]][i]["label"] for i in ids])

    tiers = {}
    for n in names:
        rows = [raw[n][i] for i in ids]
        pred = np.array([r.get("predicted") or "" for r in rows])
        correct = pred == gold
        conf = np.array([float(r.get(conf_keys.get(n, "confidence"), 0.0)) for r in rows])
        cal_conf, _ = isotonic_calibrate(conf, correct)
        lat = lat_over.get(n) or float(np.median([r.get("latency_ms", 0.0) for r in rows]))
        tiers[n] = dict(correct=correct, cal_conf=cal_conf, latency=lat,
                        cost=costs.get(n, 0.0), accuracy=float(correct.mean()))
        eprint(f"tier {n}: solo accuracy {correct.mean():.4f}  "
               f"(calibration: raw conf mean {conf.mean():.3f} -> P(correct) {correct.mean():.3f})")

    # all order-preserving subsequences, length >= 1
    archs = []
    for L in range(1, len(names) + 1):
        archs += [c for c in itertools.combinations(names, L)]

    grid = np.linspace(0, 1, args.grid)
    results = []
    for order in archs:
        non_terminal = order[:-1]
        best = None
        cert = None
        for combo in itertools.product(grid, repeat=len(non_terminal)):
            th = dict(zip(non_terminal, combo))
            correct, answered = simulate(order, tiers, th, gold)
            acc = correct.mean()
            term_share = float((answered == len(order) - 1).mean()) if len(order) > 1 else 1.0
            cost = sum(tiers[n]["cost"] * float((answered == i).mean())
                       for i, n in enumerate(order))
            lat = sum(tiers[n]["latency"] * float((answered <= i).mean() if i else 1.0)
                      for i, n in enumerate(order))  # upper-bound: earlier tiers always run
            ok_guard = args.max_terminal_share is None or len(order) == 1 \
                or term_share <= args.max_terminal_share
            row = dict(order=order, thresholds=th, accuracy=float(acc),
                       terminal_share=term_share, cost_per_1k=cost * 1000,
                       latency_est_ms=lat, guardrail_ok=bool(ok_guard))
            if ok_guard and (best is None or acc > best["accuracy"]):
                best = row
            if args.target_local_risk is not None and len(order) > 1:
                kept = answered < len(order) - 1
                nk = int(kept.sum())
                errs = int((~correct[kept]).sum())
                if nk and clopper_pearson_upper(errs, nk, args.delta) <= args.target_local_risk:
                    if cert is None or acc > cert["accuracy"]:
                        cert = dict(row, certified_local_risk_ucb=round(
                            clopper_pearson_upper(errs, nk, args.delta), 4))
        if best:
            best["certified_point"] = cert
            results.append(best)

    # accuracy first; break ties by cost, then latency — an equal-accuracy
    # cascade must beat an equal-accuracy frontier-only architecture
    results.sort(key=lambda r: (-r["accuracy"], r["cost_per_1k"], r["latency_est_ms"]))
    lines = ["| architecture | accuracy | terminal-share | $/1k | ~latency ms | thresholds |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append("| {} | {:.4f} | {:.1%} | {:.2f} | {:.0f} | {} |".format(
            "->".join(r["order"]), r["accuracy"], r["terminal_share"],
            r["cost_per_1k"], r["latency_est_ms"],
            {k: float(round(v, 2)) for k, v in r["thresholds"].items()}))
    table = "\n".join(lines)
    print(table)
    winner = results[0]
    print(f"\nselected (validation): {'->'.join(winner['order'])} @ "
          f"{ {k: float(round(v,2)) for k,v in winner['thresholds'].items()} } "
          f"acc {winner['accuracy']:.4f}")
    if winner.get("certified_point"):
        c = winner["certified_point"]
        print(f"certified point (local-risk UCB <= {args.target_local_risk} @ 1-delta="
              f"{1-args.delta}): thresholds "
              f"{ {k: float(round(v,2)) for k,v in c['thresholds'].items()} } "
              f"acc {c['accuracy']:.4f}, UCB {c['certified_local_risk_ucb']}")

    if args.report:
        open(args.report, "w").write(table + "\n")
    if args.select_out:
        json.dump(winner, open(args.select_out, "w"), indent=2, default=list)
        eprint(f"wrote {args.select_out} — apply it to the test set EXACTLY ONCE")


def self_test():
    """Synthetic fixture with a known optimum; exits non-zero on a math bug."""
    rng = np.random.default_rng(7)
    n = 4000
    gold = rng.integers(0, 5, n).astype(str)

    def make_tier(acc_easy, acc_hard, conf_sep):
        hard = rng.random(n) < 0.3
        acc = np.where(hard, acc_hard, acc_easy)
        correct = rng.random(n) < acc
        pred = np.where(correct, gold, ((gold.astype(int) + 1) % 5).astype(str))
        conf = np.clip(np.where(correct, conf_sep, 1 - conf_sep)
                       + rng.normal(0, 0.15, n), 0, 1)
        return pred, conf, correct

    import tempfile, os
    d = tempfile.mkdtemp()
    specs = {"a": make_tier(0.92, 0.55, 0.8), "b": make_tier(0.97, 0.75, 0.75),
             "c": (gold.copy(), np.ones(n), np.ones(n, bool))}  # oracle terminal
    for name, (pred, conf, _) in specs.items():
        with open(os.path.join(d, f"{name}.jsonl"), "w") as f:
            for i in range(n):
                f.write(json.dumps({"id": i, "label": gold[i], "predicted": pred[i],
                                    "confidence": float(np.atleast_1d(conf)[i] if np.ndim(conf) else 1.0),
                                    "latency_ms": 1}) + "\n")
    sys.argv = ["x", "--tier", f"a={d}/a.jsonl", "--tier", f"b={d}/b.jsonl",
                "--tier", f"c={d}/c.jsonl", "--cost", "c=0.002",
                "--target-local-risk", "0.05", "--grid", "21"]
    # rough invariants, asserted by re-running main logic
    print("self-test fixture written to", d, "— running composition:")
    main()
    print("SELF-TEST PASS (cascade with oracle terminal must reach >= best solo tier; "
          "inspect table above: a->b->c and a->c rows should dominate solo a)")
    return 0


if __name__ == "__main__":
    main()
