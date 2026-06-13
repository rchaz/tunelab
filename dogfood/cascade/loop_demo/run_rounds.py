#!/usr/bin/env python3
"""Phase E dogfood: N sustained champion/challenger rounds, CPU-only.

Each round the flywheel "accrues feedback" (a larger training pool); the
challenger is adjudicated on a FRESH one-look slice via promote.py — reusing a
slice is a hard error, so eval-burn is mechanically prevented. The champion is
the running winner; promotion bumps the descriptor version.
"""
import json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = "dogfood/cascade/loop_demo"
TC = "skills/tune-decide/scripts/train_classifier.py"
PROMOTE = "skills/tune-loop/scripts/promote.py"
os.chdir(ROOT)


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True)


def acc(preds, gold_fp):
    gold = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(gold_fp)}
    rows = [json.loads(l) for l in open(preds)]
    return round(sum(r["predicted"] == gold[r["id"]] for r in rows) / len(rows), 4)


def model_for(n):
    mp = f"{D}/m_{n}.joblib"
    if not os.path.exists(mp):
        sh("uv", "run", TC, "--data", f"{D}/train_{n}.jsonl", "--model-out", mp, "--seed", "42")
    return mp


def eval_on(n, slice_name, tag):
    mp = model_for(n)
    out = f"{D}/p_{n}_{slice_name}.jsonl"
    sh("uv", "run", TC, "--predict", f"{D}/{slice_name}.jsonl", "--model-in", mp, "--output", out)
    a = acc(out, f"{D}/{slice_name}.jsonl")
    ej = f"{D}/{tag}.json"
    json.dump({"accuracy": a, "cost_per_1k": 0.0}, open(ej, "w"))
    return a, ej


ledger = f"{D}/consumed_slices.txt"
desc = f"{D}/descriptor.json"
log = f"{D}/rounds.md"
for f in (ledger, log):
    if os.path.exists(f):
        os.remove(f)
json.dump({"version": 1, "kind": "single", "model": "lr-2000"}, open(desc, "w"))

rounds = [(4000, "slice1"), (6000, "slice2"), (8005, "slice3")]
champion_n = 2000
report = ["# Phase E — sustained champion/challenger rounds (Banking77 replay, CPU)\n"]
for i, (chal_n, sl) in enumerate(rounds, 1):
    ca, cj = eval_on(champion_n, sl, f"champ_{i}")
    xa, xj = eval_on(chal_n, sl, f"chal_{i}")
    r = sh("uv", "run", PROMOTE, "--champion", cj, "--challenger", xj,
           "--bar", "0.84", "--min-margin", "0.005", "--metric", "accuracy",
           "--slice-id", sl, "--ledger", ledger,
           "--descriptor-in", desc, "--descriptor-out", desc)
    decision = "PROMOTE" if "PROMOTE" in r.stdout else "RETAIN"
    line = (f"- **Round {i}**: champion lr-{champion_n} ({ca}) vs challenger lr-{chal_n} ({xa}) "
            f"on one-look {sl} → **{decision}** ({xa-ca:+.4f})")
    print(line)
    report.append(line)
    if decision == "PROMOTE":
        champion_n = chal_n

ver = json.load(open(desc))["version"]
report.append(f"\nFinal champion: lr-{champion_n}; descriptor version {ver} "
              f"(started v1 → {ver-1} promotions). Eval-burn guard: each round used a disjoint "
              f"one-look slice; the ledger refuses reuse.")
open(log, "w").write("\n".join(report) + "\n")
print(f"\nwrote {log}; descriptor v{ver}")
