#!/usr/bin/env python3
"""Evidence tests for skills/tune-data/scripts/split_data.py.

Invokes the real script via subprocess. Prints one 'PASS: <check>' line per
check; exits non-zero on the first failure.
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "skills", "tune-data", "scripts", "split_data.py")
FIX = os.path.join(ROOT, "tests", "fixtures", "hygiene")
STRAT = os.path.join(FIX, "split_stratified.jsonl")
TINY = os.path.join(FIX, "split_tiny.jsonl")
SPLITS = ("train", "valid", "test")


def run(args):
    return subprocess.run(
        ["python3", SCRIPT] + args, capture_output=True, text=True
    )


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def read_records(outdir):
    out = {}
    for name in SPLITS:
        path = os.path.join(outdir, f"{name}.jsonl")
        if not os.path.exists(path):
            fail(f"missing expected output file {path}")
        with open(path) as f:
            out[name] = [json.loads(line) for line in f if line.strip()]
    return out


def main():
    tmp = tempfile.mkdtemp(prefix="tunelab-split-test-")

    # 1. Stratified split: rare class (n=5) conserved; empty-split warning fires
    #    whenever the rare class is absent from a split.
    out_a = os.path.join(tmp, "strat-a")
    r = run(["--input", STRAT, "--outdir", out_a, "--label-key", "label"])
    if r.returncode != 0:
        fail(f"stratified run exited {r.returncode}: {r.stderr}")
    if r.stdout != "":
        fail(f"split_data wrote to stdout (should be stderr-only): {r.stdout!r}")
    recs = read_records(out_a)
    print("PASS: stratified run writes all three split files (train/valid/test.jsonl)")

    total = sum(len(v) for v in recs.values())
    if total != 100:
        fail(f"record conservation violated: 100 in, {total} out")
    rare_counts = {name: sum(1 for rec in v if rec["label"] == "rare") for name, v in recs.items()}
    if sum(rare_counts.values()) != 5:
        fail(f"rare class not conserved: {rare_counts} (expected 5 total)")
    rare_missing = [name for name, c in rare_counts.items() if c == 0]
    if rare_missing and "warning: class 'rare'" not in r.stderr:
        fail(f"rare class absent from {rare_missing} but no warning fired: {r.stderr}")
    print(
        f"PASS: rare class (n=5) conserved across splits {rare_counts}; "
        f"absence from {rare_missing or 'none'} was warned, not silent"
    )

    # 2. Same seed twice -> byte-identical files.
    out_b = os.path.join(tmp, "strat-b")
    r2 = run(["--input", STRAT, "--outdir", out_b, "--label-key", "label"])
    if r2.returncode != 0:
        fail(f"second seeded run exited {r2.returncode}: {r2.stderr}")
    for name in SPLITS:
        with open(os.path.join(out_a, f"{name}.jsonl"), "rb") as f1, \
                open(os.path.join(out_b, f"{name}.jsonl"), "rb") as f2:
            if f1.read() != f2.read():
                fail(f"{name}.jsonl differs between identical seeded runs")
    print("PASS: same seed twice -> byte-identical train/valid/test.jsonl")

    # 3. Ratio validation.
    r = run(["--input", STRAT, "--outdir", os.path.join(tmp, "bad1"), "--ratios", "0.5,0.2,0.2"])
    if r.returncode == 0:
        fail("--ratios summing to 0.9 was accepted")
    print(f"PASS: --ratios not summing to 1.0 -> non-zero exit ({r.returncode})")

    r = run(["--input", STRAT, "--outdir", os.path.join(tmp, "bad2"), "--ratios", "-0.1,1.0,0.1"])
    if r.returncode == 0:
        fail("negative ratio was accepted")
    print(f"PASS: negative ratio -> non-zero exit ({r.returncode})")

    r = run(["--input", STRAT, "--outdir", os.path.join(tmp, "bad3"), "--ratios", "a,b,c"])
    if r.returncode == 0:
        fail("non-numeric ratios were accepted")
    if "Traceback" in r.stderr:
        fail(f"non-numeric ratios crashed with a traceback: {r.stderr}")
    print(f"PASS: non-numeric ratios -> clean non-zero exit ({r.returncode}), no traceback")

    # 4. --label-key and --label-from-assistant are mutually exclusive.
    r = run([
        "--input", STRAT, "--outdir", os.path.join(tmp, "bad4"),
        "--label-key", "label", "--label-from-assistant",
    ])
    if r.returncode != 2:
        fail(f"passing both stratification flags should exit 2 (argparse), got {r.returncode}")
    print("PASS: --label-key + --label-from-assistant together -> argparse error (exit 2)")

    # 5. Unstratified empty splits warn (test.jsonl is the tune-eval contract).
    out_c = os.path.join(tmp, "tiny")
    r = run(["--input", TINY, "--outdir", out_c, "--ratios", "0.98,0.01,0.01"])
    if r.returncode != 0:
        fail(f"tiny unstratified run exited {r.returncode}: {r.stderr}")
    recs = read_records(out_c)
    if len(recs["train"]) != 10 or recs["valid"] or recs["test"]:
        fail(f"expected 10/0/0 split, got { {k: len(v) for k, v in recs.items()} }")
    if "warning: empty split(s)" not in r.stderr:
        fail(f"empty valid+test splits produced no warning: {r.stderr}")
    print("PASS: unstratified run with empty valid/test prints empty-split warning (exit 0)")


if __name__ == "__main__":
    main()
