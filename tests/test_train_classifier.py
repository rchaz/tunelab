#!/usr/bin/env python3
"""Evidence tests for skills/tune-decide/scripts/train_classifier.py.

Invokes the real script via `uv run` subprocess. Uses the real CFPB data at
dogfood/level1/data/raw.jsonl (10 classes, 300/class) for the accuracy checks
and committed fixtures for the failure-path checks. First run downloads the
~125MB local embedding model from Hugging Face (cached afterwards).

Prints one 'PASS: <check>' line per check; exits non-zero on the first failure.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "skills", "tune-decide", "scripts", "train_classifier.py")
RAW = os.path.join(ROOT, "dogfood", "level1", "data", "raw.jsonl")
FIX = os.path.join(ROOT, "tests", "fixtures", "embeddings")
SMALL = os.path.join(FIX, "labeled_small.jsonl")
TINY = os.path.join(FIX, "labeled_tiny.jsonl")
LOCAL_MODEL = "sentence-transformers/static-retrieval-mrl-en-v1"


def run(args, timeout=600):
    return subprocess.run(
        ["uv", "run", SCRIPT] + args, capture_output=True, text=True, timeout=timeout
    )


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    tmp = tempfile.mkdtemp(prefix="tunelab-trainclf-test-")

    # Build a 1000-row stratified subset (100/class) + 20 held-out rows (2/class)
    # from the real CFPB data.
    by_label = defaultdict(list)
    with open(RAW) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                by_label[rec["label"]].append(rec)
    labels = sorted(by_label)
    if len(labels) != 10:
        fail(f"expected 10 CFPB classes, found {len(labels)}")
    train_rows = [r for lb in labels for r in by_label[lb][:100]]
    heldout_rows = [r for lb in labels for r in by_label[lb][100:102]]
    train_path = os.path.join(tmp, "train1000.jsonl")
    heldout_path = os.path.join(tmp, "heldout20.jsonl")
    write_jsonl(train_path, train_rows)
    write_jsonl(heldout_path, heldout_rows)

    # 1. Train on 1000 rows, local backend: exit 0, honest holdout metrics,
    #    accuracy >= 0.55 (benchmark ~0.70+ at this size; 0.55 = regression floor).
    bundle = os.path.join(tmp, "clf.joblib")
    r1 = run(["--data", train_path, "--model-out", bundle])
    if r1.returncode != 0:
        fail(f"train run exited {r1.returncode}: {r1.stderr}")
    m_acc = re.search(r"held-out accuracy: ([0-9.]+) \(n=(\d+)\)", r1.stdout)
    m_f1 = re.search(r"held-out macro-F1: ([0-9.]+)", r1.stdout)
    if not m_acc:
        fail(f"no held-out accuracy line on stdout: {r1.stdout}")
    if not m_f1:
        fail(f"no held-out macro-F1 line on stdout: {r1.stdout}")
    acc = float(m_acc.group(1))
    if acc < 0.55:
        fail(f"held-out accuracy {acc} below the 0.55 regression floor")
    if "classifier: LogisticRegression" not in r1.stderr:
        fail(f"missing one-line WHY for the LR default on stderr: {r1.stderr}")
    if "saved ->" in r1.stdout:
        fail("'saved ->' diagnostic went to stdout (convention: stderr)")
    if "saved ->" not in r1.stderr:
        fail(f"missing 'saved ->' line on stderr: {r1.stderr}")
    print(
        f"PASS: train on 1000 CFPB rows (local backend) exits 0; "
        f"held-out accuracy {acc} >= 0.55 and macro-F1 {m_f1.group(1)} printed"
    )

    # 2. Determinism: identical command (same default seed) -> identical metrics.
    r2 = run(["--data", train_path, "--model-out", os.path.join(tmp, "clf2.joblib")])
    if r2.returncode != 0:
        fail(f"second seeded train run exited {r2.returncode}: {r2.stderr}")
    acc_line_1 = [l for l in r1.stdout.splitlines() if "held-out accuracy" in l]
    acc_line_2 = [l for l in r2.stdout.splitlines() if "held-out accuracy" in l]
    f1_line_1 = [l for l in r1.stdout.splitlines() if "macro-F1" in l]
    f1_line_2 = [l for l in r2.stdout.splitlines() if "macro-F1" in l]
    if acc_line_1 != acc_line_2 or f1_line_1 != f1_line_2:
        fail(f"same seed, different metrics: {acc_line_1} vs {acc_line_2}, {f1_line_1} vs {f1_line_2}")
    print(f"PASS: same train command twice (same seed) -> identical held-out accuracy line {acc_line_1[0]!r}")

    # 3. Predict on 20 held-out rows: every row gets predicted + confidence in [0,1];
    #    the bundle round-trips backend/embed_model (stderr echo + conflict refusals).
    preds_path = os.path.join(tmp, "preds.jsonl")
    rp = run(["--predict", heldout_path, "--model-in", bundle, "--output", preds_path])
    if rp.returncode != 0:
        fail(f"predict run exited {rp.returncode}: {rp.stderr}")
    if rp.stdout != "":
        fail(f"predict mode wrote to stdout (diagnostics belong on stderr): {rp.stdout!r}")
    if f"embedding with local:{LOCAL_MODEL}" not in rp.stderr:
        fail(f"bundle backend/embed_model not echoed from the bundle: {rp.stderr}")
    if "wrote 20 predictions" not in rp.stderr:
        fail(f"missing 'wrote 20 predictions' on stderr: {rp.stderr}")
    with open(preds_path) as f:
        preds = [json.loads(line) for line in f if line.strip()]
    if len(preds) != 20:
        fail(f"expected 20 prediction rows, got {len(preds)}")
    for i, p in enumerate(preds):
        if p.get("predicted") not in labels:
            fail(f"prediction row {i} has predicted={p.get('predicted')!r}, not a training label")
        c = p.get("confidence")
        if not isinstance(c, (int, float)) or not (0.0 <= c <= 1.0):
            fail(f"prediction row {i} confidence {c!r} not in [0,1]")
    print("PASS: predict on 20 held-out rows -> 20 rows, predicted in label set, confidence in [0,1]")

    rb = run(["--predict", heldout_path, "--model-in", bundle, "--output", preds_path, "--backend", "openai"])
    if rb.returncode == 0 or "backend='local'" not in rb.stderr:
        fail(f"conflicting --backend openai not refused with the bundle's backend: {rb.returncode} {rb.stderr}")
    rm = run(["--predict", heldout_path, "--model-in", bundle, "--output", preds_path,
              "--embed-model", "minishlab/potion-base-32M"])
    if rm.returncode == 0 or LOCAL_MODEL not in rm.stderr:
        fail(f"conflicting --embed-model not refused with the bundle's model: {rm.returncode} {rm.stderr}")
    print("PASS: bundle round-trips backend + embed_model (echoed on stderr; conflicting flags refused)")

    # 4. Major finding: missing xgboost exits with the rerun command BEFORE any
    #    embedding pass (auto + --extra-keys resolves to xgboost).
    rx = run(["--data", SMALL, "--extra-keys", "amount", "--model-out", os.path.join(tmp, "x.joblib")])
    if rx.returncode == 0:
        fail("auto+extra-keys (xgboost) succeeded without xgboost installed?")
    if "uv run --with xgboost" not in rx.stderr:
        fail(f"missing exact rerun command in: {rx.stderr}")
    if "embedded" in rx.stderr or "loading" in rx.stderr:
        fail(f"embedding ran before the xgboost availability exit: {rx.stderr}")
    print("PASS: missing xgboost -> 'uv run --with xgboost' exit BEFORE any embedding (no 'loading'/'embedded' on stderr)")

    # 5. The printed rerun command works verbatim (xgboost path end-to-end).
    cmd = None
    for line in rx.stderr.splitlines():
        if line.strip().startswith("uv run --with xgboost"):
            cmd = line.strip()
            break
    if cmd is None:
        fail(f"could not extract rerun command from: {rx.stderr}")
    rxx = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    if rxx.returncode != 0:
        fail(f"verbatim rerun command failed ({rxx.returncode}): {rxx.stderr}")
    if "classifier: XGBoost" not in rxx.stderr or "held-out macro-F1" not in rxx.stdout:
        fail(f"xgboost rerun missing WHY line or metrics: {rxx.stderr} / {rxx.stdout}")
    print("PASS: printed rerun command works verbatim (xgboost trains, WHY + macro-F1 printed)")

    # 6. --data and --predict are mutually exclusive (argparse exit 2).
    rme = run(["--data", SMALL, "--predict", heldout_path, "--model-in", bundle, "--output", preds_path])
    if rme.returncode != 2:
        fail(f"--data + --predict together should exit 2 (argparse), got {rme.returncode}: {rme.stderr}")
    print("PASS: --data + --predict together -> argparse error (exit 2), no silent train-only run")

    # 7. Tiny dataset -> clean plain-language exit, no sklearn traceback.
    rt = run(["--data", TINY, "--model-out", os.path.join(tmp, "t.joblib")])
    if rt.returncode == 0:
        fail("4-record dataset was accepted for a 20% stratified holdout")
    if "Traceback" in rt.stderr:
        fail(f"tiny dataset crashed with a traceback: {rt.stderr}")
    if "Level 1 needs at least" not in rt.stderr:
        fail(f"tiny dataset exit lacks the plain-language minimum-data message: {rt.stderr}")
    print(f"PASS: 4-record dataset -> clean exit ({rt.returncode}) with minimum-data message, no traceback")

    # 8. Constant extra-keys column: warned, scaled to zero, still trains (lr).
    flag_bundle = os.path.join(tmp, "flag.joblib")
    rc = run(["--data", SMALL, "--extra-keys", "flag", "--classifier", "lr", "--model-out", flag_bundle])
    if rc.returncode != 0:
        fail(f"constant-column train exited {rc.returncode}: {rc.stderr}")
    if "constant" not in rc.stderr:
        fail(f"no constant-column warning on stderr: {rc.stderr}")
    print("PASS: constant extra-keys column -> stderr warning, training still succeeds")

    # 9. Predict-time flag conflicts against the bundle are hard errors:
    #    --text-key, --classifier, --extra-keys (plus backend/model covered above).
    rtk = run(["--predict", SMALL, "--model-in", flag_bundle, "--output", preds_path,
               "--text-key", "narrative"])
    if rtk.returncode == 0 or "trained on field 'text'" not in rtk.stderr:
        fail(f"--text-key conflict not refused: {rtk.returncode} {rtk.stderr}")
    rcc = run(["--predict", SMALL, "--model-in", flag_bundle, "--output", preds_path,
               "--classifier", "xgboost"])
    if rcc.returncode == 0 or "lr model" not in rcc.stderr:
        fail(f"--classifier conflict not refused: {rcc.returncode} {rcc.stderr}")
    rek = run(["--predict", SMALL, "--model-in", flag_bundle, "--output", preds_path,
               "--extra-keys", "amount"])
    if rek.returncode == 0 or "extra-keys" not in rek.stderr:
        fail(f"--extra-keys conflict not refused: {rek.returncode} {rek.stderr}")
    print("PASS: predict refuses conflicting --text-key / --classifier / --extra-keys against the bundle")

    # 10. Empty predict file and missing text field -> clean indexed errors, no traceback.
    empty = os.path.join(tmp, "empty.jsonl")
    open(empty, "w").close()
    re_ = run(["--predict", empty, "--model-in", flag_bundle, "--output", preds_path])
    if re_.returncode == 0 or "Traceback" in re_.stderr or "no records" not in re_.stderr:
        fail(f"empty predict file not handled cleanly: {re_.returncode} {re_.stderr}")
    badrec = os.path.join(tmp, "badrec.jsonl")
    write_jsonl(badrec, [{"text": "hello there", "flag": 0}, {"flag": 0}])
    rbad = run(["--predict", badrec, "--model-in", flag_bundle, "--output", preds_path])
    if rbad.returncode == 0 or "Traceback" in rbad.stderr:
        fail(f"missing text field crashed or passed: {rbad.returncode} {rbad.stderr}")
    if "--predict record 1 is missing field 'text'" not in rbad.stderr:
        fail(f"missing text field lacks the indexed clean error: {rbad.stderr}")
    print("PASS: empty predict file and missing-text record -> clean indexed errors, no traceback")


if __name__ == "__main__":
    main()
