#!/usr/bin/env python3
"""Evidence tests for skills/tune-decide/scripts/centroid_classify.py.

Invokes the real script via `uv run` subprocess. Uses the real CFPB data at
dogfood/level1/data/raw.jsonl (3-class subset, 20 examples/class) for the main
check and temp files for the failure paths. First run downloads the ~125MB
local embedding model from Hugging Face (cached afterwards).

Prints one 'PASS: <check>' line per check; exits non-zero on the first failure.
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "skills", "tune-decide", "scripts", "centroid_classify.py")
RAW = os.path.join(ROOT, "dogfood", "level1", "data", "raw.jsonl")
CLASSES = ["credit_card", "mortgage", "student_loan"]


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
    tmp = tempfile.mkdtemp(prefix="tunelab-centroid-test-")

    by_label = defaultdict(list)
    with open(RAW) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["label"] in CLASSES:
                    by_label[rec["label"]].append(rec)

    # 20 examples/class for centroids; 30 inputs (10/class) drawn from later
    # rows, label stripped (kept as 'gold' to verify key pass-through).
    examples = [r for lb in CLASSES for r in by_label[lb][:20]]
    inputs = [
        {"text": r["text"], "gold": r["label"]}
        for lb in CLASSES
        for r in by_label[lb][20:30]
    ]
    ex_path = os.path.join(tmp, "examples.jsonl")
    in_path = os.path.join(tmp, "inputs.jsonl")
    out_path = os.path.join(tmp, "preds.jsonl")
    write_jsonl(ex_path, examples)
    write_jsonl(in_path, inputs)

    # 1. Main check: 3 classes x 20 examples, classify 30 inputs, local backend.
    r = run(["--examples", ex_path, "--classify", in_path, "--output", out_path])
    if r.returncode != 0:
        fail(f"centroid run exited {r.returncode}: {r.stderr}")
    if r.stdout != "":
        fail(f"centroid_classify wrote to stdout (results belong in --output): {r.stdout!r}")
    with open(out_path) as f:
        preds = [json.loads(line) for line in f if line.strip()]
    if len(preds) != 30:
        fail(f"expected 30 prediction rows, got {len(preds)}")
    for i, p in enumerate(preds):
        if p.get("predicted") not in CLASSES:
            fail(f"row {i}: predicted={p.get('predicted')!r} not one of the 3 example labels")
        c = p.get("confidence")
        if not isinstance(c, (int, float)) or c < 0:
            fail(f"row {i}: confidence (cosine margin) {c!r} is not >= 0")
        if "gold" not in p:
            fail(f"row {i}: extra input key 'gold' was not passed through")
    print(
        "PASS: centroid_classify (3 classes x 20 examples, 30 inputs) exits 0; "
        "all predictions from the 3 labels; confidence = margin >= 0; extra keys preserved"
    )
    if "confidence margins:" not in r.stderr or "predicted distribution:" not in r.stderr:
        fail(f"margin/distribution diagnostics missing from stderr: {r.stderr}")
    print("PASS: margin quartiles + predicted distribution diagnostics on stderr")

    # 2. Empty --classify file -> clean exit, no model2vec traceback.
    empty = os.path.join(tmp, "empty.jsonl")
    open(empty, "w").close()
    re_ = run(["--examples", ex_path, "--classify", empty, "--output", out_path])
    if re_.returncode == 0 or "Traceback" in re_.stderr or "no records" not in re_.stderr:
        fail(f"empty --classify not handled cleanly: {re_.returncode} {re_.stderr}")
    if "loading" in re_.stderr:
        fail(f"empty --classify still loaded the embedding model: {re_.stderr}")
    print("PASS: empty --classify file -> clean 'no records' exit before any embedding, no traceback")

    # 3. Missing fields -> indexed clean errors, no traceback, before embedding.
    bad_ex = os.path.join(tmp, "bad_examples.jsonl")
    write_jsonl(bad_ex, [{"text": "a credit card complaint", "label": "credit_card"}, {"text": "no label here"}])
    rb = run(["--examples", bad_ex, "--classify", in_path, "--output", out_path])
    if rb.returncode == 0 or "Traceback" in rb.stderr:
        fail(f"missing 'label' crashed or passed: {rb.returncode} {rb.stderr}")
    if "--examples record 1 is missing field 'label'" not in rb.stderr:
        fail(f"missing 'label' lacks the indexed clean error: {rb.stderr}")
    bad_in = os.path.join(tmp, "bad_inputs.jsonl")
    write_jsonl(bad_in, [{"narrative": "wrong key"}])
    ri = run(["--examples", ex_path, "--classify", bad_in, "--output", out_path])
    if ri.returncode == 0 or "Traceback" in ri.stderr:
        fail(f"missing text key crashed or passed: {ri.returncode} {ri.stderr}")
    if "--classify record 0 is missing field 'text'" not in ri.stderr:
        fail(f"missing text key lacks the indexed clean error: {ri.stderr}")
    if "loading" in rb.stderr or "loading" in ri.stderr:
        fail("field validation ran after the embedding model load (should fail fast)")
    print("PASS: missing 'label'/text fields -> indexed clean errors before any embedding, no traceback")


if __name__ == "__main__":
    main()
