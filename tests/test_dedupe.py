#!/usr/bin/env python3
"""Evidence tests for skills/tune-data/scripts/dedupe.py.

Invokes the real script via subprocess. Prints one 'PASS: <check>' line per
check; exits non-zero on the first failure.
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "skills", "tune-data", "scripts", "dedupe.py")
FIX = os.path.join(ROOT, "tests", "fixtures", "hygiene")
BASIC = os.path.join(FIX, "dedupe_basic.jsonl")
TEMPLATED = os.path.join(FIX, "dedupe_templated.jsonl")


def run(args, hashseed=None):
    env = dict(os.environ)
    if hashseed is not None:
        env["PYTHONHASHSEED"] = str(hashseed)
    return subprocess.run(
        ["python3", SCRIPT] + args, capture_output=True, text=True, env=env
    )


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def read_lines(path):
    with open(path) as f:
        return [line for line in f if line.strip()]


def main():
    tmp = tempfile.mkdtemp(prefix="tunelab-dedupe-test-")

    # 1. Determinism: same input, different process hash salts -> byte-identical output.
    for name, fixture in (("basic", BASIC), ("templated", TEMPLATED)):
        out1 = os.path.join(tmp, f"det1-{name}.jsonl")
        out2 = os.path.join(tmp, f"det2-{name}.jsonl")
        r1 = run(["--input", fixture, "--output", out1], hashseed=1)
        r2 = run(["--input", fixture, "--output", out2], hashseed=999)
        if r1.returncode != 0 or r2.returncode != 0:
            fail(f"dedupe exited non-zero on {name} fixture: {r1.stderr} {r2.stderr}")
        with open(out1, "rb") as f1, open(out2, "rb") as f2:
            b1, b2 = f1.read(), f2.read()
        if b1 != b2:
            fail(f"output differs across PYTHONHASHSEED=1/999 on {name} fixture")
        if r1.stderr.replace(out1, "OUT") != r2.stderr.replace(out2, "OUT"):
            fail(f"stderr differs across PYTHONHASHSEED=1/999 on {name} fixture")
        print(f"PASS: determinism — byte-identical output and stderr across PYTHONHASHSEED=1/999 ({name} fixture)")

    # 2. Exact dup, near-dup, distinct survivors, first-occurrence-wins.
    out = os.path.join(tmp, "basic-out.jsonl")
    r = run(["--input", BASIC, "--output", out], hashseed=1)
    if r.returncode != 0:
        fail(f"dedupe exited {r.returncode} on basic fixture: {r.stderr}")
    if r.stdout != "":
        fail(f"dedupe wrote to stdout (should be stderr-only): {r.stdout!r}")
    kept = read_lines(out)
    kept_texts = " ".join(kept)

    if len(kept) != 5:
        fail(f"expected 5 survivors from 7 basic records, got {len(kept)}: {kept}")
    if "1 exact dups" not in r.stderr:
        fail(f"expected '1 exact dups' in stderr summary, got: {r.stderr}")
    if kept_texts.count("summarize the") != 1:
        fail("case/whitespace exact-dup variant not collapsed to one record")
    print("PASS: exact duplicate (case/whitespace variant) removed by normalized-hash pass")

    if "1 near-dups" not in r.stderr:
        fail(f"expected '1 near-dups' in stderr summary, got: {r.stderr}")
    if "overseas" in kept_texts and "abroad" in kept_texts:
        fail("one-word-edit near-dup pair not collapsed at default threshold 0.80")
    print("PASS: near-dup (one-word edit on 16-word record) removed at default threshold 0.80")

    if "abroad" not in kept_texts:
        fail("first-occurrence-wins violated: earlier 'abroad' record was dropped")
    if "kept:" not in r.stderr or "dropped:" not in r.stderr:
        fail(f"sample near-dup pair not printed to stderr: {r.stderr}")
    print("PASS: first occurrence wins (earlier 'abroad' record kept, 'overseas' dropped, sample pair printed)")

    if "sourdough" not in kept_texts:
        fail("clearly distinct record did not survive")
    print("PASS: clearly distinct records survive")

    if "drill battery" not in kept_texts or "claw hammers" not in kept_texts:
        fail("records sharing a system prompt but differing in user/assistant content were merged")
    print("PASS: shared system prompt with different user/assistant content NOT merged (system exclusion)")

    # 3. Templated fixture: collapse warning at 0.80; --threshold 0.95 retains all 30.
    out_t = os.path.join(tmp, "templated-080.jsonl")
    r = run(["--input", TEMPLATED, "--output", out_t], hashseed=1)
    if r.returncode != 0:
        fail(f"dedupe exited {r.returncode} on templated fixture: {r.stderr}")
    if "NOTE:" not in r.stderr or "near-dups" not in r.stderr or "0.95" not in r.stderr:
        fail(f"templated-collapse NOTE (with 0.95 suggestion) not printed: {r.stderr}")
    if len(read_lines(out_t)) >= 30:
        fail("templated records were not collapsed at default threshold 0.80")
    print("PASS: templated 30-record fixture triggers collapse NOTE (suggests 0.95) at default 0.80")

    out_t95 = os.path.join(tmp, "templated-095.jsonl")
    r = run(["--input", TEMPLATED, "--output", out_t95, "--threshold", "0.95"], hashseed=1)
    if r.returncode != 0:
        fail(f"dedupe exited {r.returncode} at --threshold 0.95: {r.stderr}")
    n95 = len(read_lines(out_t95))
    if n95 != 30:
        fail(f"--threshold 0.95 should retain all 30 templated records, got {n95}")
    print("PASS: --threshold 0.95 retains all 30 templated records")

    # Output must be valid JSONL.
    for line in kept:
        json.loads(line)
    print("PASS: output is valid JSONL")


if __name__ == "__main__":
    main()
