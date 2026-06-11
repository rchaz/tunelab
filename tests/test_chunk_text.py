#!/usr/bin/env python3
"""Evidence tests for chunk_text.py — runs the real script via subprocess.

Uses small token budgets (--target 100 / --max 200 / --min 10) so the
committed fixtures stay hand-readable: corpus/doc.md (headings+paragraphs),
corpus/huge.txt (one >max paragraph of indented lines), corpus/tiny.txt
(sub-min), corpus/skip.rst (must be ignored by recursion).
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = "/Users/rc/code/tunelab"
SCRIPT = os.path.join(ROOT, "skills", "tune-data", "scripts", "chunk_text.py")
CORPUS_REL = os.path.join("tests", "fixtures", "dataprep", "corpus")

TARGET, MAX, MIN = 100, 200, 10  # tokens; chars = tokens*4


def ok(name, cond, detail=""):
    if not cond:
        print(f"FAIL: {name} -- {detail}", file=sys.stderr)
        sys.exit(1)
    print(f"PASS: {name}")


def run(args):
    # cwd=ROOT so the relative "source" paths in records are stable.
    return subprocess.run(
        ["python3", SCRIPT] + args, capture_output=True, text=True, cwd=ROOT
    )


def main():
    td = tempfile.mkdtemp(prefix="chunktest_")
    out1 = os.path.join(td, "chunks1.jsonl")
    out2 = os.path.join(td, "chunks2.jsonl")
    flags = [
        "--target-tokens", str(TARGET),
        "--max-tokens", str(MAX),
        "--min-tokens", str(MIN),
    ]

    r = run(["--input", CORPUS_REL, "--output", out1] + flags)
    ok("chunk_text on fixture dir exits 0", r.returncode == 0, r.stderr)

    with open(out1) as f:
        recs = [json.loads(line) for line in f]
    ok("output is non-empty JSONL", len(recs) > 0)
    ok(
        "every record has text and source keys",
        all("text" in r_ and "source" in r_ for r_ in recs),
    )
    ok(
        "source is the relative input path",
        all(r_["source"].startswith(CORPUS_REL) for r_ in recs),
    )
    ok(
        "all chunks within [min,max] char bounds",
        all(MIN * 4 <= len(r_["text"]) <= MAX * 4 for r_ in recs),
        str(sorted(len(r_["text"]) for r_ in recs)),
    )

    huge = [r_ for r_ in recs if r_["source"].endswith("huge.txt")]
    ok("huge single paragraph was hard-split into multiple chunks", len(huge) >= 2, str(len(huge)))

    fixture_words = open(os.path.join(ROOT, CORPUS_REL, "huge.txt")).read().split()
    chunk_words = " ".join(r_["text"] for r_ in huge).split()
    ok(
        "hard-split conserved every word of huge.txt",
        sorted(fixture_words) == sorted(chunk_words),
        f"{len(fixture_words)} fixture words vs {len(chunk_words)} chunk words",
    )
    ok(
        "hard-split preserved leading indentation on every line",
        all(
            line.startswith("    ")
            for r_ in huge
            for line in r_["text"].splitlines()
            if line.strip()
        ),
    )

    ok(
        "tiny sub-min file dropped",
        not any(r_["source"].endswith("tiny.txt") for r_ in recs)
        and "dropped" in r.stderr,
        r.stderr,
    )
    ok(
        "directory recursion skipped the .rst file",
        not any(r_["source"].endswith(".rst") for r_ in recs),
    )
    ok(
        "per-file counts and token distribution on stderr",
        "doc.md" in r.stderr and "total:" in r.stderr and "median" in r.stderr,
        r.stderr,
    )
    ok("nothing on stdout (results go to files)", r.stdout == "", r.stdout)

    r2 = run(["--input", CORPUS_REL, "--output", out2] + flags)
    ok(
        "deterministic: two runs byte-identical",
        r2.returncode == 0
        and open(out1, "rb").read() == open(out2, "rb").read(),
    )

    r3 = run(["--input", os.path.join(td, "no_such_dir"), "--output", os.path.join(td, "x.jsonl")])
    ok("missing input path exits non-zero", r3.returncode != 0, str(r3.returncode))

    print("ALL CHECKS PASSED (test_chunk_text)")


if __name__ == "__main__":
    main()
