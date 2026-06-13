#!/usr/bin/env python3
"""Mechanical grounding gate for compression / extraction tasks. Stdlib only.

A compressor that INVENTS a value is worse than no compressor: downstream
consumers can't tell the corruption from the truth. This gate catches it by
string matching, not vibes — and it runs identically on the teacher (filtering
training data) and the student (scoring eval), which is the whole point.

For each (source_blob, output) pair it checks:

  atomic grounding   every "hard" token in the output — numbers, identifiers,
                     paths, codes, hex, versions — must appear VERBATIM in the
                     source. New connective English is fine; new or altered
                     VALUES are failures. Reformatting counts as failure
                     ("5.54 kB" -> "5.54kB"), and composed tokens are split on
                     separators before matching ("ClassName.method" must ground
                     as ClassName AND method).
  length budget      output_chars / input_chars <= --max-ratio (default 0.40);
                     the compression has to actually compress.

Reports, per the distiller metric card:
  hallucinated-value rate  = fraction of outputs with >=1 ungrounded hard token
  zero-hallucination rate  = 1 - the above (the pre-registered bar: >= 0.99)
  ratio p50/p90/max        = compression achieved
  over-budget count        = outputs above --max-ratio

  uv run grounding_gate.py --pairs preds_tuned.jsonl \
    --source-key input --output-key predicted --max-ratio 0.40 \
    --report eval/gate_tuned.md --flagged eval/gate_flagged.jsonl

  uv run grounding_gate.py --self-test

  pairs lines: {<source-key>: "raw blob...", <output-key>: "compressed..."}
  For chat preds from run_test_set.py: --source-key input --output-key predicted
  (input is the user blob; expected/teacher is ignored here).
"""

import argparse
import json
import re
import sys


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# A "hard" token is anything that carries data a consumer might reuse verbatim:
# contains a digit, OR looks like an identifier/path/code (has _ / . - : or
# mixed case / all-caps runs). Plain lowercase English words are NOT hard.
HARD_RE = re.compile(r"[A-Za-z0-9_./:\-]*\d[A-Za-z0-9_./:\-]*"      # has a digit
                     r"|[A-Za-z][A-Za-z0-9]*[._/:\-][A-Za-z0-9._/:\-]+"  # has a separator
                     r"|[A-Z]{2,}[A-Za-z0-9]*")                     # ALLCAPS / CamelCASE run
SPLIT_RE = re.compile(r"[\s,;()\[\]{}<>\"'`|=]+")
SUBSPLIT_RE = re.compile(r"[._/:\-]+")


def normalize_source(blob: str) -> str:
    # match verbatim but case-foldable; keep all chars so "5.54 kB" != "5.54kB"
    return blob.lower()


def hard_tokens(text: str):
    """Yield hard tokens AND their separator-split atoms (composed-token rule)."""
    for raw in SPLIT_RE.split(text):
        raw = raw.strip(".,:;)(")
        if not raw:
            continue
        if HARD_RE.fullmatch(raw):
            yield raw
            atoms = [a for a in SUBSPLIT_RE.split(raw) if a]
            if len(atoms) > 1:
                for a in atoms:
                    # only atoms that are themselves data-bearing need grounding
                    if any(c.isdigit() for c in a) or len(a) >= 2:
                        yield a


def ungrounded(blob: str, output: str):
    src = normalize_source(blob)
    bad = []
    for tok in hard_tokens(output):
        if tok.lower() not in src:
            bad.append(tok)
    return bad


def check_pair(blob, output, max_ratio):
    bad = ungrounded(blob, output)
    ratio = len(output) / max(len(blob), 1)
    return {
        "ungrounded": bad,
        "n_ungrounded": len(bad),
        "ratio": round(ratio, 4),
        "over_budget": ratio > max_ratio,
        "clean": len(bad) == 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs")
    ap.add_argument("--source-key", default="input")
    ap.add_argument("--output-key", default="predicted")
    ap.add_argument("--max-ratio", type=float, default=0.40)
    ap.add_argument("--report", default=None)
    ap.add_argument("--flagged", default=None, help="write flagged records here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.pairs:
        ap.error("--pairs is required (unless --self-test)")

    rows = [json.loads(l) for l in open(args.pairs)]
    results = []
    flagged = []
    for r in rows:
        blob = r.get(args.source_key, "")
        out = r.get(args.output_key, "") or ""
        res = check_pair(blob, out, args.max_ratio)
        results.append(res)
        if not res["clean"] or res["over_budget"]:
            flagged.append({**{k: r.get(k) for k in (args.source_key, args.output_key, "id")},
                            **res})

    n = len(results)
    clean = sum(r["clean"] for r in results)
    over = sum(r["over_budget"] for r in results)
    ratios = sorted(r["ratio"] for r in results)
    p = lambda q: ratios[min(int(q * n), n - 1)] if n else 0.0
    zero_halluc = clean / n if n else 0.0

    lines = [
        f"# grounding gate report ({n} outputs)",
        "",
        f"- **zero-hallucination rate: {zero_halluc:.4f}** (bar >= 0.99: "
        f"{'PASS' if zero_halluc >= 0.99 else 'FAIL'})",
        f"- hallucinated-value rate: {1-zero_halluc:.4f} ({n-clean}/{n} outputs "
        f"with >=1 ungrounded hard token)",
        f"- compression ratio: p50 {p(0.5):.3f} / p90 {p(0.9):.3f} / max {ratios[-1] if n else 0:.3f}"
        f" (bar p50 <= 0.35)",
        f"- over-budget (> {args.max_ratio}): {over}/{n}",
    ]
    report = "\n".join(lines)
    print(report)
    if args.report:
        open(args.report, "w").write(report + "\n")
    if args.flagged and flagged:
        with open(args.flagged, "w") as f:
            for fl in flagged:
                f.write(json.dumps(fl) + "\n")
        eprint(f"wrote {len(flagged)} flagged records -> {args.flagged}")


def self_test():
    blob = ("Written: /Users/rc/run/step5.txt\nstatus: OK (exit 0)\n"
            "size 5.54 kB, ClassName.method took 12.3ms\n")
    cases = [
        ("clean", "step5.txt OK exit 0, 5.54 kB, ClassName.method 12.3ms", True),
        ("reformatted value", "size 5.54kB exit 0", False),       # 5.54kB not verbatim
        ("invented number", "took 99.9ms exit 0", False),         # 99.9ms not in source
        ("composed token ok", "ClassName.method 12.3ms", True),   # both atoms present
        ("english only fine", "the file was written and the run succeeded", True),
    ]
    ok = True
    for name, out, expect_clean in cases:
        res = check_pair(blob, out, 0.40)
        got = res["clean"]
        flag = "ok" if got == expect_clean else "XX MISMATCH"
        if got != expect_clean:
            ok = False
        print(f"  [{flag}] {name}: clean={got} (expected {expect_clean}) "
              f"ungrounded={res['ungrounded']}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
