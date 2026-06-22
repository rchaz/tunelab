#!/usr/bin/env python3
"""Evidence tests for skills/tune-eval/scripts/eval_classifier.py (stdlib, real subprocess).

Fixture preds_known.jsonl (20 records, 3 gold classes, controlled errors):
  expected a x8: 6 predicted a (one as 'A' — case norm), 1 -> b, 1 -> 'zzz-junk' (hallucinated)
  expected b x6: 5 predicted b, 1 -> a
  expected c x6: 6 predicted c (one as ' c ' — whitespace norm)
Known truth: accuracy 17/20 = 0.850; gold-only macro-F1 = (0.800 + 0.833 + 1.000)/3 = 0.878
Confusion rows (cols a, b, c, zzz-junk): a=[6,1,0,1]  b=[1,5,0,0]  c=[0,0,6,0]
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "skills", "tune-eval", "scripts", "eval_classifier.py")
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "evallocal")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name} {detail}", file=sys.stderr)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, "report.md")
        r = subprocess.run(
            ["python3", SCRIPT, "--predictions", os.path.join(FIXTURES, "preds_known.jsonl"),
             "--report", report],
            capture_output=True, text=True)
        out = r.stdout
        check("eval_classifier exits 0 on known fixture", r.returncode == 0, r.stderr)
        check("summary line has n and class count", "n = 20    classes = 3" in out, out.splitlines()[:1])
        check("exact accuracy 0.850 in output", "accuracy = 0.850" in out)
        check("exact macro-F1 0.878 in output (gold-classes-only)", "macro-F1 = 0.878" in out)

        # Per-class table values (a/b/c) and the starred hallucinated row.
        check("per-class a line prec/rec/f1/support",
              any(l.split() == ["a", "0.857", "0.750", "0.800", "8"] for l in out.splitlines()), out)
        check("per-class b line prec/rec/f1/support",
              any(l.split() == ["b", "0.833", "0.833", "0.833", "6"] for l in out.splitlines()), out)
        check("per-class c line prec/rec/f1/support",
              any(l.split() == ["c", "1.000", "1.000", "1.000", "6"] for l in out.splitlines()), out)
        check("hallucinated label starred in per-class table",
              any(l.split() == ["zzz-junk", "0.000", "0.000", "0.000", "0", "*"] for l in out.splitlines()), out)

        # Confusion matrix: locate the caption, parse the three gold rows.
        lines = out.splitlines()
        cap = next(i for i, l in enumerate(lines) if l.startswith("confusion matrix"))
        header = lines[cap + 2].split()
        check("matrix columns include hallucinated label", header == ["a", "b", "c", "zzz-junk"], header)
        rows = {}
        j = cap + 3
        while j < len(lines) and lines[j].strip():
            toks = lines[j].split()
            rows[toks[0]] = [int(t) for t in toks[1:]]
            j += 1
        check("matrix row a counts [6,1,0,1]", rows.get("a") == [6, 1, 0, 1], rows)
        check("matrix row b counts [1,5,0,0]", rows.get("b") == [1, 5, 0, 0], rows)
        check("matrix row c counts [0,0,6,0]", rows.get("c") == [0, 0, 6, 0], rows)
        check("matrix rows are gold classes only (no zzz-junk row)",
              sorted(rows) == ["a", "b", "c"], sorted(rows))

        check("hallucinated label flagged in WARNING block",
              "WARNING: 1 hallucinated label(s)" in out and "'zzz-junk' (x1)" in out, out)
        check("sklearn-convention note printed when hallucinated labels exist",
              "macro-F1 averages gold classes only" in out and "union-of-labels" in out, out)

        with open(report) as f:
            rep = f.read()
        check("report file contains the same metrics text",
              "accuracy = 0.850" in rep and "macro-F1 = 0.878" in rep and rep.startswith("# Classification evaluation"))

        # Malformed line (missing 'predicted' on line 2): clean exit with line number.
        r2 = subprocess.run(
            ["python3", SCRIPT, "--predictions", os.path.join(FIXTURES, "preds_bad.jsonl")],
            capture_output=True, text=True)
        check("bad line exits non-zero with file:line and error type",
              r2.returncode != 0 and ":2:" in r2.stderr and "KeyError" in r2.stderr, r2.stderr)
        check("no traceback on bad line", "Traceback" not in r2.stderr, r2.stderr)

        # Gold may be named 'label' (classifier passthrough / raw datasets), not
        # only 'expected'; 'predicted' is still required.
        labeled = os.path.join(tmp, "preds_label.jsonl")
        with open(labeled, "w") as f:
            for e, p in [("a", "a"), ("a", "a"), ("b", "b"), ("b", "a")]:
                f.write('{"label": "%s", "predicted": "%s"}\n' % (e, p))
        r3 = subprocess.run(["python3", SCRIPT, "--predictions", labeled],
                            capture_output=True, text=True)
        check("'label' accepted as the gold key (accuracy 0.750)",
              r3.returncode == 0 and "accuracy = 0.750" in r3.stdout,
              r3.stdout + r3.stderr)

    if failures:
        sys.exit(f"{len(failures)} check(s) failed: {failures}")


if __name__ == "__main__":
    main()
