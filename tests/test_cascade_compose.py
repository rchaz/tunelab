#!/usr/bin/env python3
"""Evidence test for skills/tune-eval/scripts/cascade_compose.py.

Runs the script's built-in --self-test (deterministic, fixture-only, no dataset
or model download) so the composition math AND the gold-key alias (label|expected)
are covered in CI. The composer previously had no test_*.py at all.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "skills", "tune-eval", "scripts", "cascade_compose.py")


def main():
    r = subprocess.run(["uv", "run", SCRIPT, "--self-test"],
                       capture_output=True, text=True, cwd=ROOT)
    out = r.stdout + r.stderr
    ok = (r.returncode == 0
          and "SELF-TEST PASS" in out
          and "gold-key alias label|expected OK" in out)
    if not ok:
        print(f"FAIL: cascade_compose --self-test\nrc={r.returncode}\n{out}", file=sys.stderr)
        sys.exit(1)
    print("PASS: cascade_compose --self-test (composition math + gold-key alias label|expected)")
    print("ALL CHECKS PASSED (test_cascade_compose)")


if __name__ == "__main__":
    main()
