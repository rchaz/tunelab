#!/usr/bin/env python3
"""Tests for skills/tune-eval/scripts/judge_eval.py against the fake anthropic
SDK in tests/shims (no API key, no network). The shim always picks winner
'first', so the a/b tally must match the blind-position distribution implied
by the fixed --seed."""

import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "skills", "tune-eval", "scripts", "judge_eval.py")
SHIMS = os.path.join(ROOT, "tests", "shims")
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "apiscripts")

PREDS_A = os.path.join(FIXTURES, "preds_a.jsonl")
PREDS_B = os.path.join(FIXTURES, "preds_b.jsonl")
SKIP_IDS = {"p05", "p09"}  # REFUSE_ME / BANANA_ME markers in the fixtures
SEED = 42


def base_env(**extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = SHIMS
    env["ANTHROPIC_API_KEY"] = "shim-dummy-key"
    env.pop("SHIM_DEBUG", None)
    env.update(extra)
    return env


def run(args, env=None):
    return subprocess.run(
        ["uv", "run", SCRIPT] + args,
        capture_output=True, text=True, env=env or base_env(), cwd=ROOT,
    )


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def expected_results(ids, skip_ids, seed):
    # Mirror the script exactly: the rnd draw precedes the API call, so the
    # draw happens for every id including skipped ones; shim winner is always
    # 'first', so result is 'b' iff B was shown first.
    rnd = random.Random(seed)
    out = {}
    for _id in sorted(ids, key=str):
        b_first = rnd.random() < 0.5
        if _id in skip_ids:
            continue
        out[_id] = "b" if b_first else "a"
    return out


def main():
    all_ids = [r["id"] for r in read_jsonl(PREDS_A)]
    expected = expected_results(all_ids, SKIP_IDS, SEED)
    exp_b = sum(1 for v in expected.values() if v == "b")
    exp_a = sum(1 for v in expected.values() if v == "a")
    assert exp_a > 0 and exp_b > 0, "seed must exercise both blind orders"

    with tempfile.TemporaryDirectory() as tmp:
        verdicts = os.path.join(tmp, "verdicts.jsonl")
        r = run(["--a", PREDS_A, "--b", PREDS_B,
                 "--criteria", "Faithful to the task; concise",
                 "--output", verdicts, "--seed", str(SEED)],
                env=base_env(SHIM_DEBUG="1"))
        assert r.returncode == 0, r.stderr
        print("PASS: judge run on two 12-item pred files exits 0")

        rows = read_jsonl(verdicts)
        assert len(rows) == 12 - len(SKIP_IDS), (
            f"expected {12 - len(SKIP_IDS)} verdicts, got {len(rows)}")
        print(f"PASS: verdicts file has {len(rows)} lines "
              "(12 items minus 2 skips); skipped ids excluded")

        got = {row["id"]: row["result"] for row in rows}
        assert got == expected, f"\n got: {got}\nwant: {expected}"
        print("PASS: every verdict matches the blind-position mapping for "
              f"seed {SEED} (shim always picks 'first'; "
              f"B-first -> 'b' x{exp_b}, A-first -> 'a' x{exp_a})")

        assert f"B (tuned) wins: {exp_b} (" in r.stdout, r.stdout
        assert f"A (base) wins:  {exp_a} (" in r.stdout, r.stdout
        assert "ties:           0 (0%)" in r.stdout, r.stdout
        print(f"PASS: stdout tally consistent with expected distribution "
              f"(B {exp_b} / ties 0 / A {exp_a})")

        assert "skipped:        2 (refusals/parse errors — details on stderr)" \
            in r.stdout, r.stdout
        print("PASS: skipped count (2) printed in final summary")

        assert "id=p05 skipped (stop_reason=refusal)" in r.stderr, r.stderr
        print("PASS: refusal pair skipped with stderr note "
              "'id=p05 skipped (stop_reason=refusal)'")
        assert "id=p09 skipped (unexpected winner value: 'banana')" in r.stderr, \
            r.stderr
        print("PASS: out-of-enum winner skipped (defense in depth) instead of "
              "being mis-tallied")

        assert ("note: only 10 pairs judged — differences under ~10 points "
                "are noise at this sample size") in r.stdout, r.stdout
        print("PASS: noise warning printed in final summary (n<150)")

        assert "SHIM max_tokens=8192" in r.stderr, r.stderr
        print("PASS: judge requests use max_tokens=8192 (headroom for "
              "adaptive thinking)")

    # --- skipped: 0 is printed even when nothing was skipped ----------------
    with tempfile.TemporaryDirectory() as tmp:
        verdicts = os.path.join(tmp, "verdicts.jsonl")
        r = run(["--a", PREDS_A, "--b", PREDS_B, "--criteria", "c",
                 "--output", verdicts, "--seed", str(SEED), "--limit", "4"])
        assert r.returncode == 0, r.stderr
        assert "skipped:        0" in r.stdout, r.stdout
        assert "refusals/parse errors" not in r.stdout, r.stdout
        assert len(read_jsonl(verdicts)) == 4
        print("PASS: 'skipped:        0' always printed in summary "
              "(no details suffix when zero)")

    # --- all pairs skipped --------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        pa, pb = os.path.join(tmp, "a.jsonl"), os.path.join(tmp, "b.jsonl")
        for path, tag in ((pa, "base"), (pb, "tuned")):
            with open(path, "w", encoding="utf-8") as f:
                for i in (1, 2):
                    f.write(json.dumps({
                        "id": f"x{i}", "input": f"REFUSE_ME item {i}",
                        "expected": "ref", "predicted": f"{tag} {i}",
                    }) + "\n")
        verdicts = os.path.join(tmp, "verdicts.jsonl")
        r = run(["--a", pa, "--b", pb, "--criteria", "c",
                 "--output", verdicts])
        assert r.returncode == 0, r.stderr
        assert "note: no pairs judged — all pairs were skipped" in r.stdout, \
            r.stdout
        assert "B (tuned) wins: 0 (0%)" in r.stdout, r.stdout
        assert read_jsonl(verdicts) == []
        print("PASS: judged==0 prints 'no pairs judged' note instead of the "
              "noise warning; no division crash")

    # --- --provider flag (PLAN §5 symmetry with distill_generate) -----------
    r = run(["--help"])
    assert "--provider" in r.stdout and "anthropic" in r.stdout, r.stdout
    assert "OpenAI lands in Phase 2" in r.stdout, r.stdout
    print("PASS: --provider {anthropic} flag present with Phase-2 help text")

    print("ALL judge_eval CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
