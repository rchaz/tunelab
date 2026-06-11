#!/usr/bin/env python3
"""Evidence tests for skills/tune-eval/scripts/run_test_set.py (real subprocess runs).

Includes a LIVE smoke against mlx-community/Qwen3-0.6B-4bit (0.34GB, real
hybrid-thinking chat template) — first run downloads the checkpoint into
~/.cache/huggingface. Strip semantics are unit-checked by importing the real
module under uv with mlx-lm available.
"""

import json
import os
import subprocess
import sys

SCRIPT = "/Users/rc/code/tunelab/skills/tune-eval/scripts/run_test_set.py"
FIXTURES = "/Users/rc/code/tunelab/tests/fixtures/evallocal"
MODEL = "mlx-community/Qwen3-0.6B-4bit"

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name} {detail}", file=sys.stderr)


STRIP_CODE = """
import sys
sys.path.insert(0, "/Users/rc/code/tunelab/skills/tune-eval/scripts")
from run_test_set import strip_thinking as s
assert s("<think>reasoning</think>\\n\\nbilling") == "billing", "matched pair"
assert s("<think>a</think>x<think>b</think> y") == "x y", "multiple pairs"
assert s("reasoning...\\n</think>\\n\\nbilling") == "billing", "pre-opened template (bare </think>)"
assert s("<think>never closes, budget exhausted") == "", "truncated self-opened"
assert s("r</think>\\nanswer\\n<think>more reasoning trunc") == "answer", "pre-opened then truncated re-open"
assert s("clean answer") == "clean answer", "no-op on clean text"
print("STRIP_OK")
"""


def run_script(test_file, output, *extra, timeout=540):
    return subprocess.run(
        ["uv", "run", SCRIPT, "--model", MODEL, "--test-file", test_file,
         "--output", output, "--max-tokens", "60", *extra],
        capture_output=True, text=True, timeout=timeout)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main(tmp):
    # 1. Strip semantics on the real module (major fix: bare-</think> from
    # pre-opened Qwen3.5-style templates; minor fix: truncated lone <think>).
    r = subprocess.run(
        ["uv", "run", "--with", "mlx-lm>=0.21", "python3", "-c", STRIP_CODE],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    check("strip_thinking handles matched/pre-opened/truncated reasoning",
          r.returncode == 0 and "STRIP_OK" in r.stdout, r.stderr[-500:])

    # 2. Unrecognized record fails fast, with index, BEFORE the model load.
    bad_out = os.path.join(tmp, "bad.jsonl")
    r = run_script(os.path.join(FIXTURES, "cpt_stray.jsonl"), bad_out, timeout=120)
    check("stray CPT record exits non-zero naming the record",
          r.returncode != 0 and "record 1" in r.stderr and "text" in r.stderr, r.stderr[-500:])
    check("record validation happens before model load",
          "loading" not in r.stderr and not os.path.exists(bad_out), r.stderr[-500:])

    # 3. LIVE smoke: 3-record chat fixture, base model (no adapter).
    chat_out = os.path.join(tmp, "chat_preds.jsonl")
    r = run_script(os.path.join(FIXTURES, "chat_test.jsonl"), chat_out)
    check("live chat run exits 0", r.returncode == 0, r.stderr[-800:])
    rows = read_jsonl(chat_out) if os.path.exists(chat_out) else []
    check("live chat run wrote 3 prediction rows", len(rows) == 3, rows)
    check("every row has exactly the 4 contract keys",
          all(sorted(row) == ["expected", "id", "input", "predicted"] for row in rows), rows)
    check("no '<think>' or '</think>' in any predicted",
          all("<think>" not in row["predicted"] and "</think>" not in row["predicted"] for row in rows),
          [row["predicted"] for row in rows])
    check("ids and expected carried from the fixture",
          [row["id"] for row in rows] == ["r1", "r2", "r3"]
          and [row["expected"] for row in rows] == ["blue", "positive", ""], rows)
    check("predictions are non-empty answer text",
          all(row["predicted"].strip() for row in rows), [row["predicted"] for row in rows])

    # 4. --limit honors the limit.
    lim_out = os.path.join(tmp, "lim_preds.jsonl")
    r = run_script(os.path.join(FIXTURES, "chat_test.jsonl"), lim_out, "--limit", "1", timeout=180)
    rows = read_jsonl(lim_out) if r.returncode == 0 else []
    check("--limit 1 yields exactly 1 row (the first record)",
          r.returncode == 0 and len(rows) == 1 and rows[0]["id"] == "r1", (r.stderr[-300:], rows))

    # 5. --limit 0 means zero records, not 'no limit'.
    zero_out = os.path.join(tmp, "zero_preds.jsonl")
    r = run_script(os.path.join(FIXTURES, "chat_test.jsonl"), zero_out, "--limit", "0", timeout=180)
    check("--limit 0 yields 0 rows",
          r.returncode == 0 and read_jsonl(zero_out) == [] and "wrote 0 predictions" in r.stderr,
          r.stderr[-300:])

    # 6. Completions-format fixture works; id falls back to row index.
    comp_out = os.path.join(tmp, "comp_preds.jsonl")
    r = run_script(os.path.join(FIXTURES, "completions_test.jsonl"), comp_out, timeout=180)
    rows = read_jsonl(comp_out) if r.returncode == 0 else []
    check("completions format: 2 rows, expected=completion, id=row index",
          r.returncode == 0 and len(rows) == 2
          and [row["expected"] for row in rows] == ["cold", "4"]
          and [row["id"] for row in rows] == [0, 1]
          and all(sorted(row) == ["expected", "id", "input", "predicted"] for row in rows),
          (r.stderr[-300:], rows))

    # 7. --enable-thinking live: Qwen3 self-opens <think>; at 60 tokens the
    # reasoning truncates — the lone-opener strip must still leave no markers.
    think_out = os.path.join(tmp, "think_preds.jsonl")
    r = run_script(os.path.join(FIXTURES, "chat_test.jsonl"), think_out,
                   "--limit", "1", "--enable-thinking", timeout=180)
    rows = read_jsonl(think_out) if r.returncode == 0 else []
    check("--enable-thinking output still carries no think markers",
          r.returncode == 0 and len(rows) == 1
          and "<think>" not in rows[0]["predicted"] and "</think>" not in rows[0]["predicted"],
          (r.stderr[-300:], rows))

    if failures:
        sys.exit(f"{len(failures)} check(s) failed: {failures}")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        main(tmp)
