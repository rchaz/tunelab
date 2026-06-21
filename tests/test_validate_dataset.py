#!/usr/bin/env python3
"""Evidence tests for validate_dataset.py — runs the real script via subprocess.

Committed fixtures under tests/fixtures/dataprep/: good_chat, broken_chat,
tools, text. Edge-case dirs (empty splits, content-less tool_calls turns,
chat/tools mixes, over-long records) are generated into a tempdir because
they are either byte-sensitive or too large to keep hand-readable.
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "skills", "tune-data", "scripts", "validate_dataset.py")
FIX = os.path.join(ROOT, "tests", "fixtures", "dataprep")

CHAT_REC = {
    "messages": [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]
}
TOOLS_REC = {
    "messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "Sunny."},
    ],
    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
}


def ok(name, cond, detail=""):
    if not cond:
        print(f"FAIL: {name} -- {detail}", file=sys.stderr)
        sys.exit(1)
    print(f"PASS: {name}")


def run(data_dir):
    return subprocess.run(
        ["python3", SCRIPT, "--data-dir", data_dir], capture_output=True, text=True
    )


def make_dir(td, name, **splits):
    d = os.path.join(td, name)
    os.makedirs(d)
    for split, lines in splits.items():
        with open(os.path.join(d, f"{split}.jsonl"), "w") as f:
            for rec in lines:
                f.write((rec if isinstance(rec, str) else json.dumps(rec)) + "\n")
    return d


def main():
    # Committed fixtures.
    r = run(os.path.join(FIX, "good_chat"))
    ok(
        "clean chat dir: exit 0, OK, format=chat, label distribution",
        r.returncode == 0
        and "OK — safe to train" in r.stdout
        and "format=chat" in r.stdout
        and "label distribution" in r.stdout,
        r.stdout,
    )

    r = run(os.path.join(FIX, "broken_chat"))
    ok("broken chat dir exits 1", r.returncode == 1, r.stdout)
    for snippet in (
        "invalid JSON",
        "must end with an assistant turn",
        "empty user content",
        "format completions != file format chat",
    ):
        ok(f"broken chat dir reports ERROR: {snippet}", snippet in r.stdout, r.stdout)

    r = run(os.path.join(FIX, "tools"))
    ok(
        "tools dir accepted: exit 0, format=tools",
        r.returncode == 0 and "format=tools" in r.stdout,
        r.stdout,
    )

    r = run(os.path.join(FIX, "text"))
    ok(
        "text (CPT) dir: exit 0, format=text",
        r.returncode == 0
        and "format=text" in r.stdout
        and "OK — safe to train" in r.stdout,
        r.stdout,
    )

    td = tempfile.mkdtemp(prefix="valtest_")

    # >2048-approx-token record -> warning naming --max-seq-length, not an error.
    long_rec = {"prompt": "word " * 2200, "completion": "done"}
    d = make_dir(td, "longrec", train=[long_rec], valid=[{"prompt": "a", "completion": "b"}])
    r = run(d)
    ok(
        "over-long record warns naming --max-seq-length, exit 0",
        r.returncode == 0
        and "exceeds mlx_lm --max-seq-length default 2048" in r.stdout,
        r.stdout,
    )

    # Regression (major 1): final assistant turn with tool_calls and NO
    # "content" key, chat format (no top-level tools array) — used to KeyError.
    no_content = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "tool_calls": [{"name": "get_weather"}]},
        ]
    }
    d = make_dir(td, "nocontent", train=[no_content])
    r = run(d)
    ok(
        "content-less tool_calls assistant turn: no crash, exit 0, suggest-tools warning",
        r.returncode == 0
        and "Traceback" not in r.stderr
        and "no top-level 'tools' array" in r.stdout,
        r.stdout + r.stderr,
    )

    # Regression (major 2a): empty train.jsonl must FAIL the gate.
    d = make_dir(td, "emptytrain", train=["", ""])
    r = run(d)
    ok(
        "empty train.jsonl: exit 1 citing mlx-lm 'Training set not found or empty'",
        r.returncode == 1 and "Training set not found or empty" in r.stdout,
        r.stdout,
    )

    # Regression (major 2b): present-but-empty valid.jsonl still warns.
    d = make_dir(td, "emptyvalid", train=[CHAT_REC], valid=[""])
    r = run(d)
    ok(
        "empty valid.jsonl: exit 0 with validation-loss-monitoring warning",
        r.returncode == 0
        and "validation loss monitoring will be unavailable" in r.stdout,
        r.stdout,
    )

    # Regression (minor): non-dict entry in messages -> ERROR line, no traceback.
    d = make_dir(td, "nondict", train=['{"messages": ["just a string", {"role": "assistant", "content": "ok"}]}'])
    r = run(d)
    ok(
        "non-dict message entry: exit 1 with per-line ERROR, no traceback",
        r.returncode == 1
        and "non-object entry" in r.stdout
        and "Traceback" not in r.stderr,
        r.stdout + r.stderr,
    )

    # Regression (minor): chat/tools mixing is a warning, not a gate failure —
    # mlx-lm 0.31.3 ChatDataset reads "tools" per record.
    d = make_dir(td, "mixfile", train=[CHAT_REC, TOOLS_REC])
    r = run(d)
    ok(
        "intra-file chat+tools mix: exit 0 with mix warning",
        r.returncode == 0 and "mlx-lm accepts the mix" in r.stdout,
        r.stdout,
    )

    d = make_dir(td, "mixsplits", train=[CHAT_REC], valid=[TOOLS_REC])
    r = run(d)
    ok(
        "cross-split chat train + tools valid: exit 0 with mix warning",
        r.returncode == 0 and "splits mix chat and tools" in r.stdout,
        r.stdout,
    )

    # Regression (nit): detection order matches mlx-lm create_dataset —
    # a record with both shapes trains as completions, so report completions.
    both = dict(CHAT_REC, prompt="p text", completion="c text")
    d = make_dir(td, "bothshapes", train=[both])
    r = run(d)
    ok(
        "record with prompt+completion AND messages reported as completions",
        r.returncode == 0 and "format=completions" in r.stdout,
        r.stdout,
    )

    print("ALL CHECKS PASSED (test_validate_dataset)")


if __name__ == "__main__":
    main()
