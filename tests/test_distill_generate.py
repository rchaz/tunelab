#!/usr/bin/env python3
"""Tests for skills/tune-data/scripts/distill_generate.py against the fake
anthropic SDK in tests/shims (no API key, no network — the real script runs
via 'uv run' with PYTHONPATH pointing at the shim)."""

import json
import os
import subprocess
import sys
import tempfile
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "skills", "tune-data", "scripts", "distill_generate.py")
SHIMS = os.path.join(ROOT, "tests", "shims")
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "apiscripts")

LABELS = ["billing", "receipt", "spam", "other"]


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


def expected_label(text):
    return LABELS[zlib.crc32(text.encode("utf-8")) % len(LABELS)]


def classify_args(out, train, extra=None):
    return [
        "--mode", "classify",
        "--input", os.path.join(FIXTURES, "distill_inputs.jsonl"),
        "--labels", ",".join(LABELS),
        "--system", "You label inbound emails.",
        "--output", out, "--train-out", train,
    ] + (extra or [])


def main():
    # --- shim shadowing probe ---------------------------------------------
    probe = subprocess.run(
        ["uv", "run", "--no-project", "--with", "anthropic>=0.92",
         "python", "-c", "import anthropic; print(anthropic.__file__)"],
        capture_output=True, text=True, env=base_env(), cwd=ROOT,
    )
    assert probe.returncode == 0, probe.stderr
    assert os.path.join("tests", "shims") in probe.stdout, (
        f"shim did not shadow real package: {probe.stdout!r}")
    print("PASS: shim shadows real anthropic package under uv run "
          f"(anthropic.__file__ = {probe.stdout.strip()})")

    inputs = read_jsonl(os.path.join(FIXTURES, "distill_inputs.jsonl"))
    good = [r for r in inputs if "REFUSE_ME" not in r["text"]]

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "labeled.jsonl")
        train = os.path.join(tmp, "train.jsonl")

        # --- classify on 10 inputs incl. refusal marker -------------------
        r = run(classify_args(out, train))
        assert r.returncode == 0, r.stderr
        print("PASS: classify run on 10 inputs (1 refusal) exits 0")

        raw = read_jsonl(out)
        assert len(raw) == 9, f"expected 9 raw records, got {len(raw)}"
        assert [rec["id"] for rec in raw] == [g["id"] for g in good]
        print("PASS: --output has 9 records; refusal id r07 absent")

        for rec in raw:
            assert rec["label"] in LABELS, rec
            assert rec["label"] == expected_label(rec["text"]), rec
        print("PASS: all 9 labels are from the allowed set and match the "
              "shim's deterministic per-input label")

        chats = read_jsonl(train)
        assert len(chats) == 9, f"expected 9 train records, got {len(chats)}"
        for chat, rec in zip(chats, raw):
            msgs = chat["messages"]
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
            assert msgs[1]["content"] == rec["text"]
            assert msgs[2]["content"] == rec["label"]
        print("PASS: --train-out is valid 3-role chat format aligned with raw records")

        with open(train, encoding="utf-8") as f:
            assert "Café" in f.read()
        print("PASS: non-ASCII text round-trips un-escaped in train-out")

        assert "id=r07 skipped (stop_reason=refusal)" in r.stderr, r.stderr
        print("PASS: refusal skipped with stderr note "
              "'id=r07 skipped (stop_reason=refusal)'")
        assert "9 written, 1 skipped" in r.stderr, r.stderr
        assert "re-run to retry" in r.stderr, r.stderr
        print("PASS: final summary prints '9 written, 1 skipped' and re-run warning")

        # --- resume: re-run same command -----------------------------------
        r2 = run(classify_args(out, train))
        assert r2.returncode == 0, r2.stderr
        assert "resuming: 9 already done" in r2.stderr, r2.stderr
        assert "processing 1 of 10" in r2.stderr, r2.stderr
        assert "0 written, 1 skipped" in r2.stderr, r2.stderr
        assert len(read_jsonl(out)) == 9
        print("PASS: resume re-run reports 'resuming: 9 already done' and "
              "attempts only the remaining record ('processing 1 of 10')")

        # --- truncated trailing line in --output ---------------------------
        with open(out, encoding="utf-8") as f:
            lines = f.readlines()
        with open(out, "w", encoding="utf-8") as f:
            f.writelines(lines[:-1])
            f.write('{"id": "r10", "te')  # partial line, no newline
        r3 = run(classify_args(out, train))
        assert r3.returncode == 0, r3.stderr
        assert "truncated trailing line" in r3.stderr, r3.stderr
        assert "resuming: 8 already done" in r3.stderr, r3.stderr
        assert "processing 2 of 10" in r3.stderr, r3.stderr
        assert "1 written, 1 skipped" in r3.stderr, r3.stderr
        print("PASS: truncated trailing line in --output is warned about and "
              "the partial record is retried (no crash)")

        raw3 = read_jsonl(out)  # raises if any line is corrupt
        assert [rec["id"] for rec in raw3] == [g["id"] for g in good]
        print("PASS: after recovery --output parses fully with all 9 ids")

        assert len(read_jsonl(train)) == 10
        print("PASS: retry appended a duplicate train record (dedupe's job), "
              "never a missing one")

        r4 = run(classify_args(out, train))
        assert "resuming: 9 already done" in r4.stderr, r4.stderr
        assert "truncated trailing line" not in r4.stderr
        print("PASS: subsequent resume is clean — repair is one-shot")

    # --- generate mode on 3 inputs ----------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "gen.jsonl")
        train = os.path.join(tmp, "gen_train.jsonl")
        r = run([
            "--mode", "generate",
            "--input", os.path.join(FIXTURES, "generate_inputs.jsonl"),
            "--system", "Draft a reply in our support voice.",
            "--output", out, "--train-out", train,
        ])
        assert r.returncode == 0, r.stderr
        assert "3 written, 0 skipped" in r.stderr, r.stderr
        gen_inputs = read_jsonl(os.path.join(FIXTURES, "generate_inputs.jsonl"))
        raw = read_jsonl(out)
        chats = read_jsonl(train)
        assert len(raw) == len(chats) == 3
        for inp, rec, chat in zip(gen_inputs, raw, chats):
            want = "GEN::" + inp["text"]
            assert rec["generated"] == want, rec
            assert chat["messages"][2]["content"] == want, chat
        print("PASS: generate mode train-out assistant content matches shim "
              "output ('GEN::<input>') for all 3 inputs")

    # --- max-tokens flag handling (SHIM_DEBUG surfaces the request value) --
    def one_shot(mode, extra, debug_env):
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--mode", mode,
                    "--input", os.path.join(FIXTURES,
                        "distill_inputs.jsonl" if mode == "classify"
                        else "generate_inputs.jsonl"),
                    "--system", "s",
                    "--output", os.path.join(tmp, "o.jsonl"),
                    "--train-out", os.path.join(tmp, "t.jsonl"),
                    "--limit", "1"] + extra
            if mode == "classify":
                args += ["--labels", ",".join(LABELS)]
            return run(args, env=debug_env)

    dbg = base_env(SHIM_DEBUG="1")
    r = one_shot("classify", [], dbg)
    assert "SHIM max_tokens=256" in r.stderr, r.stderr
    print("PASS: classify default max_tokens is 256")
    r = one_shot("classify", ["--max-tokens", "99"], dbg)
    assert "SHIM max_tokens=99" in r.stderr, r.stderr
    print("PASS: explicit --max-tokens 99 is honored in classify mode "
          "(no silent override)")
    r = one_shot("generate", [], dbg)
    assert "SHIM max_tokens=1024" in r.stderr, r.stderr
    print("PASS: generate default max_tokens is 1024")

    # --- label parsing edge cases ------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        r = run(["--mode", "classify",
                 "--input", os.path.join(FIXTURES, "distill_inputs.jsonl"),
                 "--labels", "billing,receipt,",  # trailing comma
                 "--system", "s",
                 "--output", os.path.join(tmp, "o.jsonl"),
                 "--train-out", os.path.join(tmp, "t.jsonl"),
                 "--limit", "2"])
        assert r.returncode == 0, r.stderr
        for rec in read_jsonl(os.path.join(tmp, "o.jsonl")):
            assert rec["label"] in ("billing", "receipt"), rec
        print("PASS: trailing comma in --labels yields no empty-string label")

    with tempfile.TemporaryDirectory() as tmp:
        r = run(["--mode", "classify",
                 "--input", os.path.join(FIXTURES, "distill_inputs.jsonl"),
                 "--labels", ",,", "--system", "s",
                 "--output", os.path.join(tmp, "o.jsonl"),
                 "--train-out", os.path.join(tmp, "t.jsonl")])
        assert r.returncode != 0
        assert "empty label set" in r.stderr, r.stderr
        print("PASS: --labels ',,' exits non-zero with 'empty label set'")

    # --- --provider flag is present and gated ------------------------------
    r = run(["--help"])
    assert "--provider" in r.stdout and "anthropic" in r.stdout, r.stdout
    assert "OpenAI lands in Phase 2" in r.stdout, r.stdout
    print("PASS: --provider {anthropic} flag present with Phase-2 help text")

    print("ALL distill_generate CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
