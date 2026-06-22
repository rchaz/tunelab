#!/usr/bin/env python3
"""Tests for skills/tune-data/scripts/distill_generate.py against the fake
anthropic and openai SDKs in tests/shims (no API key, no network — the real
script runs via 'uv run' with PYTHONPATH pointing at the shims)."""

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
    env.pop("OPENAI_API_KEY", None)
    env.pop("SHIM_DEBUG", None)
    env.update(extra)
    return env


def openai_env(**extra):
    # Deliberately drops ANTHROPIC_API_KEY: the openai path must not need it.
    env = dict(os.environ)
    env["PYTHONPATH"] = SHIMS
    env["OPENAI_API_KEY"] = "shim-dummy-key"
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("SHIM_DEBUG", None)
    env.update(extra)
    return env


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


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

    # ===== --provider openai (fake openai SDK in tests/shims) ===============
    probe = subprocess.run(
        ["uv", "run", "--no-project", "--with", "openai>=2.41",
         "python", "-c", "import openai; print(openai.__file__)"],
        capture_output=True, text=True, env=openai_env(), cwd=ROOT,
    )
    assert probe.returncode == 0, probe.stderr
    assert os.path.join("tests", "shims") in probe.stdout, (
        f"shim did not shadow real package: {probe.stdout!r}")
    print("PASS: shim shadows real openai package under uv run "
          f"(openai.__file__ = {probe.stdout.strip()})")

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "labeled.jsonl")
        train = os.path.join(tmp, "train.jsonl")

        # --- classify on 10 inputs incl. refusal marker, openai path -------
        r = run(classify_args(out, train, ["--provider", "openai"]),
                env=openai_env())
        assert r.returncode == 0, r.stderr
        assert "processing 10 of 10 records with gpt-5.5" in r.stderr, r.stderr
        print("PASS: openai classify run exits 0; default model resolves "
              "to gpt-5.5 (ANTHROPIC_API_KEY absent from env)")

        raw = read_jsonl(out)
        assert len(raw) == 9, f"expected 9 raw records, got {len(raw)}"
        assert [rec["id"] for rec in raw] == [g["id"] for g in good]
        for rec in raw:
            assert rec["label"] == expected_label(rec["text"]), rec
        chats = read_jsonl(train)
        assert len(chats) == 9
        for chat, rec in zip(chats, raw):
            msgs = chat["messages"]
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
            assert msgs[2]["content"] == rec["label"]
        print("PASS: openai classify writes the same 9 deterministic labels "
              "and aligned 3-role train records as the anthropic path")

        assert "id=r07 skipped (refusal: (refusing))" in r.stderr, r.stderr
        assert "9 written, 1 skipped" in r.stderr, r.stderr
        print("PASS: openai refusal content part skipped with stderr note "
              "'id=r07 skipped (refusal: (refusing))'")

        # --- resume on the openai path --------------------------------------
        r2 = run(classify_args(out, train, ["--provider", "openai"]),
                 env=openai_env())
        assert r2.returncode == 0, r2.stderr
        assert "resuming: 9 already done" in r2.stderr, r2.stderr
        assert "processing 1 of 10" in r2.stderr, r2.stderr
        assert "0 written, 1 skipped" in r2.stderr, r2.stderr
        print("PASS: openai resume re-run reports 'resuming: 9 already done' "
              "and retries only the refused record")

    # --- generate mode, openai path ----------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "gen.jsonl")
        train = os.path.join(tmp, "gen_train.jsonl")
        r = run(["--mode", "generate",
                 "--input", os.path.join(FIXTURES, "generate_inputs.jsonl"),
                 "--system", "Draft a reply in our support voice.",
                 "--output", out, "--train-out", train,
                 "--provider", "openai"], env=openai_env())
        assert r.returncode == 0, r.stderr
        assert "3 written, 0 skipped" in r.stderr, r.stderr
        for rec, chat in zip(read_jsonl(out), read_jsonl(train)):
            assert rec["generated"].startswith("GEN::"), rec
            assert chat["messages"][2]["content"] == rec["generated"], chat
        print("PASS: openai generate mode writes shim output ('GEN::<input>') "
              "to raw and train outputs for all 3 inputs")

    # --- openai request shape: max_output_tokens / store / reasoning --------
    dbg = openai_env(SHIM_DEBUG="1")
    r = one_shot("classify", ["--provider", "openai"], dbg)
    assert "SHIM max_output_tokens=256 store=False effort=none" in r.stderr, r.stderr
    print("PASS: openai classify default max_output_tokens=256 with "
          "store=False and reasoning effort 'none' on every call")
    r = one_shot("classify", ["--provider", "openai", "--max-tokens", "99"], dbg)
    assert "SHIM max_output_tokens=99 store=False effort=none" in r.stderr, r.stderr
    print("PASS: explicit --max-tokens 99 honored on the openai path")
    r = one_shot("generate", ["--provider", "openai"], dbg)
    assert "SHIM max_output_tokens=1024 store=False effort=none" in r.stderr, r.stderr
    print("PASS: openai generate default max_output_tokens=1024")

    # --- non-reasoning openai models omit `reasoning` (gpt-4o etc. 400 on it) --
    # The real Responses API rejects `reasoning` on gpt-4o/4.1/etc.; gating lets a
    # user probe their actual incumbent as the ceiling instead of getting a 400.
    r = one_shot("classify", ["--provider", "openai", "--model", "gpt-4o"], dbg)
    assert r.returncode == 0, r.stderr
    assert "effort=None" in r.stderr, r.stderr  # param omitted -> shim prints None
    print("PASS: gpt-4o (non-reasoning) omits the reasoning param — incumbent "
          "ceiling probe no longer 400s")
    r = one_shot("classify", ["--provider", "openai", "--model", "gpt-5.4-nano"], dbg)
    assert "effort=none" in r.stderr, r.stderr  # gpt-5.x still gets reasoning
    print("PASS: gpt-5.x reasoning models still send reasoning effort=none")

    # --- --gold-key emits cascade/eval-ready {id,text,predicted,expected} -------
    # Lets a frontier ceiling probe over labeled data score + compose with no
    # manual re-join (gold under 'expected', prediction under 'predicted').
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.jsonl")
        write_jsonl(inp, [{"id": "g1", "text": "hello", "label": "billing"},
                          {"id": "g2", "text": "world", "label": "spam"}])
        out = os.path.join(tmp, "o.jsonl")
        r = run(["--mode", "classify", "--input", inp, "--labels", ",".join(LABELS),
                 "--system", "s", "--output", out,
                 "--train-out", os.path.join(tmp, "t.jsonl"),
                 "--gold-key", "label", "--provider", "openai"], env=openai_env())
        assert r.returncode == 0, r.stderr
        recs = read_jsonl(out)
        assert len(recs) == 2, recs
        for rec, gold in zip(recs, ["billing", "spam"]):
            assert set(rec) == {"id", "text", "predicted", "expected"}, rec
            assert rec["expected"] == gold, rec        # gold preserved, not clobbered
            assert rec["predicted"] in LABELS, rec     # prediction under its own key
            assert "label" not in rec, rec
        print("PASS: --gold-key emits cascade/eval-ready {id,text,predicted,expected} "
              "(gold preserved under 'expected', prediction under 'predicted')")

    # --- --gold-key naming a missing field fails loud before any API call -------
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.jsonl")
        write_jsonl(inp, [{"id": "g1", "text": "hello"}])  # no 'label' field
        r = run(["--mode", "classify", "--input", inp, "--labels", ",".join(LABELS),
                 "--system", "s", "--output", os.path.join(tmp, "o.jsonl"),
                 "--train-out", os.path.join(tmp, "t.jsonl"),
                 "--gold-key", "label", "--provider", "openai"], env=openai_env())
        assert r.returncode != 0 and "gold-key" in r.stderr, r.stderr
        print("PASS: --gold-key naming a missing field exits non-zero with a clear message")

    # --- incomplete (truncated) openai response is skipped, never written ---
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.jsonl")
        write_jsonl(inp, [{"id": "t1", "text": "TRUNCATE_ME long prompt"}])
        out = os.path.join(tmp, "o.jsonl")
        r = run(["--mode", "generate", "--input", inp, "--system", "s",
                 "--output", out, "--train-out", os.path.join(tmp, "t.jsonl"),
                 "--provider", "openai"], env=openai_env())
        assert r.returncode == 0, r.stderr
        assert "id=t1 skipped (status=incomplete (max_output_tokens))" in r.stderr, r.stderr
        assert "0 written, 1 skipped" in r.stderr, r.stderr
        assert read_jsonl(out) == []
        print("PASS: openai status=incomplete response skipped with reason, "
              "never written (truncation-filter side of the provider asymmetry)")

    # --- usage-less response under-counts to 0 instead of crashing ----------
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.jsonl")
        write_jsonl(inp, [{"id": "u1", "text": "NOUSAGE_ME usage is None"}])
        r = run(["--mode", "classify", "--input", inp,
                 "--labels", ",".join(LABELS), "--system", "s",
                 "--output", os.path.join(tmp, "o.jsonl"),
                 "--train-out", os.path.join(tmp, "t.jsonl"),
                 "--provider", "openai"], env=openai_env())
        assert r.returncode == 0, r.stderr
        assert "1 written, 0 skipped" in r.stderr, r.stderr
        assert "tokens: 0 in / 0 out" in r.stderr, r.stderr
        print("PASS: usage=None openai response (Response.usage is Optional) "
              "under-counts to 0 tokens instead of crashing the run")

    # --- first-5-calls-all-api-errors abort ----------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.jsonl")
        write_jsonl(inp, [{"id": f"e{i}", "text": f"ERROR_ME {i}"} for i in range(6)])
        r = run(["--mode", "classify", "--input", inp,
                 "--labels", ",".join(LABELS), "--system", "s",
                 "--output", os.path.join(tmp, "o.jsonl"),
                 "--train-out", os.path.join(tmp, "t.jsonl"),
                 "--provider", "openai"], env=openai_env())
        assert r.returncode != 0, r.stderr
        assert ("aborting: first 5 calls all failed with api errors — last error: "
                "shim-injected api error (model=gpt-5.5, provider=openai") in r.stderr, r.stderr
        assert r.stderr.count("skipped (api error") == 5, r.stderr
        print("PASS: run aborts non-zero after the first 5 calls all fail "
              "with api errors (6th record never attempted) — a typo'd "
              "--model/--provider cannot burn a full run")

    # --- provider isolation: each path never imports the other SDK ----------
    # A poisoned module that raises on import is placed AHEAD of tests/shims
    # on PYTHONPATH; the run only stays green if that SDK is never imported.
    with tempfile.TemporaryDirectory() as tmp:
        poison_an = os.path.join(tmp, "poison_an")
        poison_oa = os.path.join(tmp, "poison_oa")
        os.makedirs(poison_an)
        os.makedirs(poison_oa)
        with open(os.path.join(poison_an, "anthropic.py"), "w", encoding="utf-8") as f:
            f.write('raise ImportError("poisoned: anthropic imported on the openai path")\n')
        with open(os.path.join(poison_oa, "openai.py"), "w", encoding="utf-8") as f:
            f.write('raise ImportError("poisoned: openai imported on the anthropic path")\n')

        out = os.path.join(tmp, "o.jsonl")
        r = run(classify_args(out, os.path.join(tmp, "t.jsonl"),
                              ["--provider", "openai", "--limit", "3"]),
                env=openai_env(PYTHONPATH=os.pathsep.join([poison_an, SHIMS])))
        assert r.returncode == 0, r.stderr
        assert len(read_jsonl(out)) == 3
        print("PASS: --provider openai runs green with a poisoned anthropic "
              "module first on PYTHONPATH — openai path never imports anthropic")

        out2 = os.path.join(tmp, "o2.jsonl")
        r = run(classify_args(out2, os.path.join(tmp, "t2.jsonl"), ["--limit", "3"]),
                env=base_env(PYTHONPATH=os.pathsep.join([poison_oa, SHIMS])))
        assert r.returncode == 0, r.stderr
        assert len(read_jsonl(out2)) == 3
        print("PASS: --provider anthropic runs green with a poisoned openai "
              "module first on PYTHONPATH — anthropic path never imports openai")

    # --- per-provider key gates ----------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        gate_args = classify_args(os.path.join(tmp, "o.jsonl"),
                                  os.path.join(tmp, "t.jsonl"))
        r = run(gate_args + ["--provider", "openai"], env=base_env())
        assert r.returncode != 0
        assert "OPENAI_API_KEY is not set" in r.stderr, r.stderr
        print("PASS: --provider openai without OPENAI_API_KEY exits non-zero "
              "with 'OPENAI_API_KEY is not set' (ANTHROPIC key present, unused)")
        r = run(gate_args, env=openai_env())
        assert r.returncode != 0
        assert "ANTHROPIC_API_KEY is not set" in r.stderr, r.stderr
        print("PASS: default provider without ANTHROPIC_API_KEY exits non-zero "
              "with 'ANTHROPIC_API_KEY is not set' (OPENAI key present, unused)")

    # --- --provider flag covers both providers in --help ---------------------
    r = run(["--help"])
    assert "--provider {anthropic,openai}" in r.stdout, r.stdout
    assert "OpenAI lands in Phase 2" not in r.stdout, r.stdout
    assert "gpt-5.5" in r.stdout and "claude-opus-4-8" in r.stdout, r.stdout
    print("PASS: --provider {anthropic,openai} flag present with per-provider "
          "model defaults in --help (Phase-2 gating text gone)")

    print("ALL distill_generate CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
