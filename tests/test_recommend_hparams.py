#!/usr/bin/env python3
"""Evidence tests for skills/tune-train/scripts/recommend_hparams.py.

Invokes the real script via subprocess; prints one 'PASS: <check>' line per
check and exits non-zero on the first failure.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "skills", "tune-train", "scripts", "recommend_hparams.py")
FIXTURES = os.path.join(HERE, "fixtures", "train")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name} {detail}", file=sys.stderr)


def run(args):
    return subprocess.run(
        [sys.executable, SCRIPT] + args, capture_output=True, text=True
    )


def write_chat(path, n, content_chars=40):
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": f"question {i} " + "x" * (content_chars // 2)},
                    {"role": "assistant", "content": f"answer {i} " + "y" * (content_chars // 2)},
                ]
            }) + "\n")


def flag_val(out, flag):
    m = re.search(rf"{re.escape(flag)} (\d+)", out)
    return int(m.group(1)) if m else None


with tempfile.TemporaryDirectory() as td:
    # --- Check 1: n=200 chat, 1.7b, 16GB ---
    d1 = os.path.join(td, "c1")
    os.makedirs(d1)
    write_chat(os.path.join(d1, "train.jsonl"), 200)
    r = run(["--train-file", os.path.join(d1, "train.jsonl"),
             "--model-size", "1.7b", "--task", "sft", "--memory-gb", "16"])
    check("chat n=200 1.7b exits 0", r.returncode == 0, r.stderr)
    expected_iters = max(100, int(5 * 200 / 4))  # epochs 5, batch 4
    check(f"chat n=200 iters == {expected_iters}",
          flag_val(r.stdout, "--iters") == expected_iters,
          f"got {flag_val(r.stdout, '--iters')}")
    check("chat n=200 --num-layers 8", flag_val(r.stdout, "--num-layers") == 8,
          f"got {flag_val(r.stdout, '--num-layers')}")
    check("chat n=200 --mask-prompt present", "--mask-prompt" in r.stdout)
    se, spe = flag_val(r.stdout, "--save-every"), flag_val(r.stdout, "--steps-per-eval")
    check("chat n=200 --save-every present and <= --steps-per-eval",
          se is not None and spe is not None and se <= spe, f"save-every={se} steps-per-eval={spe}")
    check("chat n=200 --seed 42 present", "--seed 42" in r.stdout)

    # --- Check 2: n=3000 chat, 8b, 16GB ---
    d2 = os.path.join(td, "c2")
    os.makedirs(d2)
    write_chat(os.path.join(d2, "train.jsonl"), 3000)
    r = run(["--train-file", os.path.join(d2, "train.jsonl"),
             "--model-size", "8b", "--task", "sft", "--memory-gb", "16"])
    check("chat n=3000 8b exits 0", r.returncode == 0, r.stderr)
    check("chat n=3000 8b LR 5e-05", "--learning-rate 5e-05" in r.stdout, r.stdout)
    check("chat n=3000 8b@16GB --grad-checkpoint present", "--grad-checkpoint" in r.stdout)
    check("chat n=3000 --num-layers 16", flag_val(r.stdout, "--num-layers") == 16,
          f"got {flag_val(r.stdout, '--num-layers')}")

    # --- Check 3: CPT on a text-format file ---
    d3 = os.path.join(td, "c3")
    os.makedirs(d3)
    with open(os.path.join(d3, "train.jsonl"), "w") as f:
        for i in range(500):
            f.write(json.dumps({"text": f"Domain sentence number {i}. " * 5}) + "\n")
    r = run(["--train-file", os.path.join(d3, "train.jsonl"),
             "--model-size", "0.6b", "--task", "cpt", "--memory-gb", "16"])
    check("cpt text exits 0", r.returncode == 0, r.stderr)
    check("cpt text NO --mask-prompt", "--mask-prompt" not in r.stdout, r.stdout)
    check("cpt text LR 1e-05", "--learning-rate 1e-05" in r.stdout, r.stdout)

    # --- Check 4: one very long record -> --max-seq-length ---
    d4 = os.path.join(td, "c4")
    os.makedirs(d4)
    long_chars = 12000  # ~3000 approx tokens, well over 2048
    with open(os.path.join(d4, "train.jsonl"), "w") as f:
        f.write(json.dumps({"text": "z" * long_chars}) + "\n")
        for i in range(60):
            f.write(json.dumps({"text": f"short record {i}"}) + "\n")
    r = run(["--train-file", os.path.join(d4, "train.jsonl"),
             "--model-size", "0.6b", "--task", "cpt", "--memory-gb", "16"])
    check("long record exits 0", r.returncode == 0, r.stderr)
    msl = flag_val(r.stdout, "--max-seq-length")
    approx = long_chars // 4
    check("long record --max-seq-length emitted", msl is not None, r.stdout)
    check("long record --max-seq-length multiple of 1024",
          msl is not None and msl % 1024 == 0, f"got {msl}")
    check(f"long record --max-seq-length >= approx tokens ({approx})",
          msl is not None and msl >= approx, f"got {msl}")

    # --- Check 5: --lora-rank 16 -> lora_config.yaml + '-c' form ---
    d5 = os.path.join(td, "c5")
    os.makedirs(d5)
    write_chat(os.path.join(d5, "train.jsonl"), 200)
    outdir = os.path.join(td, "c5out")
    r = run(["--train-file", os.path.join(d5, "train.jsonl"),
             "--model-size", "1.7b", "--task", "sft", "--memory-gb", "16",
             "--lora-rank", "16", "--outdir", outdir])
    cfg_path = os.path.join(outdir, "lora_config.yaml")
    check("--lora-rank 16 exits 0", r.returncode == 0, r.stderr)
    check("--lora-rank 16 writes lora_config.yaml", os.path.isfile(cfg_path))
    cfg_text = open(cfg_path).read() if os.path.isfile(cfg_path) else ""
    yaml_ok = True
    for ln in cfg_text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if not re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*:(\s.*|)$", ln):
            yaml_ok = False
            break
    check("lora_config.yaml structurally valid YAML mapping lines", yaml_ok, cfg_text)
    check("lora_config.yaml contains rank: 16", re.search(r"^\s*rank: 16\s*$", cfg_text, re.M) is not None)
    check("lora_config.yaml contains lora_parameters", "lora_parameters:" in cfg_text)
    check("--lora-rank command uses -c", re.search(r"mlx_lm\.lora -c .*lora_config\.yaml", r.stdout) is not None, r.stdout)

    # --- Check 6: empty train file ---
    d6 = os.path.join(td, "c6")
    os.makedirs(d6)
    open(os.path.join(d6, "train.jsonl"), "w").close()
    r = run(["--train-file", os.path.join(d6, "train.jsonl"),
             "--model-size", "1.7b", "--memory-gb", "16"])
    check("empty train file exits non-zero", r.returncode != 0)
    check("empty train file message is clear", "empty" in r.stderr.lower(), r.stderr)

    # --- Review-fix regressions ---
    # num-layers 8 under 300 is unqualified (applies to CPT too)
    d7 = os.path.join(td, "c7")
    os.makedirs(d7)
    with open(os.path.join(d7, "train.jsonl"), "w") as f:
        for i in range(41):
            f.write(json.dumps({"text": f"tiny corpus chunk {i}"}) + "\n")
    r = run(["--train-file", os.path.join(d7, "train.jsonl"),
             "--model-size", "0.6b", "--task", "cpt", "--memory-gb", "16"])
    check("cpt n=41 --num-layers 8 (heuristic unqualified by task)",
          r.returncode == 0 and flag_val(r.stdout, "--num-layers") == 8,
          f"rc={r.returncode} got {flag_val(r.stdout, '--num-layers')}")

    # basename != train.jsonl warns on stderr but still exits 0
    d8 = os.path.join(td, "c8")
    os.makedirs(d8)
    write_chat(os.path.join(d8, "cpt.jsonl"), 50)
    r = run(["--train-file", os.path.join(d8, "cpt.jsonl"),
             "--model-size", "1.7b", "--memory-gb", "16"])
    check("non-train.jsonl basename warns on stderr",
          r.returncode == 0 and "train.jsonl" in r.stderr and "WARNING" in r.stderr, r.stderr)

    # tool_calls + tools payloads count toward the max-seq-length estimate
    d9 = os.path.join(td, "c9")
    os.makedirs(d9)
    big_args = json.dumps({"query": "q" * 9000})
    with open(os.path.join(d9, "train.jsonl"), "w") as f:
        f.write(json.dumps({
            "messages": [
                {"role": "user", "content": "run the search"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "search", "arguments": big_args}}]},
            ],
            "tools": [{"type": "function", "function": {"name": "search"}}],
        }) + "\n")
        for i in range(30):
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"}]}) + "\n")
    r = run(["--train-file", os.path.join(d9, "train.jsonl"),
             "--model-size", "1.7b", "--memory-gb", "16"])
    check("tool_calls payload counted -> --max-seq-length emitted",
          r.returncode == 0 and flag_val(r.stdout, "--max-seq-length") is not None,
          r.stdout + r.stderr)

    # tools-format fixture (assistant content null) handled cleanly
    r = run(["--train-file", os.path.join(FIXTURES, "tools-chat", "train.jsonl"),
             "--model-size", "0.6b", "--memory-gb", "16"])
    check("tools-format fixture handled (chat fmt, exit 0, mask-prompt)",
          r.returncode == 0 and "(chat)" in r.stdout and "--mask-prompt" in r.stdout,
          r.stdout + r.stderr)

    # malformed --model-size exits cleanly, no traceback
    d10 = os.path.join(td, "c10")
    os.makedirs(d10)
    write_chat(os.path.join(d10, "train.jsonl"), 20)
    r = run(["--train-file", os.path.join(d10, "train.jsonl"),
             "--model-size", "800m", "--memory-gb", "16"])
    check("--model-size 800m exits non-zero with clean message",
          r.returncode != 0 and "model-size" in r.stderr and "Traceback" not in r.stderr,
          r.stderr)

    # unrecognized record format exits non-zero with pointer to validate_dataset
    d11 = os.path.join(td, "c11")
    os.makedirs(d11)
    with open(os.path.join(d11, "train.jsonl"), "w") as f:
        f.write(json.dumps({"foo": 1}) + "\n")
    r = run(["--train-file", os.path.join(d11, "train.jsonl"),
             "--model-size", "1.7b", "--memory-gb", "16"])
    check("garbage record exits non-zero with clear message",
          r.returncode != 0 and "unrecognized record format" in r.stderr, r.stderr)

if failures:
    print(f"\n{len(failures)} check(s) FAILED", file=sys.stderr)
    sys.exit(1)
print("\nall checks passed")
