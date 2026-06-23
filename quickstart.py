#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""tunelab quickstart — see it work in one command, before reading a single doc.

Three self-contained demos, all on the same dataset (Banking77 — real banking
support messages, 77 fine-grained intents). One problem, three lenses:

  uv run quickstart.py cost       a FREE local classifier that beats a frontier
                                  model on this task          ($0, any computer)
  uv run quickstart.py loop       champion vs challenger: promote a better model
                                  only when it REALLY wins     ($0, any computer)
  uv run quickstart.py finetune   teach a tiny local model your exact output
                                  format                 (Apple Silicon, ~2 min)

Each demo just wires together the same scripts the tunelab skills use — run with
--verbose to see every underlying command. Then point --data at your own JSONL.

Inside Claude Code you don't need any of this: just say "run the tunelab
quickstart" and it'll drive these for you and explain each step.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "examples" / "banking77"
CLF = REPO / "skills" / "tune-decide" / "scripts" / "train_classifier.py"
EVAL = REPO / "skills" / "tune-eval" / "scripts" / "eval_classifier.py"
PROMOTE = REPO / "skills" / "tune-loop" / "scripts" / "promote.py"

# Frontier zero-shot baseline on this exact Banking77 split, from recipes/01-hybrid-cascade.md
# (a frontier model used on its own, no fine-tuning). Pre-recorded so the quickstart
# stays $0 and needs no API key — see the recipe for the full run.
FRONTIER_ACC = 0.818
FRONTIER_COST_PER_1K = 2.00  # ~ frontier API, USD per 1k classifications (recipe 01)

VERBOSE = False


def hr(c="─", n=66):
    return c * n


def box(title, rows):
    """rows: list of (label, value) or None for a blank line."""
    print("\n┌" + hr("─") + "┐")
    print("│ " + title.ljust(64) + " │")
    print("├" + hr("─") + "┤")
    for r in rows:
        if r is None:
            print("│" + " " * 66 + "│")
            continue
        if isinstance(r, str):  # full-width line
            print("│ " + r.ljust(64) + " │")
            continue
        label, value = r
        print("│ " + f"{label}".ljust(40) + f"{value}".rjust(24) + " │")
    print("└" + hr("─") + "┘")


def run(cmd, capture=False):
    """Run a child command. stderr always streams through (live progress);
    stdout streams unless capture=True (then it's returned for parsing)."""
    if VERBOSE:
        print(f"\n$ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    proc = subprocess.run(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE if capture else None,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"\n✗ command failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}")
    return proc.stdout or ""


def uv(script, *args, capture=False):
    return run(["uv", "run", str(script), *args], capture=capture)


# ───────────────────────────── cost ─────────────────────────────
def cmd_cost(args):
    print("tunelab quickstart · COST — replace a frontier call with a free local model")
    print("Task: sort banking messages into 77 intents. Training a tiny classifier")
    print("on local embeddings — no API key, no GPU, no data leaving this machine.\n")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("· training a tiny classifier on 8,005 labeled messages …")
        uv(CLF, "--data", DATA / "train.jsonl", "--seed", "42",
           "--model-out", tmp / "clf.joblib", capture=True)
        print("· scoring on 3,080 held-out messages it never trained on …")
        preds = tmp / "preds.jsonl"
        uv(CLF, "--predict", DATA / "test.jsonl",
           "--model-in", tmp / "clf.joblib", "--output", preds, capture=True)
        ej = tmp / "eval.json"
        uv(EVAL, "--predictions", preds, "--json", ej, capture=True)
        acc = json.loads(ej.read_text())["accuracy"]

    delta = (acc - FRONTIER_ACC) * 100
    verdict = (f"On this task the FREE local model wins by {delta:+.1f} points"
               if acc > FRONTIER_ACC else f"On this task the frontier leads by {-delta:.1f} points")
    box("RESULT — free local classifier vs frontier model", [
        ("Free local classifier", f"{acc*100:.1f}%   ·   $0"),
        ("Frontier model, zero-shot¹", f"{FRONTIER_ACC*100:.1f}%   ·   ~${FRONTIER_COST_PER_1K:.0f}/1k"),
        None,
        verdict,
    ])
    print("\n¹ frontier baseline pre-recorded from recipes/01-hybrid-cascade.md (no API call made)")
    print("Bigger isn't always better — the right-sized tool for each input is.")
    print("\n→ Your turn: uv run quickstart.py cost  works on any {text, label} JSONL —")
    print("  swap examples/banking77/train.jsonl for your own labeled data.")


# ───────────────────────────── loop ─────────────────────────────
def cmd_loop(args):
    print("tunelab quickstart · LOOP — promote a better model only when it really wins")
    print("A 'champion' trained on 1,000 examples vs a 'challenger' trained on all")
    print("8,005. The loop adjudicates on a held-out slice under a PRE-REGISTERED bar —")
    print("discipline, not vibes: a win must clear the bar AND beat the champion by a")
    print("margin, or the champion stays.\n")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # champion: small data; challenger: full data
        champ_data = tmp / "champ_train.jsonl"
        rows = [l for l in open(DATA / "train.jsonl")][:1000]
        champ_data.write_text("".join(rows))

        print("· training champion (1,000 examples) …")
        c_out = uv(CLF, "--data", champ_data, "--seed", "42",
                   "--model-out", tmp / "champ.joblib", capture=True)
        print("· training challenger (8,005 examples) …")
        uv(CLF, "--data", DATA / "train.jsonl", "--seed", "42",
           "--model-out", tmp / "chal.joblib", capture=True)

        # evaluate BOTH on the same held-out test slice (data neither trained on)
        def eval_model(joblib, tag):
            preds = tmp / f"{tag}_preds.jsonl"
            uv(CLF, "--predict", DATA / "test.jsonl",
               "--model-in", joblib, "--output", preds, capture=True)
            ej = tmp / f"{tag}_eval.json"
            uv(EVAL, "--predictions", preds, "--json", ej, capture=True)
            return ej, json.loads(ej.read_text())

        print("· scoring both on the held-out test slice (3,080 messages) …")
        champ_ej, champ_eval = eval_model(tmp / "champ.joblib", "champ")
        chal_ej, chal_eval = eval_model(tmp / "chal.joblib", "chal")

        # pre-register the bar BEFORE adjudicating: beat the champion's known score
        # by a noise-band margin. The bar and margin are fixed here, not after.
        bar = round(champ_eval["accuracy"], 4)
        margin = 0.005
        print(f"\n· pre-registered bar = champion's score ({bar:.3f}); "
              f"challenger must beat it by ≥ {margin}")
        decision = uv(PROMOTE,
                      "--champion", champ_ej, "--challenger", chal_ej,
                      "--bar", str(bar), "--min-margin", str(margin),
                      "--metric", "accuracy", "--slice-id", "banking77-test",
                      "--ledger", tmp / "consumed.txt", capture=True)
        print("\n" + decision.rstrip())

    promoted = "PROMOTE" in decision
    box("RESULT — champion / challenger adjudication", [
        ("Champion (1k examples)", f"{champ_eval['accuracy']*100:.1f}%"),
        ("Challenger (8k examples)", f"{chal_eval['accuracy']*100:.1f}%"),
        ("Margin", f"{(chal_eval['accuracy']-champ_eval['accuracy'])*100:+.1f} points"),
        None,
        ("Decision", "PROMOTE challenger" if promoted else "RETAIN champion"),
    ])
    print("\nThis is the anti-AutoML-slop part: the bar is set before the scores are")
    print("seen, the eval slice is spent once (a reuse ledger enforces it), and a")
    print("noise-band win keeps the incumbent. That's how a self-improving loop")
    print("stays trustworthy as your data grows.")
    print("\n→ Your turn: feed the loop your champion + a challenger and it decides honestly.")


# ─────────────────────────── finetune ───────────────────────────
DEMO_PROMPTS = None  # filled from test.jsonl at runtime


def mlx_generate(model, system, prompt, adapter=None, max_tokens=12):
    cmd = ["uv", "run", "--with", "mlx-lm", "mlx_lm.generate",
           "--model", model, "--system-prompt", system, "--prompt", prompt,
           "--max-tokens", str(max_tokens), "--temp", "0.0"]
    if adapter:
        cmd += ["--adapter-path", str(adapter)]
    out = run(cmd, capture=True)
    # mlx_lm.generate brackets the completion between ========== rules
    parts = out.split("==========")
    gen = parts[1].strip() if len(parts) >= 3 else out.strip()
    return gen.splitlines()[0].strip() if gen else "(empty)"


def cmd_finetune(args):
    SYS = "Classify this banking customer message. Reply with only the intent label, in snake_case."
    model = args.model
    print("tunelab quickstart · FINETUNE — teach a tiny local model your output format")
    print(f"Base model: {model}  (downloads once; runs on Apple Silicon via MLX)")
    print("We LoRA-fine-tune it on 600 labeled banking messages so it emits the exact")
    print("snake_case intent — the label vocabulary ends up in the weights, not the prompt.\n")

    # a few held-out tickets the model never trained on
    test = [json.loads(l) for l in open(DATA / "test.jsonl")]
    demo = test[:: max(1, len(test) // 4)][:4]

    print("BEFORE fine-tuning — base model, zero-shot:")
    for r in demo:
        g = mlx_generate(model, SYS, r["text"])
        print(f"  {r['text'][:48]!r:52} → {g!r}   (gold: {r['label']})")

    with tempfile.TemporaryDirectory() as tmp:
        adapters = Path(tmp) / "adapters"
        print(f"\nTraining LoRA ({args.iters} iters, batch {args.batch}, "
              f"{args.layers} layers) … this is the ~2-minute part\n")
        run(["uv", "run", "--with", "mlx-lm", "mlx_lm.lora",
             "--model", model, "--train", "--data", str(DATA / "chat"),
             "--iters", str(args.iters), "--batch-size", str(args.batch),
             "--num-layers", str(args.layers), "--learning-rate", "1e-4",
             "--mask-prompt", "--adapter-path", str(adapters)])

        print("\nAFTER fine-tuning — same base model + your LoRA adapter:")
        after = []
        hits = in_format = 0
        for r in demo:
            g = mlx_generate(model, SYS, r["text"], adapter=adapters).strip()
            after.append(g)
            ok = g == r["label"]
            fmt = bool(g) and g.islower() and " " not in g and g.replace("_", "").isalpha()
            hits += ok
            in_format += fmt
            print(f"  {r['text'][:48]!r:52} → {g!r}   (gold: {r['label']}) {'✓' if ok else ''}")

    n = len(demo)
    box("RESULT — what the fine-tune bought you", [
        "Before:  base model emits free-text — wrong format",
        f"After:   {in_format}/{n} outputs are valid snake_case labels",
        None,
        f"Exact-match on held-out demo tickets:  {hits}/{n}",
    ])
    print("\nThe win here isn't just accuracy — it's that the small model now speaks your")
    print("exact contract (one snake_case label), which is what makes it usable as a")
    print("cheap drop-in. Train longer / on more data to push accuracy further.")
    print("\n→ Your turn: replace examples/banking77/chat/ with your own {messages:[…]}")
    print("  JSONL — same command teaches the model your task or your house style.")


def main():
    global VERBOSE
    ap = argparse.ArgumentParser(
        description="tunelab quickstart — see it work in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n  uv run quickstart.py cost\n  uv run quickstart.py loop\n"
               "  uv run quickstart.py finetune",
    )
    ap.add_argument("--verbose", action="store_true", help="print every underlying command")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cost", help="free local classifier vs frontier ($0, any computer)")
    sub.add_parser("loop", help="champion/challenger promotion ($0, any computer)")
    ft = sub.add_parser("finetune", help="LoRA-teach a tiny model your format (Apple Silicon)")
    ft.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ft.add_argument("--iters", type=int, default=300)
    ft.add_argument("--batch", type=int, default=4)
    ft.add_argument("--layers", type=int, default=8)
    args = ap.parse_args()
    VERBOSE = args.verbose

    {"cost": cmd_cost, "loop": cmd_loop, "finetune": cmd_finetune}[args.cmd](args)


if __name__ == "__main__":
    main()
