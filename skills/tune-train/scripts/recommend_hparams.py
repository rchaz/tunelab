#!/usr/bin/env python3
"""Recommend mlx_lm.lora hyperparameters from dataset stats. Stdlib only.

  python3 recommend_hparams.py --train-file data/train.jsonl \
      --model-size 1.7b --task sft --memory-gb 16 \
      [--model-id mlx-community/Qwen3.5-2B-4bit] [--lora-rank 16] [--outdir .]

Heuristics encoded (override freely when the case is unusual):
- iters = epochs * n / batch; epochs 5/3/2 as the dataset grows
- LoRA LR 1e-4 (<=4B) / 5e-5 (7-8B); CPT default 1e-5, sane range 5e-6..2e-5
  (lower end for small corpora / fragile bases); halve any LR if loss oscillates
- --num-layers 8 for tiny datasets (<300) to cut overfitting capacity
- --mask-prompt only for chat/completions SFT — NEVER for text/CPT
  (mlx-lm raises ValueError on text datasets)
- --save-every min(100, steps-per-eval) so numbered checkpoints exist for resume
- --grad-checkpoint when the model crowds RAM (>=8B on <=16GB, >=4B on <=8GB)
- --max-seq-length when the longest record approaches mlx-lm's 2048 default.
  Estimate = chars/4 over message content + tool_calls + the tools array; that
  undercounts chat-template overhead, so the recommendation fires from ~90% of
  2048 and rounds up to the next multiple of 1024 (minimum 3072)

LoRA rank/scale/dropout are YAML-config-only in mlx-lm 0.31.x (no CLI flags):
pass --lora-rank to get a lora_config.yaml plus a 'mlx_lm.lora -c' invocation
instead of the long flag form.
"""

import argparse
import json
import math
import os
import sys


def record_chars(rec, line):
    """Approximate content size of one training record in characters."""
    if "messages" in rec:
        total = 0
        for m in rec["messages"]:
            if isinstance(m, dict):
                c = m.get("content")
                total += len(c) if isinstance(c, str) else len(json.dumps(c)) if c else 0
                # assistant tool-call turns usually have content None — the
                # payload lives in tool_calls and is rendered into the prompt
                if m.get("tool_calls"):
                    total += len(json.dumps(m["tool_calls"]))
        # the top-level tools array is rendered into the prompt by the chat template
        if rec.get("tools"):
            total += len(json.dumps(rec["tools"]))
        return total
    if "text" in rec:
        return len(rec["text"]) if isinstance(rec["text"], str) else len(line)
    if "prompt" in rec:
        return len(str(rec.get("prompt", ""))) + len(str(rec.get("completion", "")))
    return len(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--model-size", required=True, help="e.g. 0.6b, 1.7b, 4b, 8b")
    ap.add_argument("--task", choices=["sft", "cpt"], default="sft")
    ap.add_argument("--memory-gb", type=int, default=16)
    ap.add_argument("--model-id", default="<mlx-community model id>")
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="set LoRA rank (YAML-config-only in mlx-lm) — emits lora_config.yaml + '-c' form")
    ap.add_argument("--outdir", default=".",
                    help="where lora_config.yaml is written when --lora-rank is given")
    args = ap.parse_args()

    n = 0
    fmt = None
    max_chars = 0
    with open(args.train_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            rec = json.loads(line)
            if fmt is None:
                if "messages" in rec:
                    fmt = "chat"
                elif "text" in rec:
                    fmt = "text"
                elif "prompt" in rec:
                    fmt = "completions"
                else:
                    sys.exit(f"unrecognized record format (keys: {sorted(rec)}) — "
                             "expected messages / text / prompt+completion; "
                             "run validate_dataset.py first")
            max_chars = max(max_chars, record_chars(rec, line))
    if n == 0:
        sys.exit("train file is empty")

    try:
        size_b = float(args.model_size.lower().rstrip("b"))
    except ValueError:
        sys.exit(f"--model-size {args.model_size!r} not understood — "
                 "use billions of parameters like '0.6b', '1.7b', '4b', '8b'")

    if args.task == "cpt":
        epochs, lr = 1.5, 1e-5
    elif n < 1000:
        epochs, lr = 5, 1e-4 if size_b <= 4 else 5e-5
    elif n < 5000:
        epochs, lr = 3, 1e-4 if size_b <= 4 else 5e-5
    else:
        epochs, lr = 2, 1e-4 if size_b <= 4 else 5e-5

    batch = 2 if args.memory_gb <= 8 else (4 if args.memory_gb <= 16 else 8)
    batch = max(1, min(batch, n // 10 or 1))
    iters = max(100, int(epochs * n / batch))
    num_layers = 8 if n < 300 else 16
    grad_checkpoint = (size_b >= 8 and args.memory_gb <= 16) or (size_b >= 4 and args.memory_gb <= 8)
    steps_per_eval = max(25, iters // 12)
    save_every = min(100, steps_per_eval)
    mask_prompt = fmt in ("chat", "completions") and args.task == "sft"
    data_dir = os.path.dirname(args.train_file) or "."

    # mlx_lm.lora reads <data>/train.jsonl — stats above came from a different
    # filename, so the emitted command would train on a DIFFERENT file.
    if os.path.basename(args.train_file) != "train.jsonl":
        print(f"WARNING: mlx_lm.lora trains on {os.path.join(data_dir, 'train.jsonl')}, "
              f"not {os.path.basename(args.train_file)!r} — rename or symlink your "
              "file to train.jsonl before running the command below.", file=sys.stderr)

    # mlx-lm truncates at --max-seq-length (default 2048); long records lose their
    # tails silently. chars/4 undercounts chat-template overhead and special
    # tokens, so fire from ~90% of the default and never recommend a value that
    # is not an actual raise (minimum 3072).
    approx_max_tokens = max_chars // 4
    max_seq_length = None
    if approx_max_tokens > int(2048 * 0.9):
        max_seq_length = max(3072, math.ceil(approx_max_tokens / 1024) * 1024)

    print(f"dataset: {n} records ({fmt}), model: {size_b}B, task: {args.task}")
    print(f"-> epochs ~{epochs}, batch {batch}, iters {iters}, lr {lr:g}, num-layers {num_layers}")
    if max_seq_length:
        print(f"-> longest record ~{approx_max_tokens} tokens (chars/4 estimate, which")
        print("   undercounts chat-template overhead) crowds the 2048 default —")
        print(f"   recommending --max-seq-length {max_seq_length} (next multiple of 1024,")
        print("   with headroom) so long examples are not silently truncated.")
    print()

    if args.lora_rank is not None:
        config_path = os.path.normpath(os.path.join(args.outdir, "lora_config.yaml"))
        os.makedirs(args.outdir, exist_ok=True)
        # Hand-rolled YAML: LR formatted as 1.0e-04 (dot + signed exponent) so
        # PyYAML resolves it as a float, not a string.
        cfg = [
            f"# generated by recommend_hparams.py — run: mlx_lm.lora -c {config_path}",
            "# rank/scale/dropout have no CLI flags in mlx-lm; this file is the only way to set them.",
            f'model: "{args.model_id}"',
            "train: true",
            f'data: "{data_dir}"',
            "fine_tune_type: lora",
            f"iters: {iters}",
            f"batch_size: {batch}",
            f"learning_rate: {lr:.1e}",
            f"num_layers: {num_layers}",
            "adapter_path: adapters",
            "steps_per_report: 20",
            f"steps_per_eval: {steps_per_eval}",
            "val_batches: 25",
            f"save_every: {save_every}",
            "seed: 42",
        ]
        if mask_prompt:
            cfg.append("mask_prompt: true")
        if grad_checkpoint:
            cfg.append("grad_checkpoint: true")
        if max_seq_length:
            cfg.append(f"max_seq_length: {max_seq_length}")
        cfg += [
            "lora_parameters:",
            "  # q/v-only keys are a deliberate tunelab restriction. mlx-lm's own",
            "  # default is NO keys entry, which adapts EVERY linear/embedding",
            "  # module in the trained blocks — drop the keys line to get that.",
            '  keys: ["self_attn.q_proj", "self_attn.v_proj"]',
            f"  rank: {args.lora_rank}",
            "  scale: 20.0",
            "  dropout: 0.0",
        ]
        with open(config_path, "w") as f:
            f.write("\n".join(cfg) + "\n")
        print(f"wrote {config_path} (LoRA rank {args.lora_rank} — YAML-config-only in mlx-lm)")
        print(f"\nmlx_lm.lora -c {config_path}")
    else:
        cmd = [
            "mlx_lm.lora",
            f"--model {args.model_id}",
            "--train",
            f"--data {data_dir}",
            f"--iters {iters}",
            f"--batch-size {batch}",
            f"--learning-rate {lr:g}",
            f"--num-layers {num_layers}",
            "--adapter-path adapters/",
            "--steps-per-report 20",
            f"--steps-per-eval {steps_per_eval}",
            "--val-batches 25",
            f"--save-every {save_every}",
            "--seed 42",
        ]
        if max_seq_length:
            cmd.append(f"--max-seq-length {max_seq_length}")
        if mask_prompt:
            cmd.append("--mask-prompt")
        if grad_checkpoint:
            cmd.append("--grad-checkpoint")
        print(" \\\n  ".join(cmd))

    print(f"""
RESUME (verified mlx-lm 0.31.3 behavior): checkpoints land in adapters/ as
NNNNNNN_adapters.safetensors every {save_every} iters (adapters.safetensors is
always the latest, overwritten). --resume-adapter-file restores WEIGHTS ONLY —
fresh optimizer state, iteration counter resets to 1 — so to resume an
interrupted run, do NOT re-run the full --iters:
  mlx_lm.lora ... --iters <total - completed> \\
    --resume-adapter-file adapters/<latest NNNNNNN>_adapters.safetensors
Expect a brief loss bump from the cold optimizer state.

Watch validation loss: when it bottoms out and climbs while train loss
falls, note that iteration and stop — that's your real --iters.""")


if __name__ == "__main__":
    main()
