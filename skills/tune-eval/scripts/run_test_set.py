#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm>=0.21"]
# ///
"""Run an MLX model over a held-out test set and write predictions.

  uv run run_test_set.py --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
      --adapter-path adapters/ --test-file data/test.jsonl --output preds_tuned.jsonl

Omit --adapter-path to get the base-model control run.
Supports chat ({"messages": [...]}) and completions ({"prompt","completion"}) formats.
Output lines: {"id", "input", "expected", "predicted"}

Thinking-mode models (e.g. Qwen3 hybrid chat templates): enable_thinking=False
is passed to the chat template by default so the model answers directly instead
of reasoning in <think> blocks first; templates without the variable ignore it.
Reasoning that leaks through anyway is stripped from 'predicted': matched
<think>...</think> blocks, the bare-</think> form from templates that pre-open
the block inside the prompt (Qwen3.5 style), and truncated reasoning that
never closed. Use --enable-thinking to opt back in — and raise --max-tokens to
1024+, since reasoning routinely burns >512 tokens before the answer even on
trivial tasks. Caveat: on pre-opened templates, truncated reasoning carries no
marker at all and cannot be detected — 'predicted' will be raw reasoning text
(eval_classifier surfaces these as hallucinated labels).
"""

import argparse
import json
import re
import sys

from mlx_lm import generate, load

# Some models emit <think> blocks even when the template is told not to —
# strip them defensively so 'predicted' is always the answer text.
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text):
    """Remove leaked reasoning so 'predicted' is always the answer text."""
    text = THINK_RE.sub("", text)
    # Qwen3.5-style templates pre-open '<think>\n' inside the generation
    # prompt, so the output has only a closing tag — keep what follows it.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    # Truncated reasoning that never closed: drop from the unmatched opener on.
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def split_record(rec, idx):
    """Return (prompt_messages, expected) — everything before the final assistant turn."""
    if "messages" in rec:
        msgs = rec["messages"]
        if msgs and msgs[-1].get("role") == "assistant":
            return msgs[:-1], str(msgs[-1]["content"])
        return msgs, ""
    if "prompt" in rec:
        return [{"role": "user", "content": rec["prompt"]}], str(rec.get("completion", ""))
    raise SystemExit(f"record {idx}: unrecognized format — need 'messages' or 'prompt', got keys {sorted(rec)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path")
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="let hybrid-thinking templates emit <think> reasoning (default: disabled)")
    args = ap.parse_args()

    with open(args.test_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        records = records[: args.limit]

    # Validate every record up front — the model load below is the slow step.
    parsed = [split_record(rec, i) for i, rec in enumerate(records)]

    print(f"loading {args.model}" + (f" + {args.adapter_path}" if args.adapter_path else " (base)"), file=sys.stderr)
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    with open(args.output, "w") as f:
        for i, (rec, (msgs, expected)) in enumerate(zip(records, parsed)):
            prompt = tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=args.enable_thinking,
            )
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
            predicted = strip_thinking(raw)
            user_input = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            f.write(json.dumps({
                "id": rec.get("id", i),
                "input": user_input,
                "expected": expected,
                "predicted": predicted,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(records)}", file=sys.stderr)

    print(f"wrote {len(records)} predictions -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
