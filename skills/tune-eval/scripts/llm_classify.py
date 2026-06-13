#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm>=0.31", "numpy"]
# ///
"""Classify with a local MLX model (base or LoRA-tuned) and emit routing-grade
confidence signals per record.

This is the cascade's middle tier. An LR classifier gives calibrated
probabilities for free; an LLM does not — so this script captures the raw
signals that the composition step (cascade_compose.py) calibrates on
validation data:

  conf_margin   mean per-step probability gap between the top-1 and top-2
                token while generating the label (token-margin aggregation —
                the signal that calibrates best for cascade routing)
  conf_logprob  mean log-probability of the generated tokens (fluency-ish;
                weaker but free)

Predictions are constrained to a label list after the fact: the generated
text is normalized and matched against --labels; a miss is flagged
(out_of_label_space: true) and counts as wrong downstream — an LLM that can't
emit a valid label must not be trusted by the router.

Generation is greedy (temp 0) for determinism. Thinking-style chat templates
(Qwen3 non-2507) are disabled via enable_thinking=False and any residual
<think> block is stripped before label matching.

Train-side note: this script never trains — pass --adapter-path to score a
LoRA checkpoint produced by mlx_lm.lora.

  uv run llm_classify.py --model mlx-community/Qwen3-0.6B-4bit \
    --data valid.jsonl --labels labels.json --output tier2_valid_preds.jsonl
  # tuned:  add --adapter-path runs/<id>/adapters
  # smoke:  add --limit 25

  data lines:   {"id": "...", "text": "...", "label": "..."}  (label optional)
  labels.json:  ["label_a", "label_b", ...]
  output lines: input record + {"predicted", "raw_output", "conf_margin",
                "conf_logprob", "latency_ms", "out_of_label_space"}
"""

import argparse
import json
import re
import sys
import time


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def match_label(raw: str, by_norm: dict):
    """Exact normalized match first, then unique-prefix, else None."""
    text = THINK_RE.sub("", raw).strip()
    norm = normalize(text)
    if norm in by_norm:
        return by_norm[norm], False
    hits = [v for k, v in by_norm.items() if k.startswith(norm) or norm.startswith(k)]
    if len(set(hits)) == 1:
        return hits[0], False
    return None, True


def build_prompt(tokenizer, labels, text, system_tpl):
    system = system_tpl.format(labels="\n".join(f"- {l}" for l in labels))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # template without thinking support
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


DEFAULT_SYSTEM = (
    "You are a classifier. Assign the user message to exactly one label from "
    "this list:\n{labels}\nReply with the label only — no explanation, no "
    "punctuation, nothing else."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True, help="JSON file: list of allowed labels")
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--system-file", default=None,
                    help="custom system prompt file; {labels} placeholder supported")
    args = ap.parse_args()

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    labels = json.load(open(args.labels))
    by_norm = {normalize(l): l for l in labels}
    if len(by_norm) != len(labels):
        eprint("WARNING: labels collide after normalization — matching may be lossy")
    system_tpl = open(args.system_file).read() if args.system_file else DEFAULT_SYSTEM

    eprint(f"loading {args.model}" + (f" + adapters {args.adapter_path}" if args.adapter_path else ""))
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    sampler = make_sampler(temp=0.0)

    records = [json.loads(l) for l in open(args.data)]
    if args.limit:
        records = records[: args.limit]

    n_oolm = 0
    correct = total_gold = 0
    with open(args.output, "w") as out:
        for i, rec in enumerate(records):
            prompt = build_prompt(tokenizer, labels, rec[args.text_key], system_tpl)
            t0 = time.perf_counter()
            margins, logprobs, text_out = [], [], []
            for resp in stream_generate(
                model, tokenizer, prompt, max_tokens=args.max_tokens, sampler=sampler
            ):
                text_out.append(resp.text)
                # logprobs may be bf16 on MLX — cast to f32 before numpy
                lp = np.array(resp.logprobs.astype(mx.float32))
                # top-2 probability margin at this step
                top2 = np.partition(lp, -2)[-2:]
                margins.append(float(np.exp(top2[1]) - np.exp(top2[0])))
                logprobs.append(float(lp[resp.token]))
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = "".join(text_out)
            pred, oolm = match_label(raw, by_norm)
            n_oolm += oolm
            row = dict(rec)
            row.update(
                predicted=pred,
                raw_output=raw,
                conf_margin=float(np.mean(margins)) if margins else 0.0,
                conf_logprob=float(np.mean(logprobs)) if logprobs else -99.0,
                latency_ms=round(latency_ms, 1),
                out_of_label_space=bool(oolm),
            )
            out.write(json.dumps(row) + "\n")
            if "label" in rec:
                total_gold += 1
                correct += pred == rec["label"]
            if (i + 1) % 50 == 0:
                eprint(f"  {i+1}/{len(records)}"
                       + (f"  running-acc {correct/total_gold:.3f}" if total_gold else ""))

    eprint(f"wrote {len(records)} predictions -> {args.output}")
    eprint(f"out-of-label-space: {n_oolm}/{len(records)}")
    if total_gold:
        eprint(f"accuracy vs gold: {correct}/{total_gold} = {correct/total_gold:.4f}")
        eprint("(conf_margin is RAW — calibrate it on validation via cascade_compose.py; "
               "do not threshold it by eye)")


if __name__ == "__main__":
    main()
