#!/usr/bin/env python3
"""Validate an MLX-LM training data directory before training. Stdlib only.

  python3 validate_dataset.py --data-dir data/ [--max-tokens-warn 2048]

Checks train/valid/test.jsonl: every line parses, format is one of the four
mlx-lm 0.31.3 auto-detects — chat / tools (chat messages with tool_calls plus
a top-level "tools" array) / completions / text — and is consistent within and
across splits (chat/tools mixing is only a warning: mlx-lm has no separate
tools dataset class — ChatDataset reads "tools" per record, so the mix trains
fine), chat roles are sane, content is non-empty, and records aren't
longer than mlx_lm's --max-seq-length default (2048 tokens, approximated from
the content text at chars/4). Prints counts and (for classification-shaped
data) the label distribution.

valid.jsonl is optional in mlx-lm 0.31.3 (training proceeds with a printed
warning), but without it validation-loss monitoring is unavailable — and
tune-train's overfitting detection relies on it. Flagged here as a warning.

Exit code is non-zero on errors: DO NOT TRAIN until it passes. mlx_lm's own
failure mode for malformed data is a cryptic error minutes into a run.
"""

import argparse
import json
import os
import sys


def detect_format(rec):
    # Order mirrors mlx-lm 0.31.3 create_dataset (prompt+completion, then
    # messages, then text) so a record carrying several shapes validates as
    # whatever mlx-lm will actually train it as.
    if "prompt" in rec and "completion" in rec:
        return "completions"
    if "messages" in rec:
        # tools = chat plus a top-level "tools" array; distinct format in
        # mlx-lm 0.31.3 (rendered with tool definitions in the template).
        return "tools" if "tools" in rec else "chat"
    if "text" in rec:
        return "text"
    return None


def content_text(rec, fmt):
    # Token estimate from the content the template renders, not the raw JSON
    # dump — keys/punctuation inflate the count and trip false length warnings.
    if fmt in ("chat", "tools"):
        parts = []
        for m in rec.get("messages", []):
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
            if m.get("tool_calls"):
                parts.append(json.dumps(m["tool_calls"]))
        if fmt == "tools":
            parts.append(json.dumps(rec.get("tools", [])))
        return "\n".join(parts)
    if fmt == "completions":
        s = str(rec.get("prompt", "")) + str(rec.get("completion", ""))
        # CompletionsDataset also renders a per-record "tools" array.
        if "tools" in rec:
            s += json.dumps(rec["tools"])
        return s
    return str(rec.get("text", ""))


def approx_tokens(rec, fmt):
    return len(content_text(rec, fmt)) // 4


def validate_file(path, max_tokens_warn):
    errors, warnings, labels = [], [], []
    fmt = None
    n = 0
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {lineno}: invalid JSON ({e})")
                continue
            this_fmt = detect_format(rec)
            if this_fmt is None:
                errors.append(f"line {lineno}: no messages/prompt+completion/text key")
                continue
            fmt = fmt or this_fmt
            if this_fmt != fmt:
                if {this_fmt, fmt} == {"chat", "tools"}:
                    # Not an mlx-lm error: ChatDataset reads "tools" per
                    # record, so the mix trains — flag as a data smell only.
                    warnings.append(
                        f"line {lineno}: {this_fmt} record in a {fmt} file — mlx-lm accepts "
                        "the mix, but check it's intentional"
                    )
                else:
                    errors.append(f"line {lineno}: format {this_fmt} != file format {fmt}")
                    continue

            if this_fmt in ("chat", "tools"):
                msgs = rec["messages"]
                if not isinstance(msgs, list):
                    errors.append(f"line {lineno}: messages is not a list")
                    continue
                if any(not isinstance(m, dict) for m in msgs):
                    errors.append(f"line {lineno}: messages contains a non-object entry")
                    continue
                ok_roles = ("system", "user", "assistant") if this_fmt == "chat" else ("system", "user", "assistant", "tool")
                roles = [m.get("role") for m in msgs]
                if not msgs or roles[-1] != "assistant":
                    errors.append(f"line {lineno}: messages must end with an assistant turn")
                if "user" not in roles:
                    errors.append(f"line {lineno}: no user turn")
                for m in msgs:
                    if m.get("role") not in ok_roles:
                        errors.append(f"line {lineno}: bad role {m.get('role')!r}")
                    # An assistant turn carrying tool_calls legitimately has no content.
                    if not str(m.get("content") or "").strip() and not (
                        m.get("role") == "assistant" and m.get("tool_calls")
                    ):
                        errors.append(f"line {lineno}: empty {m.get('role')} content")
                    if this_fmt == "chat" and m.get("tool_calls"):
                        warnings.append(
                            f"line {lineno}: tool_calls present but no top-level 'tools' array — "
                            "add one to use the tools format"
                        )
                if this_fmt == "tools" and not isinstance(rec.get("tools"), list):
                    errors.append(f"line {lineno}: 'tools' must be a list of tool definitions")
                if this_fmt == "chat":
                    assistant = next((m.get("content") for m in reversed(msgs) if m.get("role") == "assistant"), "")
                    if isinstance(assistant, str) and len(assistant) <= 40:
                        labels.append(assistant.strip())
            elif this_fmt == "completions":
                if not str(rec["prompt"]).strip() or not str(rec["completion"]).strip():
                    errors.append(f"line {lineno}: empty prompt or completion")
            elif this_fmt == "text":
                if not str(rec["text"]).strip():
                    errors.append(f"line {lineno}: empty text")

            tok = approx_tokens(rec, this_fmt)
            if tok > max_tokens_warn:
                warnings.append(
                    f"line {lineno}: ~{tok} tokens exceeds mlx_lm --max-seq-length default 2048; "
                    "raise --max-seq-length when training or shorten"
                )
    return n, fmt, errors, warnings, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    # 2048 matches mlx_lm's --max-seq-length default; longer records get truncated.
    ap.add_argument("--max-tokens-warn", type=int, default=2048)
    args = ap.parse_args()

    any_errors = False
    formats = set()
    for name in ("train", "valid", "test"):
        path = os.path.join(args.data_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            if name == "train":
                print(f"ERROR: {path} missing")
                any_errors = True
            elif name == "valid":
                print(
                    f"warning: {path} missing — optional in mlx-lm 0.31.3 (training just warns), "
                    "but validation loss monitoring will be unavailable and tune-train relies on it"
                )
            else:
                print(f"warning: {path} missing (tune-eval needs it)")
            continue
        n, fmt, errors, warnings, labels = validate_file(path, args.max_tokens_warn)
        if n == 0:
            # mlx-lm 0.31.3 treats present-but-empty exactly like missing
            # (datasets.py: 'Training set not found or empty' / valid warning).
            print(f"\n{name}.jsonl: 0 records")
            if name == "train":
                print("  ERROR: empty — mlx-lm raises 'Training set not found or empty'")
                any_errors = True
            elif name == "valid":
                print(
                    "  warning: empty — optional in mlx-lm 0.31.3 (training just warns), "
                    "but validation loss monitoring will be unavailable and tune-train relies on it"
                )
            else:
                print("  warning: empty (tune-eval needs it)")
            continue
        formats.add(fmt)
        print(f"\n{name}.jsonl: {n} records, format={fmt}")
        for w in warnings[:5]:
            print(f"  warning: {w}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more warnings")
        for e in errors[:20]:
            print(f"  ERROR: {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        any_errors |= bool(errors)

        # Short assistant turns across >90% of records looks like classification —
        # show the class balance, since skew here predicts skewed predictions.
        if labels and len(labels) >= 0.9 * n:
            dist = {}
            for lb in labels:
                dist[lb] = dist.get(lb, 0) + 1
            print(f"  label distribution: {dict(sorted(dist.items(), key=lambda kv: -kv[1]))}")

    fmts = formats - {None}
    if len(fmts) > 1:
        if fmts == {"chat", "tools"}:
            print(
                "\nwarning: splits mix chat and tools formats — mlx-lm 0.31.3 reads 'tools' "
                "per record and accepts the mix, but check it's intentional"
            )
        else:
            print(f"\nERROR: splits disagree on format: {sorted(fmts)}")
            any_errors = True

    print("\nFAILED — fix errors before training" if any_errors else "\nOK — safe to train")
    sys.exit(1 if any_errors else 0)


if __name__ == "__main__":
    main()
