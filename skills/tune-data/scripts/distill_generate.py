#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic>=0.92"]
# ///
"""Distill a teacher model's outputs into training data. Needs ANTHROPIC_API_KEY.

Classification (structured output guarantees a valid label):
  uv run distill_generate.py --mode classify --input inputs.jsonl \
    --labels "billing,receipt,spam,other" --system "You label inbound emails." \
    --output labeled.jsonl --train-out train_chat.jsonl

Generation (teacher produces the target output per input):
  uv run distill_generate.py --mode generate --input inputs.jsonl \
    --system "Draft a reply in our support voice." \
    --output labeled.jsonl --train-out train_chat.jsonl

  inputs.jsonl lines: {"text": "..."} (use --input-key for another field;
                      an "id" field is used for resume, else the line number)
  --output:    raw results {"id", "text", "label"|"generated"}
  --train-out: MLX chat-format training records, ready for dedupe/split

When this script is NOT the right tool: for small datasets (up to a few hundred
items) the session-native teacher tier is the Phase-1 default — the tune-data
skill labels them directly in-session, no API key required. This script is the
scale path (pinned model, structured-output guarantees, resumability).

Resumable: already-processed ids in --output are skipped on re-run. Resume
matches on the "id" field — if the input lacks ids, line numbers are used, so
the input file must not be reordered between runs.

Refusals and unparseable structured outputs are skipped (logged to stderr,
counted at the end), never written to the outputs.
Run with --limit 25 first and read every output before scaling up.
"""

import argparse
import json
import os
import sys

import anthropic


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_done_ids(path):
    # Resume read tolerates the artifacts of a mid-write kill (exactly what
    # resume exists to recover from): a truncated trailing line is dropped so
    # its record is retried, and a missing final newline is repaired so
    # appends start on a fresh line. Corruption anywhere else still raises.
    with open(path, encoding="utf-8") as f:
        raw = f.readlines()
    lines = [ln for ln in raw if ln.strip()]
    done = set()
    for i, line in enumerate(lines):
        try:
            done.add(json.loads(line)["id"])
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print(f"warning: ignoring truncated trailing line in {path} — dropped; its record will be retried",
                  file=sys.stderr)
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines[:-1])
            return done
    if raw and not raw[-1].endswith("\n"):
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")
    return done


def ensure_trailing_newline(path):
    # A mid-write kill can leave --train-out with a partial line and no
    # newline; make appended records start fresh (the partial line is then
    # caught by validate_dataset instead of silently merging two records).
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            with open(path, "ab") as f2:
                f2.write(b"\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["classify", "generate"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--input-key", default="text")
    ap.add_argument("--labels", help="comma-separated label set (classify mode)")
    ap.add_argument("--system", required=True, help="teacher system prompt — defines the task")
    ap.add_argument("--output", required=True, help="raw teacher outputs (JSONL, appended)")
    ap.add_argument("--train-out", required=True, help="chat-format training data (JSONL, appended)")
    ap.add_argument("--provider", choices=["anthropic"], default="anthropic",
                    help="teacher API provider (OpenAI lands in Phase 2)")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="teacher max output tokens (default: 1024 in generate mode, 256 in classify mode)")
    ap.add_argument("--limit", type=int, help="process at most N new records (spot-check first!)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")
    if args.mode == "classify" and not args.labels:
        sys.exit("--labels is required in classify mode")

    client = anthropic.Anthropic(max_retries=5)
    rows = read_jsonl(args.input)
    for i, r in enumerate(rows):
        r.setdefault("id", i)

    done = set()
    if os.path.exists(args.output):
        done = read_done_ids(args.output)
        print(f"resuming: {len(done)} already done", file=sys.stderr)

    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"processing {len(todo)} of {len(rows)} records with {args.model}", file=sys.stderr)

    labels = [s.strip() for s in args.labels.split(",") if s.strip()] if args.labels else None
    if args.mode == "classify":
        if not labels:
            sys.exit("--labels parsed to an empty label set")
        system = (
            f"{args.system}\n\nClassify the user's text into exactly one of these "
            f"labels: {', '.join(labels)}."
        )
        output_config = {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"label": {"type": "string", "enum": labels}},
                    "required": ["label"],
                    "additionalProperties": False,
                },
            }
        }
        max_tokens = args.max_tokens or 256
    else:
        system, output_config, max_tokens = args.system, None, args.max_tokens or 1024

    in_tok = out_tok = written = skipped = 0
    ensure_trailing_newline(args.train_out)
    with open(args.output, "a", encoding="utf-8") as raw_f, \
         open(args.train_out, "a", encoding="utf-8") as train_f:
        for n, rec in enumerate(todo, 1):
            text = rec[args.input_key]
            kwargs = dict(
                model=args.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": text}],
            )
            if output_config:
                kwargs["output_config"] = output_config
            try:
                resp = client.messages.create(**kwargs)
            except anthropic.APIError as e:
                print(f"  id={rec['id']} skipped (api error: {e})", file=sys.stderr)
                skipped += 1
                continue
            # Tokens are spent even on refusals — account before the skip checks.
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens
            answer = next((b.text for b in resp.content if b.type == "text"), None)
            if answer is None:
                print(f"  id={rec['id']} skipped (stop_reason={resp.stop_reason})", file=sys.stderr)
                skipped += 1
                continue

            if args.mode == "classify":
                try:
                    target = json.loads(answer)["label"]
                except (json.JSONDecodeError, KeyError):
                    print(f"  id={rec['id']} skipped (unparseable structured output)", file=sys.stderr)
                    skipped += 1
                    continue
                raw_f.write(json.dumps({"id": rec["id"], "text": text, "label": target}, ensure_ascii=False) + "\n")
            else:
                target = answer.strip()
                raw_f.write(json.dumps({"id": rec["id"], "text": text, "generated": target}, ensure_ascii=False) + "\n")

            train_f.write(json.dumps({"messages": [
                {"role": "system", "content": args.system},
                {"role": "user", "content": text},
                {"role": "assistant", "content": target},
            ]}, ensure_ascii=False) + "\n")
            # Flush train before raw: the raw record is the resume marker, so it
            # must never hit disk before its train record. A kill between the two
            # flushes then yields at worst a duplicate train record on retry
            # (dedupe eats it), never a silently missing one.
            train_f.flush()
            raw_f.flush()
            written += 1
            if n % 25 == 0:
                print(f"  {n}/{len(todo)} done ({in_tok} in / {out_tok} out tokens)", file=sys.stderr)

    print(
        f"done. {written} written, {skipped} skipped. "
        f"tokens: {in_tok} in / {out_tok} out. outputs: {args.output}, {args.train_out}",
        file=sys.stderr,
    )
    if skipped:
        print(f"warning: {skipped} records skipped (see stderr above) — re-run to retry them", file=sys.stderr)


if __name__ == "__main__":
    main()
