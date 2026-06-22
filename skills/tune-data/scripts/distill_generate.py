#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic>=0.92", "openai>=2.41"]
# ///
"""Distill a teacher model's outputs into training data.

Providers (--provider; only the selected provider's key and SDK are used):
  anthropic (default)  needs ANTHROPIC_API_KEY  model default claude-opus-4-8
  openai               needs OPENAI_API_KEY     model default gpt-5.5

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
counted at the end), never written to the outputs. If the first 5 calls of a
run all fail with API errors, the run aborts — check --model/--provider
instead of burning a full run of round-trips.
Run with --limit 25 first and read every output before scaling up.

Generate-mode asymmetry: the openai path skips incomplete/truncated
generations (status != completed); the anthropic path keeps whatever text a
max_tokens-stopped response produced — spot-check anthropic generate outputs
for truncation (or raise --max-tokens).

Teacher ToS posture: most frontier providers restrict using their outputs to
train competing models — see the distillation concepts note bundled with the
tunelab plugin. Which provider's clause matters depends on what you're
training, and some users pick their teacher (an OpenAI one instead of an
Anthropic one, or vice versa) on exactly that posture. The call is yours;
the DATACARD records teacher + intended use.
"""

import argparse
import json
import os
import sys

PROVIDER_MODELS = {"anthropic": "claude-opus-4-8", "openai": "gpt-5.5"}
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
# A typo'd --model/--provider fails every call; abort once the first N calls
# of a run have all raised api errors instead of skip-warning through the lot.
ABORT_AFTER_FAILS = 5


class ProviderRefusal(Exception):
    """No usable text in a response (refusal, token cap, ...). Carries the
    usage already spent so the caller can account for it before skipping."""

    def __init__(self, reason, in_tok=0, out_tok=0):
        super().__init__(reason)
        self.in_tok, self.out_tok = in_tok, out_tok


def make_client(provider):
    """-> (client, base API-error class). SDK imports are lazy so the other
    provider's package is never imported (and never needs shimming in tests)."""
    if provider == "openai":
        import openai

        return openai.OpenAI(max_retries=5), openai.APIError
    import anthropic

    return anthropic.Anthropic(max_retries=5), anthropic.APIError


def openai_reasoning_supported(model):
    """The Responses `reasoning` param is only accepted by reasoning models
    (GPT-5.x, o-series). Sending it to gpt-4o / gpt-4.1 / etc. is a hard 400 —
    which would block probing a non-reasoning incumbent (often the user's own
    gpt-4o) as the ceiling. Gate on the model id so any OpenAI model works."""
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def call_teacher(provider, client, model, system, user, schema, max_tokens):
    """One teacher call -> (text, in_tok, out_tok).

    Raises ProviderRefusal when the response has no usable text; provider API
    errors propagate for the caller's except clause. `schema` (or None) is a
    plain JSON schema dict — strict-mode-compatible (additionalProperties
    false, all properties required) so it works verbatim on both providers.
    """
    if provider == "openai":
        kwargs = dict(
            model=model,
            instructions=system,
            input=user,
            max_output_tokens=max_tokens,
            # Responses API persists responses for 30 days by default.
            store=False,
        )
        if openai_reasoning_supported(model):
            # Cost-predictable labeling: GPT-5.x reasoning bills as output.
            kwargs["reasoning"] = {"effort": "none"}
        if schema:
            kwargs["text"] = {"format": {
                "type": "json_schema", "name": "label", "schema": schema, "strict": True,
            }}
        resp = client.responses.create(**kwargs)
        # Response.usage is Optional in the SDK; a usage-less response
        # under-counts to 0 rather than crashing the whole run.
        usage = getattr(resp, "usage", None)
        in_tok = usage.input_tokens if usage else 0
        out_tok = usage.output_tokens if usage else 0
        refusal = next(
            (part.refusal for item in resp.output if item.type == "message"
             for part in item.content if part.type == "refusal"), None)
        if refusal is not None:
            raise ProviderRefusal(f"refusal: {refusal}", in_tok, out_tok)
        if resp.status != "completed":
            why = f"status={resp.status}"
            details = getattr(resp, "incomplete_details", None)
            if details is not None:
                why += f" ({details.reason})"
            raise ProviderRefusal(why, in_tok, out_tok)
        if not resp.output_text:
            raise ProviderRefusal("empty output", in_tok, out_tok)
        return resp.output_text, in_tok, out_tok

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if schema:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    resp = client.messages.create(**kwargs)
    in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise ProviderRefusal(f"stop_reason={resp.stop_reason}", in_tok, out_tok)
    return text, in_tok, out_tok


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
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                    help="teacher API provider")
    ap.add_argument("--model", default=None,
                    help="teacher model (default per provider: claude-opus-4-8 / gpt-5.5; "
                         "cheaper teachers: claude-sonnet-4-6 / gpt-5.4-mini, or "
                         "gpt-5.4-nano for easy label sets)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="teacher max output tokens (default: 1024 in generate mode, 256 in classify mode)")
    ap.add_argument("--limit", type=int, help="process at most N new records (spot-check first!)")
    args = ap.parse_args()

    if args.model is None:
        args.model = PROVIDER_MODELS[args.provider]
    key_var = PROVIDER_KEYS[args.provider]
    if not os.environ.get(key_var):
        sys.exit(f"{key_var} is not set")
    if args.mode == "classify" and not args.labels:
        sys.exit("--labels is required in classify mode")

    client, api_error = make_client(args.provider)
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
        schema = {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": labels}},
            "required": ["label"],
            "additionalProperties": False,
        }
        max_tokens = args.max_tokens or 256
    else:
        system, schema, max_tokens = args.system, None, args.max_tokens or 1024

    in_tok = out_tok = written = skipped = api_err_streak = 0
    ensure_trailing_newline(args.train_out)
    with open(args.output, "a", encoding="utf-8") as raw_f, \
         open(args.train_out, "a", encoding="utf-8") as train_f:
        for n, rec in enumerate(todo, 1):
            text = rec[args.input_key]
            try:
                answer, it, ot = call_teacher(
                    args.provider, client, args.model, system, text, schema, max_tokens)
            except api_error as e:
                print(f"  id={rec['id']} skipped (api error: {e})", file=sys.stderr)
                skipped += 1
                api_err_streak += 1
                if api_err_streak == n == ABORT_AFTER_FAILS:
                    sys.exit(
                        f"aborting: first {ABORT_AFTER_FAILS} calls all failed with api errors "
                        f"— last error: {e} "
                        f"(model={args.model}, provider={args.provider} — if the model id is "
                        f"valid, the error above is the real cause); "
                        f"records already in --output stay resumable")
                continue
            except ProviderRefusal as e:
                # Tokens are spent even on refusals — account before skipping.
                in_tok += e.in_tok
                out_tok += e.out_tok
                print(f"  id={rec['id']} skipped ({e})", file=sys.stderr)
                skipped += 1
                continue
            in_tok += it
            out_tok += ot

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
