#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic>=0.92", "openai>=2.41"]
# ///
"""Pairwise LLM-as-judge for generative outputs.

Providers (--provider; only the selected provider's key and SDK are used):
  anthropic (default)  needs ANTHROPIC_API_KEY  model default claude-opus-4-8
  openai               needs OPENAI_API_KEY     model default gpt-5.5

  uv run judge_eval.py --a preds_base.jsonl --b preds_tuned.jsonl \
      --criteria "Faithful to the task; matches the reference voice; no fabricated facts" \
      --output verdicts.jsonl

Inputs are run_test_set.py outputs ({"id", "input", "expected", "predicted"})
joined on id. Each pair is judged blind with A/B order randomized per item
(judges have position bias) and a structured-output verdict. Reports win/tie/
loss for B (your tuned model) vs A.

Refused or unparseable verdicts are skipped (logged to stderr, counted in the
summary), not tallied. If the first 5 calls of a run all fail with API errors,
the run aborts — check --model/--provider instead of burning a full run of
round-trips. With ~100 items, differences under ~10 points are noise
— the summary says so whenever fewer than 150 pairs were judged.
"""

import argparse
import json
import os
import random
import sys

PROVIDER_MODELS = {"anthropic": "claude-opus-4-8", "openai": "gpt-5.5"}
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
# A typo'd --model/--provider fails every call; abort once the first N calls
# of a run have all raised api errors instead of skip-warning through the lot.
ABORT_AFTER_FAILS = 5

# Strict-mode-compatible on both providers: additionalProperties false and
# every property required (what OpenAI's strict json_schema demands).
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["first", "second", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["winner", "reason"],
    "additionalProperties": False,
}


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


def call_judge(provider, client, model, system, user, schema):
    """One judge call -> (verdict JSON text, in_tok, out_tok).

    Raises ProviderRefusal when the response has no usable text; provider API
    errors propagate for the caller's except clause.
    """
    if provider == "openai":
        resp = client.responses.create(
            model=model,
            instructions=system,
            input=user,
            # Reasoning tokens share the output budget (and bill as output) —
            # same headroom rationale as the anthropic path's thinking budget.
            max_output_tokens=8192,
            reasoning={"effort": "low"},
            # Responses API persists responses for 30 days by default.
            store=False,
            text={"format": {
                "type": "json_schema", "name": "verdict", "schema": schema, "strict": True,
            }},
        )
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
                why += f" ({details.reason} — reasoning may have eaten the token budget)"
            raise ProviderRefusal(why, in_tok, out_tok)
        if not resp.output_text:
            raise ProviderRefusal("empty output", in_tok, out_tok)
        return resp.output_text, in_tok, out_tok

    resp = client.messages.create(
        model=model,
        # Adaptive-thinking tokens count against max_tokens; the
        # verdict itself is tiny, so the headroom is for thinking.
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
    text = next((blk.text for blk in resp.content if blk.type == "text"), None)
    if text is None:
        why = f"stop_reason={resp.stop_reason}"
        if resp.stop_reason == "max_tokens":
            why += " — thinking hit the token cap"
        raise ProviderRefusal(why, in_tok, out_tok)
    return text, in_tok, out_tok


def read_preds(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="predictions A (e.g. base model)")
    ap.add_argument("--b", required=True, help="predictions B (e.g. tuned model)")
    ap.add_argument("--criteria", required=True, help="what 'better' means for this task")
    ap.add_argument("--output", required=True)
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                    help="judge API provider")
    ap.add_argument("--model", default=None,
                    help="judge model (default per provider: claude-opus-4-8 / gpt-5.5 — "
                         "the flagship tiers; mid-tier judges: claude-sonnet-4-6 / gpt-5.4)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.model is None:
        args.model = PROVIDER_MODELS[args.provider]
    key_var = PROVIDER_KEYS[args.provider]
    if not os.environ.get(key_var):
        sys.exit(f"{key_var} is not set")
    client, api_error = make_client(args.provider)

    preds_a, preds_b = read_preds(args.a), read_preds(args.b)
    ids = sorted(set(preds_a) & set(preds_b), key=str)
    if not ids:
        sys.exit("no overlapping ids between --a and --b")
    if args.limit:
        ids = ids[: args.limit]
    print(f"judging {len(ids)} pairs with {args.model}", file=sys.stderr)

    system = (
        "You are an impartial judge comparing two candidate responses to the same input. "
        f"Judging criteria: {args.criteria}. "
        "A reference output is provided as a guide to what a good answer looks like — "
        "candidates need not match it verbatim. Pick the better candidate, or tie if "
        "they are genuinely comparable. Judge only on the criteria, not on length or polish."
    )

    rnd = random.Random(args.seed)
    tally = {"a": 0, "b": 0, "tie": 0}
    skipped = in_tok = out_tok = api_err_streak = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for n, _id in enumerate(ids, 1):
            ra, rb = preds_a[_id], preds_b[_id]
            b_first = rnd.random() < 0.5
            first, second = (rb, ra) if b_first else (ra, rb)
            user = (
                f"<input>\n{ra['input']}\n</input>\n\n"
                f"<reference>\n{ra['expected']}\n</reference>\n\n"
                f"<candidate_first>\n{first['predicted']}\n</candidate_first>\n\n"
                f"<candidate_second>\n{second['predicted']}\n</candidate_second>"
            )
            try:
                text, it, ot = call_judge(
                    args.provider, client, args.model, system, user, VERDICT_SCHEMA)
            except api_error as e:
                print(f"  id={_id} skipped (api error: {e})", file=sys.stderr)
                skipped += 1
                api_err_streak += 1
                if api_err_streak == n == ABORT_AFTER_FAILS:
                    sys.exit(
                        f"aborting: first {ABORT_AFTER_FAILS} calls all failed with api errors "
                        f"— check --model/--provider ({args.model} on {args.provider}); "
                        f"nothing was judged, fix and re-run")
                continue
            except ProviderRefusal as e:
                # Tokens are spent even on refusals — account before skipping.
                in_tok += e.in_tok
                out_tok += e.out_tok
                print(f"  id={_id} skipped ({e})", file=sys.stderr)
                skipped += 1
                continue
            in_tok += it
            out_tok += ot
            try:
                verdict = json.loads(text)
                winner, reason = verdict["winner"], verdict["reason"]
            except (json.JSONDecodeError, KeyError):
                print(f"  id={_id} skipped (unparseable verdict)", file=sys.stderr)
                skipped += 1
                continue
            # Defense in depth: both providers' schema enums should make this
            # unreachable, but a provider quirk must not corrupt the tally.
            if winner not in ("first", "second", "tie"):
                print(f"  id={_id} skipped (unexpected winner value: {winner!r})", file=sys.stderr)
                skipped += 1
                continue
            if winner == "tie":
                result = "tie"
            elif (winner == "first") == b_first:
                result = "b"
            else:
                result = "a"
            tally[result] += 1
            f.write(json.dumps({"id": _id, "result": result, "reason": reason}, ensure_ascii=False) + "\n")
            if n % 10 == 0:
                print(f"  {n}/{len(ids)}  B wins {tally['b']} / ties {tally['tie']} / A wins {tally['a']}", file=sys.stderr)

    judged = sum(tally.values())
    total = judged or 1
    print(f"\nB (tuned) wins: {tally['b']} ({tally['b']/total:.0%})")
    print(f"ties:           {tally['tie']} ({tally['tie']/total:.0%})")
    print(f"A (base) wins:  {tally['a']} ({tally['a']/total:.0%})")
    print(f"skipped:        {skipped}"
          + (" (refusals/parse errors — details on stderr)" if skipped else ""))
    print(f"tokens:         {in_tok} in / {out_tok} out")
    print(f"verdicts -> {args.output}")
    if judged == 0:
        print("note: no pairs judged — all pairs were skipped")
    elif judged < 150:
        print(f"note: only {judged} pairs judged — differences under ~10 points are noise at this sample size")


if __name__ == "__main__":
    main()
