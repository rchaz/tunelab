#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic>=0.92"]
# ///
"""Pairwise LLM-as-judge for generative outputs. Needs ANTHROPIC_API_KEY.

  uv run judge_eval.py --a preds_base.jsonl --b preds_tuned.jsonl \
      --criteria "Faithful to the task; matches the reference voice; no fabricated facts" \
      --output verdicts.jsonl

Inputs are run_test_set.py outputs ({"id", "input", "expected", "predicted"})
joined on id. Each pair is judged blind with A/B order randomized per item
(judges have position bias) and a structured-output verdict. Reports win/tie/
loss for B (your tuned model) vs A.

Refused or unparseable verdicts are skipped (logged to stderr, counted in the
summary), not tallied. With ~100 items, differences under ~10 points are noise
— the summary says so whenever fewer than 150 pairs were judged.
"""

import argparse
import json
import os
import random
import sys

import anthropic

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["first", "second", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["winner", "reason"],
    "additionalProperties": False,
}


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
    ap.add_argument("--provider", choices=["anthropic"], default="anthropic",
                    help="judge API provider (OpenAI lands in Phase 2)")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(max_retries=5)

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
    skipped = 0
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
                resp = client.messages.create(
                    model=args.model,
                    # Adaptive-thinking tokens count against max_tokens; the
                    # verdict itself is tiny, so the headroom is for thinking.
                    max_tokens=8192,
                    thinking={"type": "adaptive"},
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
                )
            except anthropic.APIError as e:
                print(f"  id={_id} skipped (api error: {e})", file=sys.stderr)
                skipped += 1
                continue
            text = next((blk.text for blk in resp.content if blk.type == "text"), None)
            if text is None:
                why = f"stop_reason={resp.stop_reason}"
                if resp.stop_reason == "max_tokens":
                    why += " — thinking hit the token cap"
                print(f"  id={_id} skipped ({why})", file=sys.stderr)
                skipped += 1
                continue
            try:
                verdict = json.loads(text)
                winner, reason = verdict["winner"], verdict["reason"]
            except (json.JSONDecodeError, KeyError):
                print(f"  id={_id} skipped (unparseable verdict)", file=sys.stderr)
                skipped += 1
                continue
            # Defense in depth: the json_schema enum should make this
            # unreachable, but a Phase-2 provider may not guarantee it.
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
    print(f"verdicts -> {args.output}")
    if judged == 0:
        print("note: no pairs judged — all pairs were skipped")
    elif judged < 150:
        print(f"note: only {judged} pairs judged — differences under ~10 points are noise at this sample size")


if __name__ == "__main__":
    main()
