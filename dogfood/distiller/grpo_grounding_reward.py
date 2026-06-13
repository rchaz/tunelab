"""GRPO reward = the grounding gate, for the distiller's RLVR round-2.

SFT taught the student to compress but not to stay grounded (eval: 0.82 vs
teacher 0.93 zero-hallucination). RLVR fixes exactly that: the mechanical gate
is already a verifiable reward, so we reward groundedness + compression directly.

mlx-lm-lora calls a reward func as (prompts, completions, answer, types) -> list[float].
`answer` carries the source blob per record (GRPO dataset built with the blob as
the answer column). Reward per completion:
    +1.0   if zero ungrounded hard tokens (the non-negotiable property)
    +bonus  max(0, 0.40 - ratio)  -- compress more, down to the 0.40 budget
    -0.0   otherwise (ungrounded output earns no grounding credit)
so a perfectly-grounded 25%-ratio compression scores ~1.15; a fluent but
hallucinating one scores at most the ratio bonus — the gradient points at
grounding first, compression second.
"""
import re

# --- grounding gate (kept in sync with skills/tune-eval/scripts/grounding_gate.py) ---
NUMERIC_RE = re.compile(r"\d\d|\d[.,:\-]|[.,:\-]\d|^\d+$")
PATH_RE = re.compile(r"^[~/]|/.*\.[A-Za-z0-9]+|/.*/")
DOTTED_RE = re.compile(r"[A-Za-z0-9]+[.:][A-Za-z0-9][A-Za-z0-9.:]*")
HEXCODE_RE = re.compile(r"(?:0x[0-9a-fA-F]+|[0-9a-fA-F]{8,})")
SUBSPLIT_RE = re.compile(r"[./:]+")
SPLIT_RE = re.compile(r"[\s,;()\[\]{}<>\"'`|=]+")


def _is_hard(tok):
    return bool(NUMERIC_RE.search(tok) or PATH_RE.search(tok)
                or DOTTED_RE.fullmatch(tok) or HEXCODE_RE.fullmatch(tok))


def _ungrounded(blob, out):
    src = blob.lower()
    bad = []
    for raw in SPLIT_RE.split(out):
        raw = raw.strip(".,:;)(")
        if not raw or not _is_hard(raw):
            continue
        toks = [raw] + [a for a in SUBSPLIT_RE.split(raw) if a and _is_hard(a)]
        for t in toks:
            if t.lower() not in src:
                bad.append(t)
    return bad


def grounding_reward(prompts, completions, answer, types=None):
    rewards = []
    for comp, blob in zip(completions, answer):
        comp = comp or ""
        blob = blob or ""
        bad = _ungrounded(blob, comp)
        ratio = len(comp) / max(len(blob), 1)
        r = 1.0 if not bad else 0.0
        r += max(0.0, 0.40 - ratio)
        # hard floor: an empty or non-compressing output is worthless
        if not comp.strip() or ratio > 1.0:
            r = 0.0
        rewards.append(r)
    return rewards


# mlx-lm-lora discovers module-level reward callables; expose under a clear name.
reward_functions = [grounding_reward]


if __name__ == "__main__":
    # self-test: grounded+compressed beats fluent+hallucinated
    blob = "Written /Users/rc/run/step5.txt status OK exit 0 size 5.54 kB took 12.3ms"
    good = "step5.txt OK exit 0, 5.54 kB, 12.3ms"
    bad = "the run wrote a file, took 99.9ms, exit 0"   # 99.9ms invented
    rs = grounding_reward([""] * 2, [good, bad], [blob, blob])
    print(f"grounded+compressed: {rs[0]:.3f}   fluent+hallucinated: {rs[1]:.3f}")
    assert rs[0] > rs[1], "grounding reward must prefer the grounded output"
    print("SELF-TEST PASS")
