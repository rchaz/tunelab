#!/usr/bin/env python3
"""Remove exact and near-duplicate records from a training JSONL file. Stdlib only.

  python3 dedupe.py --input train_chat.jsonl --output deduped.jsonl [--threshold 0.80]

Repetition is the dominant cause of memorization: a record appearing 50 times
trains ~50 epochs on that one example. Exact dups are caught by a normalized
hash; near-dups by MinHash-banded Jaccard similarity over character 4-gram
shingles (robust on short records, where word n-grams overreact to one-word
edits). Chat records are compared on user/assistant content only — a shared
system prompt is intentional boilerplate, not duplication. First occurrence
wins. Deterministic: the same input produces byte-identical output across
runs and processes (no PYTHONHASHSEED dependence). Works on chat,
completions, and text formats.
"""

import argparse
import functools
import hashlib
import json
import re
import sys
from collections import defaultdict

NUM_HASHES = 64
BANDS = 16  # 16 bands x 4 rows: surfaces candidate pairs down to Jaccard ~0.5
MASK64 = (1 << 64) - 1


def _splitmix64(x):
    x = (x + 0x9E3779B97F4B7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


# Fixed (a, b) per simulated hash function; a forced odd so h -> (a*h + b) mod 2^64
# is a bijection. Pure arithmetic — no dependence on interpreter hash salting.
SEED_PARAMS = [(_splitmix64(2 * i) | 1, _splitmix64(2 * i + 1)) for i in range(NUM_HASHES)]


def record_text(rec):
    if "messages" in rec:
        return " ".join(
            str(m.get("content", "")) for m in rec["messages"] if m.get("role") != "system"
        )
    if "prompt" in rec:
        return f"{rec['prompt']} {rec.get('completion', '')}"
    return str(rec.get("text", json.dumps(rec, sort_keys=True)))


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text, n=4):
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


@functools.lru_cache(maxsize=None)
def base_hash(s):
    # One stable 64-bit hash per shingle; the 64 "hash functions" are cheap
    # affine mixes of it (calling blake2b 64x per shingle would dominate runtime).
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def minhash(shingle_set):
    bases = [base_hash(s) for s in shingle_set]
    return [min(((a * h + b) & MASK64) for h in bases) for a, b in SEED_PARAMS]


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.80, help="Jaccard near-dup threshold")
    args = ap.parse_args()

    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]
    texts = [normalize(record_text(r)) for r in records]

    # Pass 1: exact duplicates
    seen_hash, keep = set(), []
    exact_dropped = 0
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode()).hexdigest()
        if h in seen_hash:
            exact_dropped += 1
            continue
        seen_hash.add(h)
        keep.append(i)

    # Pass 2: near-duplicates via MinHash banding among survivors
    if len(keep) >= 1000:
        # ~35s per 2k records on M1 Pro — announce so a long silence isn't read as a hang
        print(f"  computing MinHash signatures for {len(keep)} records...", file=sys.stderr)
    rows_per_band = NUM_HASHES // BANDS
    shingle_cache = {i: shingles(texts[i]) for i in keep}
    sigs = {i: minhash(shingle_cache[i]) for i in keep if shingle_cache[i]}

    buckets = defaultdict(list)
    for i in keep:
        if i not in sigs:
            continue
        for b in range(BANDS):
            band = tuple(sigs[i][b * rows_per_band : (b + 1) * rows_per_band])
            buckets[(b, band)].append(i)

    near_dropped, dropped, samples = 0, set(), []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for j, a in enumerate(members):
            if a in dropped:
                continue
            for b_idx in members[j + 1 :]:
                if b_idx in dropped:
                    continue
                if jaccard(shingle_cache[a], shingle_cache[b_idx]) >= args.threshold:
                    dropped.add(b_idx)
                    near_dropped += 1
                    if len(samples) < 3:
                        samples.append((texts[a][:80], texts[b_idx][:80]))

    for kept_t, dropped_t in samples:
        print(f"  near-dup example:\n    kept:    {kept_t}\n    dropped: {dropped_t}", file=sys.stderr)
    if near_dropped > 0.2 * len(keep):
        print(
            f"  NOTE: {near_dropped}/{len(keep)} flagged as near-dups. If your data is "
            "legitimately templated (receipts, form letters), raise --threshold (e.g. 0.95) "
            "and inspect the examples above before trusting this.",
            file=sys.stderr,
        )

    final = [i for i in keep if i not in dropped]
    with open(args.output, "w") as f:
        for i in final:
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")

    print(
        f"{len(records)} in -> {len(final)} out "
        f"({exact_dropped} exact dups, {near_dropped} near-dups removed) -> {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
