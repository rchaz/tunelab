#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["model2vec>=0.4", "numpy", "openai>=1.40"]
# ///
"""Level 0 classifier: cosine similarity to per-class embedding centroids.

No training, and with the default local backend no API key either — Levels 0-1
need no API key. Embeds a handful of labeled examples, averages each class into
a centroid, and classifies new texts by nearest centroid.

Backends (--backend, default local):
  local   model2vec static embeddings, sentence-transformers/static-retrieval-mrl-en-v1
          (downloads from Hugging Face on first run; CPU-only, no torch).
          Fallback if that id ever disappears: minishlab/potion-base-32M (same package).
  openai  text-embedding-3-small — the quality upgrade; needs OPENAI_API_KEY.

Usage:
  uv run centroid_classify.py --examples labeled.jsonl --classify inputs.jsonl --output preds.jsonl
  uv run centroid_classify.py ... --backend openai                          # quality upgrade
  uv run centroid_classify.py ... --embed-model minishlab/potion-base-32M   # local fallback

  labeled.jsonl lines: {"text": "...", "label": "..."}
  inputs.jsonl lines:  {"text": "..."}  (other keys are preserved in output)
  output lines:        input record + {"predicted": "...", "confidence": 0.07}

confidence is the cosine margin between the best and second-best centroid —
near-zero margins are good candidates for routing to a frontier model.

Caveat (benchmarked live 2026-06-10, CFPB 10-class, dogfood/embedding-bench/):
static embeddings are weak at few-shot centroids — 0.44 accuracy at 20
examples/class vs 0.62 for MiniLM (a trained classifier on the same vectors
reaches 0.73, so this is a centroid limitation, not an embedding-quality one).
With ~20 examples/class and a tight margin distribution, either add more
examples per class or rerun with --backend openai.
"""

import argparse
import json
import os
import sys

import numpy as np
from model2vec import StaticModel
from openai import OpenAI

LOCAL_DEFAULT = "sentence-transformers/static-retrieval-mrl-en-v1"
OPENAI_DEFAULT = "text-embedding-3-small"
BATCH = 256  # OpenAI API batch size


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_field(rec, i, key, where):
    if key not in rec:
        sys.exit(f"{where} record {i} is missing field {key!r}")
    return rec[key]


def make_embedder(backend, model_id):
    if backend == "local":
        print(f"loading {model_id} (downloads from Hugging Face on first run)", file=sys.stderr)
        model = StaticModel.from_pretrained(model_id)

        def encode(texts):
            vecs = np.asarray(model.encode(texts), dtype=np.float32)
            print(f"  embedded {len(texts)} texts locally", file=sys.stderr)
            return vecs

        return encode

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("--backend openai needs OPENAI_API_KEY (the default --backend local needs no key)")
    client = OpenAI()

    def encode(texts):
        vecs = []
        for i in range(0, len(texts), BATCH):
            chunk = [t.replace("\n", " ")[:8000] for t in texts[i : i + BATCH]]
            resp = client.embeddings.create(model=model_id, input=chunk)
            vecs.extend(d.embedding for d in resp.data)
            print(f"  embedded {min(i + BATCH, len(texts))}/{len(texts)}", file=sys.stderr)
        return np.array(vecs, dtype=np.float32)

    return encode


def normalize(arr):
    # local static-model vectors are NOT unit-norm — cosine math requires this
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", required=True, help="labeled JSONL: {text, label}")
    ap.add_argument("--classify", required=True, help="JSONL of texts to classify")
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-key", default="text")
    ap.add_argument(
        "--backend",
        choices=["local", "openai"],
        default="local",
        help="embedding backend (default: local — no API key needed)",
    )
    ap.add_argument(
        "--embed-model",
        default=None,
        help=f"model id within the backend (local default: {LOCAL_DEFAULT}, openai default: {OPENAI_DEFAULT})",
    )
    args = ap.parse_args()

    model_id = args.embed_model or (LOCAL_DEFAULT if args.backend == "local" else OPENAI_DEFAULT)

    # validate both files fully before any embedding happens (which may be paid)
    examples = read_jsonl(args.examples)
    ex_labels = [get_field(ex, i, "label", "--examples") for i, ex in enumerate(examples)]
    ex_texts = [get_field(ex, i, args.text_key, "--examples") for i, ex in enumerate(examples)]
    labels = sorted(set(ex_labels))
    if len(labels) < 2:
        sys.exit("need at least 2 distinct labels in --examples")
    counts = {lb: ex_labels.count(lb) for lb in labels}
    print(f"classes: {counts}", file=sys.stderr)

    inputs = read_jsonl(args.classify)
    if not inputs:
        sys.exit("--classify file has no records — nothing to classify")
    in_texts = [get_field(r, i, args.text_key, "--classify") for i, r in enumerate(inputs)]

    embed = make_embedder(args.backend, model_id)
    ex_vecs = normalize(embed(ex_texts))
    centroids = np.stack(
        [ex_vecs[[i for i, ex_lb in enumerate(ex_labels) if ex_lb == lb]].mean(axis=0) for lb in labels]
    )
    centroids = normalize(centroids)

    in_vecs = normalize(embed(in_texts))
    sims = in_vecs @ centroids.T  # cosine similarity, all normalized

    margins = []
    with open(args.output, "w") as f:
        for rec, row in zip(inputs, sims):
            order = np.argsort(row)[::-1]
            margin = float(row[order[0]] - row[order[1]])
            rec["predicted"] = labels[order[0]]
            rec["confidence"] = round(margin, 4)
            margins.append(margin)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dist = {}
    for rec in inputs:
        dist[rec["predicted"]] = dist.get(rec["predicted"], 0) + 1
    print(f"wrote {len(inputs)} predictions -> {args.output}", file=sys.stderr)
    print(f"predicted distribution: {dist}", file=sys.stderr)
    if margins:
        p25, p50, p75 = np.percentile(margins, [25, 50, 75])
        print(f"confidence margins: p25={p25:.3f} median={p50:.3f} p75={p75:.3f}", file=sys.stderr)
    if args.backend == "local":
        print(
            "note: static embeddings are weak at few-shot centroids (see docstring) — "
            "if margins look tight, add examples per class or retry with --backend openai",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
