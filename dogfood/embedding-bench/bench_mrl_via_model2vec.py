# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "model2vec>=0.4",
#   "scikit-learn>=1.4",
#   "numpy",
# ]
# ///
"""tunelab plan 14.2 follow-up — load sentence-transformers/static-retrieval-mrl-en-v1
through model2vec.StaticModel (it is a static embedding model, so model2vec can read it),
checking whether its quality is reachable WITHOUT the torch stack.
Protocol identical to bench_model2vec.py.
"""
import json, sys, time

import numpy as np

sys.path.insert(0, "/tmp/tunelab-verify/embed_bench")
from bench_model2vec import evaluate, load_dataset  # same split/probe code


def main():
    from model2vec import StaticModel
    texts, labels = load_dataset()
    mid = "sentence-transformers/static-retrieval-mrl-en-v1"
    t0 = time.perf_counter()
    model = StaticModel.from_pretrained(mid)
    load_s = time.perf_counter() - t0
    model.encode(texts[:8])  # warmup
    t0 = time.perf_counter()
    emb = np.asarray(model.encode(texts))
    enc_s = time.perf_counter() - t0
    r = evaluate(mid + " (via model2vec)", emb, labels, enc_s)
    r["model_load_secs"] = round(load_s, 2)
    print(json.dumps(r), flush=True)
    json.dump([r], open("/tmp/tunelab-verify/embed_bench/results_mrl_via_model2vec.json", "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
