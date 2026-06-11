# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "model2vec>=0.4",
#   "scikit-learn>=1.4",
#   "numpy",
# ]
# ///
"""tunelab plan 14.2 — local embedding bench, model2vec family (static embeddings, no torch).

Models: minishlab/potion-base-8M, minishlab/potion-base-32M
Dataset: /tmp/tunelab-verify/dataset/raw.jsonl (3000 rows, 10 classes)
Protocol:
  - 80/20 stratified split, seed 42
  - timed CPU encode of all 3000 texts (after warmup) -> texts/sec
  - LogisticRegression(max_iter=2000) on train embeddings -> test accuracy + macro-F1
  - Level 0 probe: 20 examples/class from train -> mean centroids -> cosine classify test
Run: uv run bench_model2vec.py  (or: <venv_python> bench_model2vec.py)
"""
import json, sys, time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATA = "/tmp/tunelab-verify/dataset/raw.jsonl"
SEED = 42

def load_dataset():
    texts, labels = [], []
    with open(DATA) as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels

def evaluate(name, emb, labels, encode_secs):
    n = len(labels)
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=labels)
    y = np.array(labels)
    Xtr, Xte, ytr, yte = emb[tr], emb[te], y[tr], y[te]

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")

    # Level 0 probe: 20/class centroids, cosine
    rng = np.random.default_rng(SEED)
    Xn = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
    classes = sorted(set(labels))
    cents = []
    for c in classes:
        ci = np.where(ytr == c)[0]
        pick = rng.choice(ci, size=min(20, len(ci)), replace=False)
        m = Xn[pick].mean(axis=0)
        cents.append(m / (np.linalg.norm(m) + 1e-12))
    C = np.stack(cents)
    Xten = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-12)
    cpred = np.array(classes)[(Xten @ C.T).argmax(axis=1)]
    cacc = accuracy_score(yte, cpred)

    return {
        "model_id": name,
        "n_texts": n,
        "dim": int(emb.shape[1]),
        "encode_wall_secs": round(encode_secs, 2),
        "encode_texts_per_sec": round(n / encode_secs, 1),
        "lr_accuracy": round(float(acc), 4),
        "lr_macro_f1": round(float(f1), 4),
        "centroid_accuracy_20_per_class": round(float(cacc), 4),
    }

def main():
    from model2vec import StaticModel
    texts, labels = load_dataset()
    out = []
    for mid in ["minishlab/potion-base-8M", "minishlab/potion-base-32M"]:
        t0 = time.perf_counter()
        model = StaticModel.from_pretrained(mid)
        load_s = time.perf_counter() - t0
        model.encode(texts[:8])  # warmup
        t0 = time.perf_counter()
        emb = np.asarray(model.encode(texts))
        enc_s = time.perf_counter() - t0
        r = evaluate(mid, emb, labels, enc_s)
        r["model_load_secs"] = round(load_s, 2)
        out.append(r)
        print(json.dumps(r), flush=True)
    json.dump(out, open("/tmp/tunelab-verify/embed_bench/results_model2vec.json", "w"), indent=2)

if __name__ == "__main__":
    sys.exit(main())
