# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "sentence-transformers>=3.0",
#   "scikit-learn>=1.4",
#   "numpy",
# ]
# ///
"""tunelab plan 14.2 — local embedding bench, sentence-transformers family (torch stack).

Models: sentence-transformers/all-MiniLM-L6-v2, sentence-transformers/static-retrieval-mrl-en-v1
Dataset: /tmp/tunelab-verify/dataset/raw.jsonl (3000 rows, 10 classes)
Protocol: identical to bench_model2vec.py. device="cpu" forced (criterion is CPU encode
speed across user machines; no silent MPS pickup on Apple Silicon).
Run: uv run bench_sentence_transformers.py  (or: <venv_python> bench_sentence_transformers.py)
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
    from sentence_transformers import SentenceTransformer
    texts, labels = load_dataset()
    out = []
    for mid in [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/static-retrieval-mrl-en-v1",
    ]:
        t0 = time.perf_counter()
        model = SentenceTransformer(mid, device="cpu")
        load_s = time.perf_counter() - t0
        model.encode(texts[:8], show_progress_bar=False)  # warmup
        t0 = time.perf_counter()
        emb = np.asarray(model.encode(texts, show_progress_bar=False, batch_size=32))
        enc_s = time.perf_counter() - t0
        r = evaluate(mid, emb, labels, enc_s)
        r["model_load_secs"] = round(load_s, 2)
        out.append(r)
        print(json.dumps(r), flush=True)
    json.dump(out, open("/tmp/tunelab-verify/embed_bench/results_sentence_transformers.json", "w"), indent=2)

if __name__ == "__main__":
    sys.exit(main())
