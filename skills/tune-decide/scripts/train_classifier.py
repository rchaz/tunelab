#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["model2vec>=0.4", "numpy", "scikit-learn>=1.4", "joblib", "openai>=1.40"]
# ///
"""Level 1 classifier: logistic regression (or XGBoost) on text embeddings.

The embedding model does the language comprehension; the classifier just
learns the decision boundary. Trains in seconds, runs for free, and with the
default local backend needs no API key — Levels 0-1 need no API key.

Backends (--backend, default local):
  local   model2vec static embeddings, sentence-transformers/static-retrieval-mrl-en-v1
          (downloads from Hugging Face on first run; CPU-only, no torch).
          Fallback if that id ever disappears: minishlab/potion-base-32M (same package).
  openai  text-embedding-3-small — the quality upgrade; needs OPENAI_API_KEY.

Classifier choice (--classifier, default auto):
  lr       LogisticRegression — fast, interpretable, calibrated probabilities
           (confidence routing needs them). The default.
  xgboost  for non-text/tabular features joining the embeddings (--extra-keys),
           or when LR demonstrably underperforms. Not in the inline deps —
           availability is checked BEFORE any embedding happens; on a miss the
           script exits with the exact `uv run --with xgboost` rerun command.
  auto     lr, unless --extra-keys is given (tabular features -> xgboost).

Train (holds out 20% and prints honest metrics, incl. macro-F1):
  uv run train_classifier.py --data labeled.jsonl --model-out classifier.joblib
  uv run --with xgboost train_classifier.py --data labeled.jsonl --extra-keys amount,age
  # auto + --extra-keys selects xgboost, which is not in the inline deps — hence
  # the --with; append `--classifier lr` instead to stay dependency-free.

Predict (backend, embed model, and text key come from the saved bundle, so
prediction-time embeddings always match training — conflicting flags are an
error, never silent):
  uv run train_classifier.py --predict inputs.jsonl --model-in classifier.joblib --output preds.jsonl

  labeled.jsonl lines: {"text": "...", "label": "..."} (+ any --extra-keys fields, numeric)
  output lines:        input record + {"predicted": "...", "confidence": 0.93}

confidence is the model's probability for the predicted class — use it to
route low-confidence inputs back to a frontier model.
"""

import argparse
import importlib.util
import json
import os
import shlex
import sys
from collections import Counter

import joblib
import numpy as np
from model2vec import StaticModel
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

LOCAL_DEFAULT = "sentence-transformers/static-retrieval-mrl-en-v1"
OPENAI_DEFAULT = "text-embedding-3-small"
BATCH = 256  # OpenAI API batch size

WHY = {
    "lr": "LogisticRegression — fast, interpretable, calibrated probabilities for confidence routing",
    "xgboost": "XGBoost — non-text/tabular features join the embeddings (or LR demonstrably underperformed)",
}


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


def extract_extras(rows, keys, where):
    feats = np.empty((len(rows), len(keys)), dtype=np.float32)
    for i, rec in enumerate(rows):
        for j, key in enumerate(keys):
            if key not in rec:
                sys.exit(f"{where} record {i} is missing extra-keys field {key!r}")
            try:
                feats[i, j] = float(rec[key])
            except (TypeError, ValueError):
                sys.exit(f"{where} record {i}: extra-keys field {key!r}={rec[key]!r} is not numeric")
    return feats


def ensure_xgboost():
    # Called before any embedding pass: an --backend openai run must never pay
    # for embeddings, exit on the missing dep, and pay again on the rerun.
    if importlib.util.find_spec("xgboost") is not None:
        return
    rerun = "uv run --with xgboost " + shlex.join(sys.argv)
    sys.exit(
        "xgboost is not installed (kept out of the inline deps — LR is the default for a reason).\n"
        f"Rerun with:\n  {rerun}"
    )


def build_classifier(kind, seed):
    if kind == "lr":
        return LogisticRegression(max_iter=2000, C=1.0)
    ensure_xgboost()
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--data", help="labeled JSONL for training: {text, label}")
    mode.add_argument("--predict", help="JSONL of texts to classify")
    ap.add_argument("--model-out", default="classifier.joblib")
    ap.add_argument("--model-in", help="trained .joblib from a previous run")
    ap.add_argument("--output", help="predictions output path")
    ap.add_argument(
        "--text-key",
        default=None,
        help="record field holding the text (default: text; in predict mode it always comes from the bundle)",
    )
    ap.add_argument(
        "--backend",
        choices=["local", "openai"],
        default=None,
        help="embedding backend (default: local; in predict mode it always comes from the bundle)",
    )
    ap.add_argument(
        "--embed-model",
        default=None,
        help=f"model id within the backend (local default: {LOCAL_DEFAULT}, openai default: {OPENAI_DEFAULT})",
    )
    ap.add_argument(
        "--classifier",
        choices=["auto", "lr", "xgboost"],
        default="auto",
        help="auto = lr, unless --extra-keys is given (see docstring for the heuristic)",
    )
    ap.add_argument(
        "--extra-keys",
        default=None,
        help="comma-separated numeric fields from the input records, appended to the embedding features",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    extra_keys = [k.strip() for k in args.extra_keys.split(",") if k.strip()] if args.extra_keys else []

    if args.data:
        backend = args.backend or "local"
        embed_model = args.embed_model or (LOCAL_DEFAULT if backend == "local" else OPENAI_DEFAULT)
        text_key = args.text_key or "text"
        kind = args.classifier
        if kind == "auto":
            kind = "xgboost" if extra_keys else "lr"
        if kind == "xgboost":
            ensure_xgboost()  # before embedding — the rerun must not re-pay for embeddings
        print(f"classifier: {WHY[kind]}", file=sys.stderr)

        rows = read_jsonl(args.data)
        texts = [get_field(r, i, text_key, "--data") for i, r in enumerate(rows)]
        y_raw = [get_field(r, i, "label", "--data") for i, r in enumerate(rows)]
        counts = Counter(y_raw)
        if len(counts) < 2:
            sys.exit("need at least 2 distinct labels")
        floor = 5 * len(counts)
        if len(rows) < floor:
            sys.exit(
                f"only {len(rows)} labeled records across {len(counts)} classes — Level 1 needs at "
                f"least {floor} (~5 per class; 20+ per class recommended) for a meaningful 20% "
                "holdout. Label more data, or try centroid_classify.py (Level 0) meanwhile."
            )
        singles = sorted(lb for lb, c in counts.items() if c < 2)
        if singles:
            sys.exit(
                f"class(es) {singles} have a single example — the stratified 20% holdout "
                "needs at least 2 per class (5+ recommended)"
            )
        # encode labels to ints once so LR and XGBoost behave identically
        # (XGBClassifier rejects string labels); bundle["classes"] maps back
        le = LabelEncoder().fit(y_raw)
        y = le.transform(y_raw)

        X = make_embedder(backend, embed_model)(texts)
        extra_mean = extra_std = None
        if extra_keys:
            extras = extract_extras(rows, extra_keys, "--data")
            # standardize so raw magnitudes don't drown the embedding features
            extra_mean = extras.mean(axis=0)
            extra_std = extras.std(axis=0)
            constant = extra_std == 0
            if constant.any():
                flat = [k for k, c in zip(extra_keys, constant) if c]
                print(
                    f"warning: extra-keys column(s) {flat} are constant in the training data — "
                    "they carry no signal; scaled to zero so a different value at predict "
                    "time cannot blow up the features",
                    file=sys.stderr,
                )
            extra_std = np.where(constant, 1.0, extra_std).astype(np.float32)
            X = np.hstack([X, (extras - extra_mean) / extra_std])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )
        clf = build_classifier(kind, args.seed)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        print(f"\nheld-out accuracy: {accuracy_score(y_te, preds):.3f} (n={len(y_te)})")
        print(f"held-out macro-F1: {f1_score(y_te, preds, average='macro'):.3f}")
        print(classification_report(le.inverse_transform(y_te), le.inverse_transform(preds), zero_division=0))
        # Refit on everything before saving — the holdout was only for the honest number
        clf_full = build_classifier(kind, args.seed).fit(X, y)
        joblib.dump(
            {
                "model": clf_full,
                "classes": le.classes_.tolist(),
                "backend": backend,
                "embed_model": embed_model,
                "text_key": text_key,
                "classifier": kind,
                "extra_keys": extra_keys,
                "extra_mean": None if extra_mean is None else extra_mean.tolist(),
                "extra_std": None if extra_std is None else extra_std.tolist(),
            },
            args.model_out,
        )
        print(
            f"saved -> {args.model_out} (bundle records backend + embed model + text key: "
            "predict always matches)",
            file=sys.stderr,
        )
    elif args.predict:
        if not (args.model_in and args.output):
            sys.exit("--predict requires --model-in and --output")
        try:
            bundle = joblib.load(args.model_in)
        except ModuleNotFoundError as e:
            if "xgboost" in str(e):
                sys.exit(
                    "this bundle holds an XGBoost model but xgboost is not installed.\n"
                    "Rerun with:\n  uv run --with xgboost " + shlex.join(sys.argv)
                )
            raise
        # the bundle is the source of truth — embedding with a different
        # backend/model/text field than training silently produces garbage predictions
        if args.backend and args.backend != bundle["backend"]:
            sys.exit(
                f"bundle was trained with backend={bundle['backend']!r}; refusing --backend "
                f"{args.backend} (predict must embed exactly like training — drop the flag or retrain)"
            )
        if args.embed_model and args.embed_model != bundle["embed_model"]:
            sys.exit(
                f"bundle was trained with embed model {bundle['embed_model']!r}; refusing "
                f"--embed-model {args.embed_model} (predict must embed exactly like training)"
            )
        bundle_text_key = bundle.get("text_key", "text")
        if args.text_key and args.text_key != bundle_text_key:
            sys.exit(
                f"bundle was trained on field {bundle_text_key!r}; refusing --text-key "
                f"{args.text_key} (predict must embed the same field as training)"
            )
        if args.classifier != "auto" and args.classifier != bundle["classifier"]:
            sys.exit(
                f"bundle holds a {bundle['classifier']} model; --classifier {args.classifier} "
                "cannot change that at predict time — drop the flag or retrain"
            )
        if args.extra_keys is not None and extra_keys != bundle["extra_keys"]:
            sys.exit(
                f"bundle was trained with extra-keys {bundle['extra_keys']!r}; refusing "
                f"--extra-keys {args.extra_keys!r} (the bundle's stored list is what predict uses)"
            )
        backend, embed_model, text_key = bundle["backend"], bundle["embed_model"], bundle_text_key
        print(f"embedding with {backend}:{embed_model} ({bundle['classifier']} bundle)", file=sys.stderr)

        rows = read_jsonl(args.predict)
        if not rows:
            sys.exit("--predict file has no records — nothing to classify")
        texts = [get_field(r, i, text_key, "--predict") for i, r in enumerate(rows)]
        X = make_embedder(backend, embed_model)(texts)
        if bundle["extra_keys"]:
            extras = extract_extras(rows, bundle["extra_keys"], "--predict")
            mean = np.asarray(bundle["extra_mean"], dtype=np.float32)
            std = np.asarray(bundle["extra_std"], dtype=np.float32)
            X = np.hstack([X, (extras - mean) / std])
        probs = bundle["model"].predict_proba(X)
        classes = bundle["classes"]
        with open(args.output, "w") as f:
            for rec, p in zip(rows, probs):
                best = int(np.argmax(p))
                rec["predicted"] = classes[best]
                rec["confidence"] = round(float(p[best]), 4)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} predictions -> {args.output}", file=sys.stderr)
    else:
        sys.exit("pass --data (train) or --predict (classify)")


if __name__ == "__main__":
    main()
