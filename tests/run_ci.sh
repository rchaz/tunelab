#!/usr/bin/env bash
# CI subset of the test suite — every test_*.py EXCEPT the ones that need a real
# dataset or a model download. Those run locally via tests/run_all.sh on Apple
# Silicon, where the data and MLX models live; forcing them onto hosted runners
# would be slow, costly, and flaky. Keeping CI to fixture-only, deterministic
# tests means a green check that honestly says "the scripts' logic is intact".
#
# LOCAL_ONLY (run via tests/run_all.sh, not here) — add heavy/data/model tests:
#   test_centroid_classify  needs dogfood/level1/data/raw.jsonl (gitignored) + embedding model
#   test_train_classifier   needs that dataset + xgboost + embedding model
#   test_run_test_set       downloads mlx-community/Qwen3-0.6B-4bit (~0.34GB) and runs MLX
set -u
cd "$(dirname "$0")/.."

LOCAL_ONLY="test_centroid_classify test_train_classifier test_run_test_set"

failed=()
for t in tests/test_*.py; do
  base="$(basename "$t" .py)"
  case " $LOCAL_ONLY " in
    *" $base "*) echo "=== $t (skipped: local-only — run via tests/run_all.sh)"; continue ;;
  esac
  echo "=== $t"
  if ! python3 "$t"; then
    failed+=("$t")
  fi
done

echo
if [ ${#failed[@]} -gt 0 ]; then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "ALL CI TEST FILES PASSED"
