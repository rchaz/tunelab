#!/usr/bin/env bash
# Run every tunelab script test. Each test_*.py is a standalone stdlib runner
# that invokes the real bundled scripts via subprocess and prints PASS lines
# (the evidence). Non-zero exit here means a script regression — do not ship.
#
# Notes: first run downloads the local embedding model (~125MB) and
# mlx-community/Qwen3-0.6B-4bit (~340MB) into ~/.cache/huggingface, and
# xgboost into uv's cache; later runs are cache-fast.
set -u
cd "$(dirname "$0")/.."

failed=()
for t in tests/test_*.py; do
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
echo "ALL TEST FILES PASSED"
