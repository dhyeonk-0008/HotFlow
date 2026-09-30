#!/usr/bin/env bash
# Download the case-study target structures from RCSB into test_cases/.
# These are the inputs referenced by hotflow/benchmark_targets.py:TARGET_CONFIGS.
#
#   bash scripts/fetch_targets.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO_ROOT/test_cases"
mkdir -p "$OUT"

# 6YVR is only fetched as mmCIF; the rest as PDB, matching TARGET_CONFIGS.
TARGETS=(9CDZ.pdb 7UXO.pdb 6YVR.cif 4Y5U.pdb 8TF5.pdb 6LUQ.pdb)

for entry in "${TARGETS[@]}"; do
    if [ -s "$OUT/$entry" ]; then
        echo "[skip] $entry already present"
        continue
    fi
    echo "[get ] $entry"
    curl -fsSL "https://files.rcsb.org/download/$entry" -o "$OUT/$entry"
done

echo
echo "Targets in $OUT:"
ls -1 "$OUT" | grep -E '\.(pdb|cif)$'
