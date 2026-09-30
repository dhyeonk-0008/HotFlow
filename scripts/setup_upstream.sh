#!/usr/bin/env bash
# Clone the two upstream repositories at the pinned commits and apply the
# HotFlow patches. Run once from the repository root:
#
#   bash scripts/setup_upstream.sh
#
# Both upstream repositories are MIT licensed. They are not vendored here;
# this script reconstructs the exact trees HotFlow was developed against.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PEPFLOW_URL="https://github.com/Ced3-han/PepFlowww.git"
PEPFLOW_COMMIT="16e0d267c2dbd96cdacbe5ac07c4dada0d61169b"  # 2025-05-25

PEPHAR_URL="https://github.com/Ced3-han/PepHAR.git"
PEPHAR_COMMIT="4a809d5b721ba1dc91b6d6d3dd1f1442eb8c1ee5"    # 2025-05-05

setup() {
    local name="$1" dir="$2" url="$3" commit="$4" patch="$5"

    if [ -e "$dir" ]; then
        echo "[$name] '$dir' already exists — skipping. Remove it to re-run."
        return 0
    fi

    echo "[$name] cloning $url"
    git clone --quiet "$url" "$dir"
    git -C "$dir" checkout --quiet "$commit"

    echo "[$name] applying $patch"
    git -C "$dir" apply --check "$REPO_ROOT/$patch"
    git -C "$dir" apply "$REPO_ROOT/$patch"
    echo "[$name] done."
}

setup PepFlow "PepFlowww" "$PEPFLOW_URL" "$PEPFLOW_COMMIT" "patches/pepflow.patch"
setup PepHAR  "PepHAR"    "$PEPHAR_URL"  "$PEPHAR_COMMIT"  "patches/pephar.patch"

cat <<'EOF'

Upstream code is ready. Two downloads remain, and neither is scriptable here
because both sit behind the upstream authors' Google Drive:

  1. PepFlow pretrained weights -> PepFlowww/model2.pt
     https://github.com/Ced3-han/PepFlowww

  2. PepHAR density + prediction checkpoints -> PepHAR/ckpts/
     https://drive.google.com/drive/folders/1jJFPZbczI7Nxai-9X5UcNsv5U8rcBUEY

     Expected layout:
       PepHAR/ckpts/density_v4_x5o2_2024_09_08__11_25_36/
       PepHAR/ckpts/prediction_d2_x2o1_2024_09_08__11_21_33/

See README.md ("Setup") for the dataset and environment steps.
EOF
