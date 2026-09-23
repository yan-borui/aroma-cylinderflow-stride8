#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
python_bin=${PYTHON:-python}
data="$DATA_DIR/airfoil_stride8_75frames.h5"
manifest="$DATA_DIR/airfoil_stride8_75frames_manifest.json"
if [[ ! -f "$PREPARED_DIR/ready.json" ]]; then
    if [[ -e "$PREPARED_DIR" ]]; then
        printf 'Incomplete method cache exists: %s; choose a new PREPARED_DIR.\n' "$PREPARED_DIR" >&2; exit 2
    fi
    attempt=$(mktemp -d "${PREPARED_DIR}.attempt.XXXXXX")
    "$python_bin" -m cylinderflow prepare --dataset "$data" --manifest "$manifest" \
        --config cylinderflow_config_4gpu.json --output-dir "$attempt/prepared" \
        2>&1 | tee "$attempt/prepare.log"
    mv -- "$attempt/prepared" "$PREPARED_DIR"
fi
