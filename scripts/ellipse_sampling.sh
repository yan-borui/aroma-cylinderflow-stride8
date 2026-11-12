#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
action=${1:-quality}
if (( $# > 0 )); then shift; fi
case "$action" in quality|benchmark) ;; *) printf 'Use quality or benchmark\n' >&2; exit 2 ;; esac
: "${RESULT_ROOT:?Set the completed training result root}"
: "${OUTPUT_DIR:?Set a new output directory}"
python_bin=${PYTHON:-python}
args=(--method aroma --output-dir "$OUTPUT_DIR" --split "${SPLIT:-validation}")
: "${DATA_DIR:?Set the prepared Ellipse dataset directory}"
args+=(--dataset "$DATA_DIR/ellipse_stride15_65frames.h5" --manifest "$DATA_DIR/ellipse_stride15_65frames_manifest.json"
       --config cylinderflow_config_4gpu.json --prepared "${PREPARED_DIR:-$RESULT_ROOT/prepared}")
args+=(--checkpoint "${CHECKPOINT:-$RESULT_ROOT/dynamics/best.pt}")
args+=(--ae-checkpoint "${AE_CHECKPOINT:-$RESULT_ROOT/ae/best.pt}")
if [[ "$action" == benchmark ]]; then
    exec "$python_bin" ellipse_sampling.py "${args[@]}" --benchmark "$@"
fi
exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=4 ellipse_sampling.py "${args[@]}" "$@"
