#!/usr/bin/env bash
# Evaluate completed Airfoil weights through the existing NAS wrapper.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
: "${RESULT_ROOT:?Set RESULT_ROOT to the completed Airfoil training run}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new sampling result directory}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
if [[ -n "${NAS_ROOT:-}" ]]; then
    export RAW_DATA_DIR=${RAW_DATA_DIR:-$NAS_ROOT/data/airfoil_raw}
    export DATA_DIR=${DATA_DIR:-$NAS_ROOT/data/airfoil_uvp_stride8}
else
    export RAW_DATA_DIR=${RAW_DATA_DIR:-/data/datasets/meshgraphnets/airfoil}
    export DATA_DIR=${DATA_DIR:-${RAW_DATA_DIR}_uvp_stride8}
fi
config=${CONFIG:-$RESULT_ROOT/config.json}
checkpoint=${CHECKPOINT:-$RESULT_ROOT/dynamics/best.pt}
ae_checkpoint=${AE_CHECKPOINT:-$RESULT_ROOT/ae/best.pt}
prepared=${PREPARED_DIR:-$RESULT_ROOT/prepared}
exec bash "$code_root/scripts/nas.sh" python sampling_ensemble.py \
    --config "$config" --checkpoint "$checkpoint" --ae-checkpoint "$ae_checkpoint" \
    --dataset "$DATA_DIR/airfoil_stride8_75frames.h5" \
    --manifest "$DATA_DIR/airfoil_stride8_75frames_manifest.json" --prepared "$prepared" \
    --output-dir "$OUTPUT_DIR" --device "${DEVICE:-cuda:0}" "$@"
