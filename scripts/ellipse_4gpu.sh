#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
python_bin=${PYTHON:-python}
action=${1:-train}
case "$action" in
    data|prepare|train|resume|validate|test) ;;
    *) printf 'Usage: bash scripts/ellipse_4gpu.sh data|prepare|train|resume|validate|test\n' >&2; exit 2 ;;
esac
export RAW_DATA_DIR=${RAW_DATA_DIR:-/data/datasets/megaflow2d}
export DATA_DIR=${DATA_DIR:-${RAW_DATA_DIR}_ellipse_stride15}
if [[ "$action" != data && "$action" != prepare ]]; then
    : "${RESULT_ROOT:?Set RESULT_ROOT for this training run}"
    : "${CUDA_VISIBLE_DEVICES:?Set four allocated GPU IDs or use the scheduler mask}"
    IFS=',' read -r -a assigned <<< "$CUDA_VISIBLE_DEVICES"
    if (( ${#assigned[@]} != 4 )); then printf 'Exactly four GPUs are required.\n' >&2; exit 2; fi
    if [[ "$action" == resume && ! -d "$RESULT_ROOT" ]]; then
        printf 'Resume requires the existing RESULT_ROOT.\n' >&2; exit 2
    fi
    if [[ "$action" == train && -e "$RESULT_ROOT" ]]; then
        printf 'Choose a new RESULT_ROOT or use resume.\n' >&2; exit 2
    fi
fi
data_options=()
if [[ "${ELLIPSE_OFFLINE:-0}" == 1 ]]; then data_options=(--offline); fi
"$python_bin" -m megaflow_data.ensure --raw-dir "$RAW_DATA_DIR" --output-dir "$DATA_DIR" "${data_options[@]}"
if [[ "$action" == data ]]; then exit 0; fi
data="$DATA_DIR/ellipse_stride15_65frames.h5"
manifest="$DATA_DIR/ellipse_stride15_65frames_manifest.json"
extra=()
if [[ "$action" == resume ]]; then extra=(--resume); fi
split=validation
if [[ "$action" == test ]]; then split=test; fi
if [[ "$action" == validate || "$action" == test ]]; then
    export EVAL_DIR=${EVAL_DIR:-$RESULT_ROOT/${split}_ellipse_4gpu}
fi


export PREPARED_DIR=${PREPARED_DIR:-$DATA_DIR/prepared_aroma_4gpu}
mkdir -p "$(dirname -- "$PREPARED_DIR")"
"$python_bin" "$code_root/airfoil_data/portable_lock.py" --wait --lock "${PREPARED_DIR}.lock" -- \
    bash "$code_root/scripts/prepare_ellipse_cache.sh"
if [[ "$action" == prepare ]]; then exit 0; fi
if [[ "$action" == validate || "$action" == test ]]; then
    checkpoint=${CHECKPOINT:-$RESULT_ROOT/dynamics/best.pt}
    ae_options=()
    ae_options=(--ae-checkpoint "${AE_CHECKPOINT:-$RESULT_ROOT/ae/best.pt}")
    export PREPARED_DIR="$RESULT_ROOT/prepared"
    exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=4 \
        --module cylinderflow.evaluate_ellipse --dataset "$data" --manifest "$manifest" \
        --prepared "$PREPARED_DIR" --config cylinderflow_config_4gpu.json \
        --checkpoint "$checkpoint" --split "$split" --output-dir "$EVAL_DIR" "${ae_options[@]}"
fi
exec "$python_bin" -m cylinderflow.run_four_gpu --dataset "$data" --manifest "$manifest" \
    --prepared "$PREPARED_DIR" --output-dir "$RESULT_ROOT" --config cylinderflow_config_4gpu.json "${extra[@]}"
