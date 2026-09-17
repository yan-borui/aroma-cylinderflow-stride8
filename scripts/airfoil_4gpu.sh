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
    data|prepare|train|resume) ;;
    *) printf 'Usage: bash scripts/airfoil_4gpu.sh data|prepare|train|resume\n' >&2; exit 2 ;;
esac
export RAW_DATA_DIR=${RAW_DATA_DIR:-/data/datasets/meshgraphnets/airfoil}
export DATA_DIR=${DATA_DIR:-${RAW_DATA_DIR}_uvp_stride8}
if [[ "$action" == train || "$action" == resume ]]; then
    : "${RESULT_ROOT:?Set RESULT_ROOT for this training run}"
    : "${CUDA_VISIBLE_DEVICES:?Set four allocated GPU IDs or use the scheduler mask}"
    IFS=',' read -r -a assigned <<< "$CUDA_VISIBLE_DEVICES"
    if (( ${#assigned[@]} != 4 )); then printf 'Exactly four GPUs are required.\n' >&2; exit 2; fi
    if [[ "$action" == train && -e "$RESULT_ROOT" ]]; then
        printf 'Choose a new RESULT_ROOT or use resume.\n' >&2; exit 2
    fi
fi
data_options=()
if [[ "${AIRFOIL_OFFLINE:-0}" == 1 ]]; then data_options=(--offline); fi
"$python_bin" -m airfoil_data.ensure --raw-dir "$RAW_DATA_DIR" --output-dir "$DATA_DIR" "${data_options[@]}"
if [[ "$action" == data ]]; then exit 0; fi
data="$DATA_DIR/airfoil_stride8_75frames.h5"
manifest="$DATA_DIR/airfoil_stride8_75frames_manifest.json"
extra=()
if [[ "$action" == resume ]]; then extra=(--resume); fi

export PREPARED_DIR=${PREPARED_DIR:-$DATA_DIR/prepared_aroma_4gpu}
mkdir -p "$(dirname -- "$PREPARED_DIR")"
(
    flock 9
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
) 9>"${PREPARED_DIR}.lock"
if [[ "$action" == prepare ]]; then exit 0; fi
exec "$python_bin" -m cylinderflow.run_four_gpu --dataset "$data" --manifest "$manifest" \
    --prepared "$PREPARED_DIR" --output-dir "$RESULT_ROOT" --config cylinderflow_config_4gpu.json "${extra[@]}"
