# Native four-GPU training (2026-09-12)

This optional entry runs the existing CylinderFlow model/loss with four DDP ranks.
Use `cylinderflow_config_4gpu.json`; the original `cylinderflow_config.json` keeps
its single-device defaults. The four native stages have these explicit batches:

| Method / stage | Per-rank microbatch | Accumulation | Global batch | Budget | Learning rate |
| --- | ---: | ---: | ---: | --- | --- |
| MGN | 1 | 1 | 4 | 25 epochs / 462,500 updates | 1e-4 × 0.9999991^global_examples_seen |
| EAGLE | 1 | 1 | 4 | 1,000 epochs / 250,000 updates | constant 1e-4 |
| AROMA AE | 1 | 16 | 64 | 10,000 epochs / 160,000 updates | native epoch cosine, 1e-3 to 1e-5 |
| AROMA dynamics | 32 | 1 | 128 | 5,000 epochs / 2,960,000 updates | native epoch cosine, 1e-3 to 1e-6 |

The user explicitly accepted MGN B1→B4 and EAGLE B2→B4. MGN keeps its original
LR at a given global exposure count: exponent 0 at the first update, then
4, 8, …; its end LR is approximately 1.89e-5. It does not apply the decay just
once per four-example update. Other optimizers, epoch budgets, native losses,
noise, and validation cadence keep their original recipe. These methods use FP32,
seed 123. The H1 search LR floor does not override native baseline schedules.

The DDP reducer averages gradients across ranks. The adapter's native losses
already normalize by the global node/sample denominator, so the DDP backward
multiplies each rank's contribution by world size. Accumulation uses `no_sync`.
AROMA preserves its partial final group (AE: 40 frames; dynamics: 104
trajectories), including the actual denominator and exposure count.

## Preparation and launch

Use the existing environment and official preparation commands in
[CYLINDERFLOW.md](CYLINDERFLOW.md). Set `DATA`, `MANIFEST`, and `PREPARED`
to those outputs, and `GPU_IDS` to four allocated IDs/UUIDs. All commands run
from this repository root. Formal data checks and Test isolation remain active.

Run the maximum-Train-graph check in a fresh directory:

```bash
python -m cylinderflow.run_four_gpu --dataset "$DATA" --manifest "$MANIFEST" \
  --prepared "$PREPARED" --gpus "$GPU_IDS" \
  --output-dir runs/preflight_4gpu_new --preflight
```

This executes eight actual DDP updates and a native prediction on the largest
Train graph on every rank. AROMA defaults to AE for this check. Check dynamics
separately with a frozen AE and its matching prepared Train latent cache:

```bash
python -m cylinderflow.run_four_gpu --dataset "$DATA" --manifest "$MANIFEST" \
  --prepared "$PREPARED_WITH_LATENTS" --gpus "$GPU_IDS" \
  --output-dir runs/preflight_dynamics_4gpu_new --preflight \
  --stage dynamics --ae-checkpoint "$SELECTED_AE"
```

Formal native-budget entry:

```bash
python -m cylinderflow.run_four_gpu --dataset "$DATA" --manifest "$MANIFEST" \
  --prepared "$PREPARED" --gpus "$GPU_IDS" --output-dir runs/formal_4gpu_new
```

Or submit one four-GPU Slurm job per baseline; the scheduler sets the GPU mask:

```bash
sbatch scripts/slurm_four_gpu.sh "$PYTHON" --dataset "$DATA" --manifest "$MANIFEST" \
  --prepared "$PREPARED" --output-dir "$RESULT_ROOT"
```

AROMA holds one allocation through AE → selected AE → Train posterior-mean
cache → dynamics. It creates a run-local prepared directory, preserves the
Train normalization, and binds the cache/dynamics to the selected AE identity.
A failed cache build retains its partial file; retry writes a new partial file.

## Checkpoints, validation, and recovery

A global deterministic epoch order is sharded across ranks without padding or
duplicating training examples. Every rank has a separate stochastic stream.
Each checkpoint stores unwrapped native model keys, optimizer/scheduler/scaler,
global epoch/group/example cursor, all rank RNG states, immutable config/data/AE
identities, and source snapshot. The new format is
`cylinderflow.native_ddp.training.v1`; the existing standalone model evaluator
can read its raw weights. Four-rank resume requires the same source, settings,
world size, data and AE.

Validation runs at the original epoch milestones: MGN every epoch, EAGLE every
50, AROMA AE every 500 and dynamics every 250. The same fixed Validation subset
and native selection metric are retained; cases are sharded across four ranks.
Evaluation restores model mode and training RNG. Completed trajectory results
are committed atomically and reused on recovery. A pre-evaluation recovery point
records pending work; a program failure exits with per-rank evidence and cannot
be silently treated as a valid score. Numeric failure records keep the original
native metric/selection treatment.

Resume the same launcher command and result directory with `--resume`. Completed
stages are reused. Source/config edits require a new run. Normal training has no
metric-based early stopping; `--preflight` is an explicitly bounded acceptance
run and uses a separate directory.

Each stage writes `metrics.jsonl`, `status.json`, `run_manifest.json`,
`last.pt`, `best.pt`, `selector.json`, `checkpoint_inventory.json`,
epoch checkpoints, physical predictions and per-case failures. Launcher attempts
have separate logs and exit records. Cost records distinguish training,
validation, elapsed allocation time, and per-rank allocated/reserved memory.
AROMA AE reconstruction and dynamics forecasting are labeled separately.

After training, the existing standalone `cylinderflow evaluate --mode validation`
entry in [CYLINDERFLOW.md](CYLINDERFLOW.md) evaluates the frozen selected model
on full Validation. The launcher ends after native training/selection; it does
not authorize Test access. Training selection remains scoped to its Validation
subset, and global-batch changes must be disclosed in comparisons.

## Verification status

The shared engine passed a two-rank CPU/Gloo acceptance using a controlled
linear model: independent global node-weighted gradient reference, accumulation,
partial final batch, exact Adam/weights/rank-RNG/cursor recovery, evaluation RNG
isolation, and interrupted pending Validation recovery.

```bash
python -m cylinderflow.ddp_acceptance --output-dir runs/cpu_ddp_acceptance_new
```

This checks the distributed engine with a controlled model. Actual native
backbones, four-GPU NCCL, maximum graph, full physical inference, memory, and
throughput need the target environment and the preflight commands above.
This source-code delivery has not launched formal training.
