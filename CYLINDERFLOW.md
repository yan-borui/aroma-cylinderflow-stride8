# CylinderFlow installation and operation

Run these commands from this clone's root. The adapter imports the pinned upstream model in this repository and has no dependency on another checkout. Python 3.11 and Linux x86-64 are the acceptance environment. Install one device-specific dependency file into a new environment; there is no W&B or other online account requirement for training. Initial package/data downloads need network access; subsequent commands are offline.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cylinderflow-cpu.txt
python -m cylinderflow doctor --device cpu --output-dir runs/doctor-cpu
python -m unittest discover -s tests_cylinderflow -v
python -m cylinderflow smoke --device cpu --output-dir runs/smoke-cpu
```

For a target GPU, create another environment and install `requirements-cylinderflow-gpu.txt`. These pins use PyTorch 2.4.0 with CUDA 11.8, plus the matching DGL or torch-scatter wheel where required. GPU architecture/driver compatibility and maximum-mesh memory must be checked on the target cluster. CPU acceptance does not certify GPU execution, mixed precision, or large effective batches. The CPU smoke keeps the **full model architecture**, uses three tiny analytic variable-node meshes, and verifies training/resume, full 64-step inference and actual PNG/GIF/MP4 generation. Its outputs are software checks, not model-quality results.

## Data preparation

Download the two fixed-revision data files, or copy those same files into your own directory. The fetch command creates a fresh directory and downloads no Test or pretrained weights:

```bash
python -m cylinderflow.fetch --output-dir data/stride8
export DATA="$PWD/data/stride8/cylinderflow_stride8_75frames.h5"
export MANIFEST="$PWD/data/stride8/cylinderflow_stride8_75frames_manifest.json"
python -m cylinderflow prepare --dataset "$DATA" --manifest "$MANIFEST" --output-dir runs/prepared --seed 123
```

`prepare` creates `normalization.json` from Train frames 0..74, data identity and static graph caches (MGN/EAGLE), plus author-constrained clusters for EAGLE. AROMA's Train-only latent cache is a separate `prepare --latents` step after AE selection. `--debug-data` explicitly relaxes the official trajectory-count requirement for development fixtures and records that status; it must not be used for a formal run. The normalizer is embedded in checkpoints and checked against the active data contract.

## Training, recovery and resource allocation

Use the method-specific commands at the top of `README.md`. `cylinderflow_config.json` holds method-native budgets, learning rates, checkpoint cadence and monitoring cadence. To change an experiment, copy this configuration and pass `--config my_config.json` to all commands. Paths are supplied by CLI. `--seed`, `--device`, `--precision fp32|bf16|fp16`, `--microbatch`, `--accumulation`, and `--threads` are explicit runtime settings. BF16/FP16 are optional and require target acceptance; FP32 is the verified default. `--max-updates N` is an explicit total-update cap for short checks, including previous updates on resume; it is not a loss/health stopping rule. A loss or gradient that becomes nonfinite is an unavoidable numerical failure and is logged with a nonzero exit code.

Each GPU runs an independent task with its own directory. A scheduler can set `CUDA_VISIBLE_DEVICES` and the command can use `--device cuda`; no GPU number, host directory, WSL runtime, proxy, DDP world size, or online experiment account is embedded. Use a distinct `--output-dir` for each task and keep each preparation directory immutable. Only `resume` reuses a run directory, requires `--checkpoint`, and checks configuration, normalizer, AE dependency and run identity. An OS lock prevents concurrent writers to the same run. Relative directories and arbitrary absolute directories are supported. Checkpoints retain failed-run evidence; no automatic restart, scheduler service or long training is launched by setup or smoke.

`last.pt` is an atomic recovery checkpoint; `epoch_*.pt` preserves monitored candidates and `best.pt`/`selector.json` retain the selected candidate. Records include optimizer, scheduler, precision scaler, Python/NumPy/PyTorch random states and the exact epoch/sample cursor. A changed batch, seed or learning-rate configuration requires a new run, so resume cannot silently change the experiment. `metrics.jsonl` logs each update by default, including loss components, gradient norm, LR, examples, time, and CUDA memory. `events.jsonl` records resume boundaries. CLI commands write `exit.json`; failures append their exception and traceback to `command_failures.jsonl`. If the OS kills a process before Python can write its exit marker, the missing marker and retained partial records expose that interruption; the cluster launcher should also retain its process exit status.

## Evaluation and standalone tools

Select on the fixed 24 Validation trajectories, then evaluate one locked checkpoint on the full 100. For AROMA add `--ae-checkpoint runs/ae/best.pt` to both commands:

```bash
python -m cylinderflow evaluate --dataset "$DATA" --manifest "$MANIFEST" --prepared runs/prepared --checkpoint runs/train/best.pt --mode select --device cuda --output-dir runs/selection
python -m cylinderflow evaluate --dataset "$DATA" --manifest "$MANIFEST" --prepared runs/prepared --checkpoint runs/train/best.pt --mode validation --device cuda --output-dir runs/validation
```

`--checkpoint` accepts several candidate files in selection mode. Read `evaluation.json` to identify the chosen checkpoint. The all-Validation command permits exactly one already-selected checkpoint and never exposes Test. It produces 100 MGN/EAGLE clips or 300 AROMA clips. Each clip is a compressed physical-mesh NPZ with all 65 states, target, pre-boundary predictions, mesh and exact time indices. Metric JSON/CSV retain failures, seed-level values and trajectory-level aggregation. Numerical/OOM failures are counted; infrastructure/programming errors preserve the failing case record and return a nonzero exit code.

The following commands use saved arrays/logs and never train a model:

```bash
python -m cylinderflow score --inputs runs/validation/candidate_000/predictions/*.npz --output-dir runs/rescored
python -m cylinderflow render --inputs runs/validation/candidate_000/predictions/trajectory_1000_seed_0.npz --labels model --output-dir runs/media
python -m cylinderflow plot-curves --run runs/train --output-dir runs/curves
```

Pass multiple methods' NPZ files for the **same case** to `render --inputs ... --labels AROMA MGN EAGLE`. The renderer verifies identical GT, mesh and time indices, then uses fixed shared scales across all 65 frames and all supplied predictions. It renders GT/prediction/error for speed, gauge-free pressure and triangle vorticity. `--scales previous/scales.json` reuses fixed bounds in a separate rendering; values outside those explicit bounds saturate. `--every 1` renders every frame; a larger stride keeps the last frame and reduces file size. PNG snapshots, GIF and MP4 are generated together. Loss/Validation plots show component losses, UV score, failed clips, LR, gradients, elapsed time, resume markers, and selected checkpoint markers when available.

## Target-cluster maximum-graph check

After `prepare`, explicitly run:

```bash
python -m cylinderflow smoke --largest-graph --dataset "$DATA" --manifest "$MANIFEST" --prepared runs/prepared --device cuda --output-dir runs/gpu-largest
```

MGN/EAGLE perform one backward/update and 64-step prediction on the largest Train mesh, with microbatch 1, recording finite status, timings and peak memory. AROMA runs the AE stage by default; additionally run with `--stage dynamics --ae-checkpoint runs/ae/best.pt` after caching latents to check its 64-step path. This preflight uses disposable random models and never replaces a formal checkpoint. It certifies only its recorded device, precision and batch; increase resource use deliberately after that evidence. Formal runs retain the full requested budgets unless explicitly configured otherwise.

Keep data, training weights, full predictions and structured logs on the training host. PNG/GIF/MP4 and curves support image-only review; whether logs/metric CSV can leave that host remains an operational agreement with its owner. This workflow generates no narrative report. Preserve upstream history and license; after cloning the private copy, restore the upstream URL with the command in the README because Git remotes are local clone configuration.

The 2026-09-05 protocol trains on all 75 frames and evaluates only the first 65.
Use a new prepared/result directory: old prefix65 normalizers, latent caches, and
checkpoints retain their prior identity and cannot resume into the new protocol.
Current checks are recorded in `ALIGNMENT_VERIFICATION.json`.
