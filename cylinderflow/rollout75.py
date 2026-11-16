"""Generate 75 physical frames with native uninterrupted autoregressive rollout."""

from __future__ import annotations

import argparse
import json
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch

from mgn_data.test_data import TestData
from .data import DT, Dataset
from .engine import config_from_file, load_ae, open_prepared, validate_dependency
from .metrics import compute_metrics, summarize_trajectories
from .models import make_model, prepare_clusters, rollout
from .runtime import (
    ROOT,
    code_identity,
    load_checkpoint,
    sample_seed,
    seed_everything,
    synchronize,
    write_csv,
    write_json,
)


def main(*, test_frames: int = 65) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "cylinderflow_config.json"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path)
    parser.add_argument("--scope", choices=("validation", "test"), default="validation")
    parser.add_argument("--test-data-dir", type=Path)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--case-count", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.test_frames = test_frames
    if min(args.case_count, args.threads) < 1:
        parser.error("case-count and threads must be positive")
    if (args.scope == "test") != (args.test_data_dir is not None):
        parser.error("only --scope test requires --test-data-dir")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("use the assigned production CUDA environment")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "request.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    try:
        run(args, device)
    except Exception as error:
        write_json(
            args.output_dir / "failure.json",
            {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def run(args: argparse.Namespace, device: torch.device) -> None:
    """Load fixed weights once, generate each trajectory, then read references."""
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = config_from_file(args.config)
    training = Dataset(args.dataset, args.manifest)
    stats = open_prepared(training, config, args.prepared)
    checkpoint = load_checkpoint(args.checkpoint)
    stage = "dynamics" if config["method"] == "aroma" else "main"
    validate_dependency(checkpoint, config, stats, stage)
    config = checkpoint["config"]
    model = make_model(config, stage, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().float().requires_grad_(False)
    autoencoder = None
    if stage == "dynamics":
        if args.ae_checkpoint is None:
            raise ValueError("AROMA requires the matching --ae-checkpoint")
        autoencoder, _ = load_ae(
            config,
            stats,
            args.ae_checkpoint,
            device,
            checkpoint["ae_checkpoint_id"],
        )
        autoencoder.eval().float().requires_grad_(False)
    data = training
    if args.scope == "test":
        data = TestData(
            args.test_data_dir,
            "cylinderflow",
            expected_frames=args.test_frames,
        )
        data.bind_training_manifest(training.manifest)
    population = list(data.splits[args.scope])
    indices = (
        args.indices if args.indices is not None else population[: args.case_count]
    )
    if (
        not indices
        or len(set(indices)) != len(indices)
        or not set(indices) <= set(population)
    ):
        raise ValueError("indices must be unique global IDs in the selected split")
    training_seed = checkpoint["settings"]["seed"]
    seeds = (0, 1, 2) if config["method"] == "aroma" else (0,)
    prepared = args.prepared
    identity = {
        "protocol": "cylinderflow.native_rollout75.v1",
        "method": config["method"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_id": checkpoint["checkpoint_id"],
        "update": checkpoint["updates"],
        "ae_checkpoint": str(args.ae_checkpoint.resolve())
        if args.ae_checkpoint
        else None,
        "ae_checkpoint_id": checkpoint.get("ae_checkpoint_id"),
        "configuration": config,
        "normalization": stats,
        "training_code": checkpoint.get("code"),
        "evaluation_code": code_identity(),
        "training_data": training.identity(),
        "evaluation_data": data.identity(),
        "scope": args.scope,
        "test_accessed": args.scope == "test",
        "indices": indices,
        "sampling_labels": list(seeds),
        "training_seed": training_seed,
        "seed_rule": "native sample_seed(training_seed, trajectory_index, sampling_label)",
        "observed_frames": [0],
        "predicted_frames": [1, 74],
        "reference_frames": [
            0,
            74 if args.scope == "validation" else args.test_frames - 1,
        ],
        "frame_dt": DT,
        "duration_seconds": 74 * DT,
        "aggregation": "score independent samples; average scores within trajectory, then equally across trajectories",
        "boundary": "native handling; inlet/wall UV fixed to observed frame zero",
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "precision": "fp32",
        },
    }
    write_json(args.output_dir / "identity.json", identity)
    with zipfile.ZipFile(
        args.output_dir / "source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for package in ("cylinderflow", "mgn_data", "aroma", "Models", "modulus"):
            for source in sorted((ROOT / package).rglob("*.py")):
                archive.write(source, source.relative_to(ROOT))
        for name in ("clusterize_eagle.py", "cylinderflow_config.json", "ROLLOUT75.md"):
            if (ROOT / name).is_file():
                archive.write(ROOT / name, name)
    if config["method"] == "eagle" and args.scope == "test":
        prepared = args.output_dir / "geometry"
        prepared.mkdir()
        prepare_clusters(data, prepared / "clusters", training_seed, indices=indices)
    rows = []
    for index in indices:
        for label in seeds:
            output = args.output_dir / f"trajectory_{index:04d}_seed_{label}"
            output.mkdir()
            derived = sample_seed(training_seed, index, label)
            row = {
                "trajectory_index": index,
                "seed": label,
                "sample_seed": derived,
                "finite": False,
                "prediction_complete": False,
            }
            try:
                initial = data.initial(index)
                seed_everything(derived)
                synchronize(device)
                begin = time.perf_counter()
                diagnostics = {}
                prediction, raw, timings = rollout(
                    model,
                    config,
                    initial,
                    stats,
                    prepared,
                    device,
                    autoencoder,
                    diagnostics,
                    future_frames=74,
                )
                synchronize(device)
                row["inference_seconds"] = time.perf_counter() - begin
                row["stage_timings"] = timings
                if prediction.shape != (75, len(initial["points"]), 3):
                    raise ValueError("native rollout must return [75,N,3]")
                # Save generation before loading reference fields or scoring.
                np.savez_compressed(
                    output / "prediction.npz",
                    prediction_schema=np.asarray("cylinderflow.native_rollout75.v1"),
                    units=np.asarray("physical_uvp"),
                    prediction=prediction,
                    pre_boundary=raw,
                    points=initial["points"],
                    cells=initial["cells"],
                    node_type=initial["node_type"],
                    raw_frame_indices=np.arange(75) * 8,
                    physical_time=np.arange(75) * DT,
                    trajectory_index=index,
                    seed=label,
                    sampler_seed=derived,
                    **diagnostics,
                )
                if not np.isfinite(prediction).all():
                    raise FloatingPointError("nonfinite 75-frame prediction")
                row["prediction_complete"] = True
                target = (
                    data.read(index, 0, 75)["field"]
                    if args.scope == "validation"
                    else data.evaluation(index)["field"]
                )
                np.savez_compressed(
                    output / "reference.npz",
                    target=target,
                    physical_time=np.arange(len(target)) * DT,
                )
                geometry = (
                    initial["points"],
                    initial["cells"],
                    initial["node_type"],
                    DT,
                )
                row["prefix64"] = compute_metrics(
                    prediction[:65], target[:65], *geometry
                )
                if len(target) == 75:
                    row["full74"] = compute_metrics(
                        prediction,
                        target,
                        *geometry,
                        expected_frames=75,
                    )
                    row["full74"]["frame74_uv_relative_rmse"] = row["full74"][
                        "per_frame_uv_relative_rmse"
                    ][74]
                scored = [row[name] for name in ("prefix64", "full74") if name in row]
                row["finite"] = all(
                    metric.get("finite")
                    and metric.get("uv_relative_rmse") is not None
                    and np.isfinite(metric["uv_relative_rmse"])
                    for metric in scored
                )
                if not row["finite"]:
                    raise FloatingPointError("nonfinite scored metric")
            except Exception as error:
                row["finite"] = False
                row["error"] = f"{type(error).__name__}: {error}"
                write_json(
                    output / "failure.json",
                    {
                        "error": row["error"],
                        "traceback": traceback.format_exc(),
                    },
                )
            write_json(output / "metrics.json", row)
            rows.append(row)
            summaries = {}
            for name in (
                ("prefix64", "full74")
                if args.scope == "validation" or args.test_frames == 75
                else ("prefix64",)
            ):
                metric_rows = [
                    {
                        **item.get(name, {}),
                        "trajectory_index": item["trajectory_index"],
                        "seed": item["seed"],
                        "finite": item["finite"],
                    }
                    for item in rows
                ]
                summaries[name] = summarize_trajectories(metric_rows)
                write_csv(
                    args.output_dir / f"{name}_trajectory_metrics.csv",
                    summaries[name]["trajectory_metrics"],
                )
            write_json(
                args.output_dir / "summary.json",
                {
                    "requested_indices": indices,
                    "sampling_labels": list(seeds),
                    "requested_samples": len(indices) * len(seeds),
                    "completed_samples": len(rows),
                    "failed_samples": sum(not item["finite"] for item in rows),
                    "metric_future_frames": {"prefix64": 64, "full74": 74}
                    if args.scope == "validation" or args.test_frames == 75
                    else {"prefix64": 64},
                    "aggregate": summaries,
                    "cases": rows,
                },
            )
            print(
                json.dumps(
                    {
                        "trajectory_index": index,
                        "seed": label,
                        "finite": row["finite"],
                        "completed_samples": len(rows),
                    }
                ),
                flush=True,
            )
    if any(not row["finite"] for row in rows):
        raise RuntimeError("rollout failures retained in per-sample records")


if __name__ == "__main__":
    main()
