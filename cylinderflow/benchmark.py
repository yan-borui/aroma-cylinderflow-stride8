"""Benchmark a selected CylinderFlow dynamics checkpoint under the common protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import DATA_REPOSITORY, DATA_REVISION, Dataset
from .engine import config_from_file, load_ae, open_prepared, validate_dependency
from .models import make_model, rollout
from .performance import benchmark, configure, sync, write_json
from .runtime import ROOT, code_identity, load_checkpoint, monitor_indices


def benchmark_checkpoint(
    dataset: Dataset,
    config: dict,
    prepared: Path,
    checkpoint_file: Path,
    output_dir: Path,
    device: torch.device,
    ae_checkpoint: Path | None = None,
    threads: int = 2,
) -> dict:
    configure(device, threads)
    stats = open_prepared(dataset, config, prepared)
    started = time.perf_counter()
    checkpoint = load_checkpoint(checkpoint_file)
    stage = checkpoint["stage"]
    if stage == "ae":
        raise ValueError(
            "forecast benchmarking requires a selected dynamics checkpoint"
        )
    validate_dependency(checkpoint, config, stats, stage)
    model = make_model(checkpoint["config"], stage, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().float()
    ae = None
    if config["method"] == "aroma":
        if ae_checkpoint is None:
            raise ValueError("AROMA benchmark requires the selected AE")
        ae = load_ae(
            config, stats, ae_checkpoint, device, checkpoint["ae_checkpoint_id"]
        )[0]
    sync(device)
    model_load_seconds = time.perf_counter() - started

    def load_case(index: int) -> dict:
        sample = dataset.initial(index)
        sample["trajectory_index"] = index
        if config["method"] == "eagle":
            sample["clusters"] = np.load(
                prepared / "clusters" / f"trajectory_{index:04d}.npy",
                allow_pickle=False,
            )
        return sample

    def predict(sample: dict) -> np.ndarray:
        return rollout(
            model,
            checkpoint["config"],
            sample,
            stats,
            prepared,
            device,
            autoencoder=ae,
            measure_stages=False,
            return_raw=False,
        )[0]

    return benchmark(
        method=config["method"],
        indices=monitor_indices(dataset.splits["validation"]),
        load_case=load_case,
        predict=predict,
        device=device,
        output_dir=output_dir,
        data_identity={
            "repository": DATA_REPOSITORY,
            "revision": DATA_REVISION,
            "split": "validation",
            "phase_offset": 0,
            "raw_frame_indices": list(range(0, 513, 8)),
        },
        provenance={
            "checkpoint": str(checkpoint_file.resolve()),
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_update": checkpoint["updates"],
            "training_seed": checkpoint["settings"]["seed"],
            "configuration": checkpoint["config"],
            "ae_checkpoint_id": checkpoint.get("ae_checkpoint_id"),
            "normalization": stats,
            "evaluation_code": code_identity(),
        },
        models=[model] + ([ae] if ae is not None else []),
        model_load_seconds=model_load_seconds,
        formal=not dataset.debug,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "cylinderflow_config.json"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--debug-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    try:
        result = benchmark_checkpoint(
            Dataset(args.dataset, args.manifest, args.debug_data),
            config_from_file(args.config),
            args.prepared,
            args.checkpoint,
            args.output_dir,
            torch.device(args.device),
            args.ae_checkpoint,
            args.threads,
        )
    except Exception as error:
        write_json(
            args.output_dir / "exit.json",
            {"exit_code": 1, "error": f"{type(error).__name__}: {error}"},
        )
        raise
    code = 0 if result["status"] == "complete" else 1
    write_json(args.output_dir / "exit.json", {"exit_code": code})
    print(json.dumps(result, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
