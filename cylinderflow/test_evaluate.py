"""Evaluate one selected forecasting checkpoint on the official Test100."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mgn_data.test_data import TestData
from . import EVALUATOR_VERSION, PROTOCOL_VERSION
from .data import DT, Dataset
from .engine import (
    config_from_file,
    evaluate_model,
    load_ae,
    open_prepared,
    validate_dependency,
)
from .models import make_model, prepare_clusters
from .runtime import ROOT, code_identity, load_checkpoint, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "cylinderflow_config.json"
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="original Train/Validation HDF5"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="original training manifest"
    )
    parser.add_argument(
        "--prepared", type=Path, required=True, help="existing training preparation"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path)
    parser.add_argument("--test-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    config = config_from_file(args.config)
    training = Dataset(args.dataset, args.manifest)
    stats = open_prepared(training, config, args.prepared)
    data = TestData(args.test_data_dir, "airfoil" if DT == 0.0016 else "cylinderflow")
    data.bind_training_manifest(training.manifest)
    checkpoint = load_checkpoint(args.checkpoint)
    stage = checkpoint["stage"]
    if stage not in {"main", "dynamics"}:
        raise ValueError(
            "Test forecasting requires a selected main/dynamics checkpoint"
        )
    validate_dependency(checkpoint, config, stats, stage)
    # Inference options are those saved with the selected weights.
    config = checkpoint["config"]
    model = make_model(config, stage, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    autoencoder = None
    if stage == "dynamics":
        if args.ae_checkpoint is None:
            parser.error("AROMA dynamics requires its matching --ae-checkpoint")
        autoencoder, _ = load_ae(
            config, stats, args.ae_checkpoint, device, checkpoint["ae_checkpoint_id"]
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prepared = args.prepared
    if config["method"] == "eagle":
        prepared = args.output_dir / "geometry"
        prepared.mkdir()
        prepare_clusters(
            data,
            prepared / "clusters",
            checkpoint["settings"]["seed"],
            indices=data.splits["test"],
        )
    provenance = {
        "mode": "test",
        "test_accessed": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_update": checkpoint["updates"],
        "ae_checkpoint_id": checkpoint.get("ae_checkpoint_id"),
        "training_code": checkpoint.get("code"),
        "evaluation_code": code_identity(),
        "training_data_identity": training.identity(),
        "evaluation_data_identity": data.identity(),
    }
    write_json(args.output_dir / "evaluation_identity.json", provenance)
    summary = evaluate_model(
        model,
        config,
        stage,
        data,
        stats,
        prepared,
        data.splits["test"],
        device,
        checkpoint["settings"]["seed"],
        args.output_dir / "test",
        autoencoder,
        provenance=provenance,
    )
    result = {
        **provenance,
        "protocol": PROTOCOL_VERSION,
        "evaluator": EVALUATOR_VERSION,
        "state": "test_complete"
        if summary["failed_clips"] == 0
        else "test_complete_with_failures",
        "summary": summary,
        "sampling_seeds": [0, 1, 2] if config["method"] == "aroma" else [0],
        "aggregation": "native metrics; equal trajectory weighting",
    }
    write_json(args.output_dir / "evaluation.json", result)
    if summary["failed_clips"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
