"""Four-GPU evaluation of fixed Validation-selected Ellipse weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import Dataset
from .distributed import Context
from .engine import (
    config_from_file,
    evaluate_model,
    load_ae,
    open_prepared,
    validate_dependency,
)
from .metrics import summarize_trajectories
from .models import make_model, prepare_clusters
from .runtime import code_identity, load_checkpoint, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "dataset",
        "manifest",
        "prepared",
        "checkpoint",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    context = Context(4, "cuda")
    try:
        config = config_from_file(args.config)
        data = Dataset(args.dataset, args.manifest)
        stats = open_prepared(data, config, args.prepared)
        checkpoint = load_checkpoint(args.checkpoint)
        stage = checkpoint["stage"]
        if stage not in ("main", "dynamics"):
            raise ValueError("forecast evaluation requires a main/dynamics checkpoint")
        validate_dependency(checkpoint, config, stats, stage)
        config = checkpoint["config"]
        model = make_model(config, stage, context.device)
        model.load_state_dict(checkpoint["model"], strict=True)
        autoencoder = None
        if stage == "dynamics":
            if args.ae_checkpoint is None:
                raise ValueError("AROMA requires the selected matching AE")
            autoencoder, _ = load_ae(
                config,
                stats,
                args.ae_checkpoint,
                context.device,
                checkpoint["ae_checkpoint_id"],
            )
        indices = data.splits[args.split]
        context.primary_call(
            lambda: args.output_dir.mkdir(parents=True, exist_ok=False)
        )
        prepared = args.prepared
        if config["method"] == "eagle":
            prepared = args.output_dir / "geometry"
            context.primary_call(
                lambda: prepare_clusters(
                    data,
                    prepared / "clusters",
                    checkpoint["settings"]["seed"],
                    indices=indices,
                )
            )
        provenance = {
            "split": args.split,
            "test_accessed": args.split == "test",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_update": checkpoint["updates"],
            "ae_checkpoint_id": checkpoint.get("ae_checkpoint_id"),
            "data_identity": data.identity(),
            "evaluation_code": code_identity(),
            "world_size": 4,
            "boundary_policy": "predict_all_uvp_nodes",
        }
        seeds = [0, 1, 2] if config["method"] == "aroma" else [0]

        def evaluate_shard():
            folder = args.output_dir / f"rank_{context.rank:03d}"
            evaluate_model(
                model,
                config,
                stage,
                data,
                stats,
                prepared,
                indices[context.rank :: context.world],
                context.device,
                checkpoint["settings"]["seed"],
                folder,
                autoencoder,
                provenance=provenance,
            )
            rows = [
                json.loads(line)
                for line in (folder / "case_metrics.jsonl").read_text().splitlines()
            ]
            for row in rows:
                if "prediction_file" in row:
                    row["prediction_file"] = str(
                        (folder / row["prediction_file"]).relative_to(args.output_dir)
                    )
            return rows

        shards = context.all_call(evaluate_shard)

        def merge():
            rows = sorted(
                [row for shard in shards for row in shard],
                key=lambda row: (row["trajectory_index"], row["seed"]),
            )
            expected = {(index, seed) for index in indices for seed in seeds}
            actual = [(row["trajectory_index"], row["seed"]) for row in rows]
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("missing or duplicate Ellipse evaluation trajectories")
            summary = summarize_trajectories(rows)
            summary.update(provenance=provenance, sampling_seeds=seeds)
            write_json(args.output_dir / "summary.json", summary)
            write_json(args.output_dir / "cases.json", {"rows": rows})
            write_csv(args.output_dir / "case_metrics.csv", rows)
            write_csv(
                args.output_dir / "trajectory_metrics.csv",
                summary["trajectory_metrics"],
            )
            write_json(
                args.output_dir / "failures.json",
                {"failures": [row for row in rows if not row["finite"]]},
            )
            return summary

        summary = context.primary_call(merge)
        if summary["failed_clips"]:
            raise SystemExit(1)
    finally:
        context.close()


if __name__ == "__main__":
    main()
