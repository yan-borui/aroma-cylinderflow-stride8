"""Method-native training, deterministic cursor resume, and final evaluation."""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch

from . import EVALUATOR_VERSION, PROTOCOL_VERSION
from .data import DT, Dataset, compute_statistics, load_statistics
from .metrics import (
    compute_metrics,
    node_area_weights,
    selection_key,
    summarize_trajectories,
)
from .models import (
    aroma_inputs,
    make_model,
    prepare_clusters,
    rollout,
    tensor,
    training_loss,
)
from .runtime import (
    append_json,
    autocast,
    code_identity,
    load_checkpoint,
    monitor_indices,
    peak_memory,
    read_jsonl,
    restore_rng,
    rng_state,
    sample_seed,
    save_checkpoint,
    seed_everything,
    synchronize,
    write_csv,
    write_json,
)


def config_from_file(file_name: str | Path) -> dict:
    config = json.loads(Path(file_name).read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("config protocol mismatch")
    from .runtime import ROOT

    upstream = json.loads((ROOT / "cylinderflow_upstream.json").read_text())
    if config.get("method") != upstream["method"]:
        raise ValueError("this clone only supports its own method")
    return config


def validate_dependency(checkpoint: dict, config: dict, stats: dict, stage: str):
    if (
        checkpoint["protocol"] != PROTOCOL_VERSION
        or checkpoint["config"]["method"] != config["method"]
    ):
        raise ValueError("checkpoint model/protocol mismatch")
    if checkpoint["stage"] != stage or checkpoint["statistics"] != stats:
        raise ValueError("checkpoint stage/normalizer mismatch")


def prepare(dataset: Dataset, config: dict, directory: Path, seed: int):
    directory.mkdir(parents=True, exist_ok=False)
    stats = compute_statistics(dataset, config.get("noise_std", 0.02))
    write_json(directory / "normalization.json", stats)
    write_json(directory / "data_identity.json", dataset.identity())
    if config["method"] in {"mgn", "eagle"}:
        graph_directory = directory / "graphs"
        graph_directory.mkdir()
        for index in dataset.splits["train"] + dataset.splits["validation"]:
            sample = dataset.static(index)
            np.savez_compressed(
                graph_directory / f"trajectory_{index:04d}.npz",
                edges=sample["edges"],
                edge_features=sample["edge_features"],
            )
    if config["method"] == "eagle":
        prepare_clusters(dataset, directory / "clusters", seed)
    write_json(
        directory / "ready.json",
        {
            "protocol": PROTOCOL_VERSION,
            "method": config["method"],
            "seed": seed,
            "data_identity": dataset.identity(),
        },
    )
    return stats


def open_prepared(dataset: Dataset, config: dict, directory: Path):
    ready = json.loads((directory / "ready.json").read_text())
    if ready["method"] != config["method"]:
        raise ValueError("prepared data belongs to another method")
    stats = load_statistics(directory / "normalization.json", dataset)
    if config["method"] in {"mgn", "eagle"}:
        dataset.graph_cache = directory / "graphs"
    return stats


@torch.no_grad()
def prepare_latents(dataset, config, prepared, checkpoint_file, device):
    stats = open_prepared(dataset, config, prepared)
    checkpoint = load_checkpoint(checkpoint_file)
    validate_dependency(checkpoint, config, stats, "ae")
    model = make_model(config, "ae", device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    destination = prepared / "train_latents.h5"
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(".partial.h5")
    with h5py.File(temporary, "x") as handle:
        identity = {
            "protocol": PROTOCOL_VERSION,
            "ae_checkpoint_id": checkpoint["checkpoint_id"],
            "data_identity": dataset.identity(),
            "statistics": stats,
            "representation": "posterior_mean",
            "frames": 65,
        }
        handle.attrs["identity"] = json.dumps(identity, sort_keys=True)
        for index in dataset.splits["train"]:
            sample = dataset.read(index)
            means = []
            for field in sample["field"]:
                values, coords = aroma_inputs(field, sample["points"], stats, device)
                mean, _ = model.encode(values, coords)
                means.append(mean[0].cpu().numpy())
            handle.create_dataset(
                f"trajectory_{index:04d}", data=np.stack(means), compression="gzip"
            )
    os.replace(temporary, destination)
    return identity


def epoch_groups(dataset, method, stage, seed, epoch, effective_batch):
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    indices = dataset.splits["train"]
    if method == "mgn":
        order = generator.permutation(len(indices) * 64)
        units = [(indices[int(number) // 64], int(number) % 64) for number in order]
        return [
            units[start : start + effective_batch]
            for start in range(0, len(units), effective_batch)
        ]
    order = [indices[int(number)] for number in generator.permutation(len(indices))]
    if method == "aroma" and stage == "dynamics":
        return [
            [(index, step) for index in order[start : start + effective_batch]]
            for start in range(0, len(order), effective_batch)
            for step in range(64)
        ]
    maximum = 60 if method == "eagle" else 65
    units = [(index, int(generator.integers(maximum))) for index in order]
    return [
        units[start : start + effective_batch]
        for start in range(0, len(units), effective_batch)
    ]


def load_ae(config, stats, checkpoint_file, device, expected_id=None):
    checkpoint = load_checkpoint(checkpoint_file)
    validate_dependency(checkpoint, config, stats, "ae")
    if expected_id is not None and checkpoint["checkpoint_id"] != expected_id:
        raise ValueError("the AE differs from the dynamics/cache dependency")
    model = make_model(config, "ae", device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, checkpoint["checkpoint_id"]


@torch.no_grad()
def evaluate_model(
    model,
    config,
    stage,
    dataset,
    stats,
    prepared,
    indices,
    device,
    training_seed,
    output_dir=None,
    autoencoder=None,
):
    rows = []
    model.eval()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "predictions").mkdir()
    for index in indices:
        if stage == "ae":
            target = dataset.read(index)
            errors = []
            for frame in (0, 21, 43, 64):
                values, coords = aroma_inputs(
                    target["field"][frame], target["points"], stats, device
                )
                mean, _ = model.encode(values, coords)
                prediction = model.decode(mean, coords)[0]
                prediction = (
                    (
                        prediction * tensor(stats["std"][:3], device)
                        + tensor(stats["mean"][:3], device)
                    )
                    .cpu()
                    .numpy()
                )
                weights = node_area_weights(target["points"], target["cells"])
                numerator = np.sum(
                    weights[:, None]
                    * (prediction[:, :2] - target["field"][frame, :, :2]) ** 2
                )
                denominator = np.sum(
                    weights[:, None] * target["field"][frame, :, :2] ** 2
                )
                errors.append(
                    float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))
                )
            finite = bool(np.isfinite(errors).all())
            rows.append(
                {
                    "trajectory_index": index,
                    "seed": 0,
                    "finite": finite,
                    "uv_relative_rmse": float(np.mean(errors)) if finite else None,
                }
            )
            if output_dir is not None:
                append_json(output_dir / "case_metrics.jsonl", rows[-1])
            continue
        seeds = (0, 1, 2) if config["method"] == "aroma" else (0,)
        for seed in seeds:
            derived = sample_seed(training_seed, index, seed)
            seed_everything(derived)
            begin = time.perf_counter()
            initial = dataset.initial(index)
            input_io = time.perf_counter() - begin
            # Ground-truth future loading occurs only after the prediction call.
            failure_reason = None
            diagnostics = {}
            try:
                prediction, pre_boundary, timings = rollout(
                    model,
                    config,
                    initial,
                    stats,
                    prepared,
                    device,
                    autoencoder,
                    diagnostics,
                )
            except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
                failure_reason = f"{type(error).__name__}: {error}"
                prediction = np.full(
                    (65, len(initial["points"]), 3), np.nan, dtype=np.float32
                )
                prediction[0] = initial["initial"]
                pre_boundary = prediction.copy()
                timings = {}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as error:
                if output_dir is not None:
                    append_json(
                        output_dir / "case_metrics.jsonl",
                        {
                            "trajectory_index": index,
                            "seed": seed,
                            "finite": False,
                            "failure_reason": f"{type(error).__name__}: {error}",
                            "evaluation_aborted": True,
                        },
                    )
                raise
            begin = time.perf_counter()
            target = dataset.read(index)["field"]
            input_io += time.perf_counter() - begin
            begin = time.perf_counter()
            metrics = compute_metrics(
                prediction,
                target,
                initial["points"],
                initial["cells"],
                initial["node_type"],
                DT,
            )
            primary = metrics.get("uv_relative_rmse")
            metrics["finite"] = bool(
                metrics.get("finite", False)
                and primary is not None
                and np.isfinite(primary)
            )
            metrics["metrics_seconds"] = time.perf_counter() - begin
            mask = np.isin(initial["node_type"], [4, 6])
            if np.isfinite(pre_boundary).all() and mask.any():
                metrics["boundary_uv_rmse_pre_writeback"] = float(
                    np.sqrt(
                        np.mean(
                            (
                                pre_boundary[1:, mask, :2]
                                - initial["initial"][None, mask, :2]
                            )
                            ** 2
                        )
                    )
                )
            row = {
                "trajectory_index": index,
                "seed": seed,
                "sample_seed": derived,
                "input_io_seconds": input_io,
                **metrics,
                **timings,
                **peak_memory(device),
            }
            if not row["finite"]:
                row["failure_reason"] = (
                    failure_reason or "nonfinite prediction or primary metric"
                )
            if output_dir is not None:
                begin = time.perf_counter()
                prediction_file = (
                    output_dir
                    / "predictions"
                    / f"trajectory_{index:04d}_seed_{seed}.npz"
                )
                np.savez_compressed(
                    prediction_file,
                    prediction=prediction,
                    pre_boundary=pre_boundary,
                    target=target,
                    points=initial["points"],
                    cells=initial["cells"],
                    node_type=initial["node_type"],
                    raw_frame_indices=np.arange(0, 513, 8),
                    physical_time=np.arange(65) * DT,
                    trajectory_index=index,
                    seed=seed,
                    **diagnostics,
                )
                row["prediction_file"] = str(prediction_file.relative_to(output_dir))
                row["output_io_seconds"] = time.perf_counter() - begin
                row["io_seconds"] = input_io + row["output_io_seconds"]
                append_json(output_dir / "case_metrics.jsonl", row)
            rows.append(row)
    summary = summarize_trajectories(rows)
    if output_dir is not None:
        write_csv(output_dir / "case_metrics.csv", rows)
        frame_rows = [
            {
                "trajectory_index": row["trajectory_index"],
                "seed": row["seed"],
                "stored_frame": frame,
                "raw_frame": frame * 8,
                "time": frame * DT,
                "uv_relative_rmse": value,
            }
            for row in rows
            for frame, value in enumerate(row.get("per_frame_uv_relative_rmse", []))
        ]
        write_csv(output_dir / "frame_metrics.csv", frame_rows)
        write_csv(output_dir / "trajectory_metrics.csv", summary["trajectory_metrics"])
        write_json(output_dir / "summary.json", summary)
        write_json(
            output_dir / "failures.json",
            {"failures": [row for row in rows if not row["finite"]]},
        )
    return summary


def train(
    dataset,
    config,
    prepared,
    output_dir,
    device,
    seed,
    precision,
    microbatch,
    accumulation,
    stage,
    resume=None,
    ae_checkpoint=None,
    max_updates=None,
):
    stats = open_prepared(dataset, config, prepared)
    options = config["training"][stage]
    if min(microbatch, accumulation) < 1:
        raise ValueError("microbatch and accumulation must be positive")
    effective_batch = microbatch * accumulation
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = make_model(config, stage, device)
    optimizer_class = (
        torch.optim.AdamW if config["method"] == "aroma" else torch.optim.Adam
    )
    optimizer = optimizer_class(
        model.parameters(), lr=options["learning_rate"], weight_decay=0
    )
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: 0.9999991**update)
        if config["method"] == "mgn"
        else torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, options["epochs"], eta_min=options["min_lr"]
        )
        if config["method"] == "aroma"
        else None
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    epoch = cursor = updates = examples = 0
    elapsed_prior = 0.0
    best = None
    autoencoder = None
    ae_id = None
    latent_cache = None
    if stage == "dynamics":
        if not ae_checkpoint:
            raise ValueError("dynamics requires --ae-checkpoint")
        autoencoder, ae_id = load_ae(config, stats, ae_checkpoint, device)
        latent_cache = h5py.File(prepared / "train_latents.h5", "r")
        identity = json.loads(latent_cache.attrs["identity"])
        if (
            identity["ae_checkpoint_id"] != ae_id
            or identity["statistics"] != stats
            or identity["frames"] != 65
        ):
            latent_cache.close()
            raise ValueError("latent cache does not match AE/normalizer/prefix")
    settings = {
        "seed": seed,
        "precision": precision,
        "microbatch": microbatch,
        "accumulation": accumulation,
        "effective_batch": effective_batch,
        "stage": stage,
    }
    if resume is None:
        output_dir.mkdir(parents=True, exist_ok=False)
        run_id = str(uuid.uuid4())
        write_json(
            output_dir / "run_manifest.json",
            {
                "run_id": run_id,
                "code": code_identity(),
                "config": config,
                "data": dataset.identity(),
                "settings": settings,
                "ae_checkpoint_id": ae_id,
                "device": str(device),
                "supplied_dataset_bytes": dataset.dataset.stat().st_size,
                "example_unit": "trajectory-window"
                if config["method"] == "eagle"
                else "trajectory-frame"
                if stage == "ae"
                else "adjacent-transition",
            },
        )
    else:
        checkpoint = load_checkpoint(resume)
        validate_dependency(checkpoint, config, stats, stage)
        if (
            checkpoint["config"] != config
            or checkpoint["settings"] != settings
            or checkpoint["ae_checkpoint_id"] != ae_id
        ):
            raise ValueError("resume config/settings/dependency mismatch")
        manifest = json.loads((output_dir / "run_manifest.json").read_text())
        if manifest["run_id"] != checkpoint["run_id"]:
            raise ValueError("resume directory does not belong to this checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        epoch, cursor, updates = (
            checkpoint["epoch"],
            checkpoint["cursor"],
            checkpoint["updates"],
        )
        examples, best, elapsed_prior = (
            checkpoint["examples_seen"],
            checkpoint["best"],
            checkpoint["elapsed_seconds"],
        )
        run_id = checkpoint["run_id"]
        restore_rng(checkpoint["rng"])
        append_json(
            output_dir / "events.jsonl",
            {
                "event": "resume",
                "update": updates,
                "checkpoint_id": checkpoint["checkpoint_id"],
            },
        )
    from .runtime import acquire_run_lock

    run_lock = acquire_run_lock(output_dir)
    source_identity = code_identity()
    started = time.perf_counter()
    metrics_file = output_dir / "metrics.jsonl"

    def snapshot():
        return {
            "protocol": PROTOCOL_VERSION,
            "code": source_identity,
            "run_id": run_id,
            "checkpoint_id": str(uuid.uuid4()),
            "stage": stage,
            "config": config,
            "settings": settings,
            "statistics": stats,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "rng": rng_state(),
            "epoch": epoch,
            "cursor": cursor,
            "updates": updates,
            "examples_seen": examples,
            "best": best,
            "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
            "ae_checkpoint_id": ae_id,
        }

    try:
        while epoch < options["epochs"] and (
            max_updates is None or updates < max_updates
        ):
            groups = epoch_groups(
                dataset, config["method"], stage, seed, epoch, effective_batch
            )
            if cursor > len(groups):
                raise ValueError("invalid resume cursor")
            while cursor < len(groups) and (
                max_updates is None or updates < max_updates
            ):
                group = groups[cursor]
                group_nodes = sum(
                    len(dataset.static(index)["points"]) for index, _ in group
                )
                optimizer.zero_grad(set_to_none=True)
                model.train()
                losses = {}
                for start in range(0, len(group), microbatch):
                    with autocast(device, precision):
                        loss, parts = training_loss(
                            model,
                            config,
                            stage,
                            group[start : start + microbatch],
                            dataset,
                            stats,
                            prepared,
                            device,
                            group_nodes,
                            len(group),
                            latent_cache,
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite training loss")
                    scaler.scale(loss).backward()
                    for name, value in parts.items():
                        losses[name] = losses.get(name, 0.0) + value
                scaler.unscale_(optimizer)
                gradient = torch.sqrt(
                    sum(
                        torch.sum(parameter.grad.float() ** 2)
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    )
                )
                if not torch.isfinite(gradient):
                    raise FloatingPointError("nonfinite training gradient")
                scaler.step(optimizer)
                scaler.update()
                updates += 1
                examples += len(group)
                cursor += 1
                if config["method"] == "mgn":
                    scheduler.step()
                if updates == 1 or updates % config["log_every_updates"] == 0:
                    synchronize(device)
                    row = {
                        "event": "train",
                        "stage": stage,
                        "update": updates,
                        "epoch": epoch,
                        "examples_seen": examples,
                        "loss": sum(losses.values()),
                        **losses,
                        "gradient_norm": float(gradient),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": elapsed_prior
                        + time.perf_counter()
                        - started,
                        **peak_memory(device),
                    }
                    append_json(metrics_file, row)
                    print(json.dumps(row), flush=True)
                if updates % config["checkpoint_every_updates"] == 0:
                    save_checkpoint(output_dir / "last.pt", snapshot())
            if cursor == len(groups):
                epoch += 1
                cursor = 0
                if scheduler is not None and config["method"] == "aroma":
                    scheduler.step()
                if (
                    epoch % options["validate_every_epochs"] == 0
                    or epoch == options["epochs"]
                ):
                    preserved_rng = rng_state()
                    summary = evaluate_model(
                        model,
                        config,
                        stage,
                        dataset,
                        stats,
                        prepared,
                        monitor_indices(dataset.splits["validation"]),
                        device,
                        seed,
                        autoencoder=autoencoder,
                    )
                    restore_rng(preserved_rng)
                    key = selection_key(summary, updates)
                    event = {
                        "event": "validation",
                        "stage": stage,
                        "update": updates,
                        "epoch": epoch,
                        "examples_seen": examples,
                        "failed_clips": summary["failed_clips"],
                        "uv_relative_rmse": summary["selection_uv_relative_rmse"],
                    }
                    append_json(metrics_file, event)
                    if best is None or key < tuple(best["key"]):
                        best = {
                            "key": list(key),
                            "epoch": epoch,
                            "update": updates,
                            "summary": summary,
                        }
                        payload = snapshot()
                        save_checkpoint(output_dir / "best.pt", payload)
                        write_json(
                            output_dir / "selector.json",
                            {"checkpoint_id": payload["checkpoint_id"], **best},
                        )
                    save_checkpoint(output_dir / f"epoch_{epoch:06d}.pt", snapshot())
                save_checkpoint(output_dir / "last.pt", snapshot())
        save_checkpoint(output_dir / "last.pt", snapshot())
        state = (
            "training_complete"
            if epoch == options["epochs"]
            else "stopped_at_requested_update_limit"
        )
        result = {
            "state": state,
            "updates": updates,
            "epoch": epoch,
            "cursor": cursor,
            "examples_seen": examples,
            "best": best,
            "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
            **peak_memory(device),
        }
        write_json(output_dir / "summary.json", result)
        return result
    finally:
        if latent_cache is not None:
            latent_cache.close()
        run_lock.close()


def evaluate(
    dataset,
    config,
    prepared,
    checkpoint_files,
    output_dir,
    device,
    mode,
    ae_checkpoint=None,
):
    if mode == "validation" and len(checkpoint_files) != 1:
        raise ValueError("final Validation evaluates one already-selected checkpoint")
    stats = open_prepared(dataset, config, prepared)
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = []
    for number, checkpoint_file in enumerate(checkpoint_files):
        checkpoint = load_checkpoint(checkpoint_file)
        stage = checkpoint["stage"]
        if stage == "ae" and mode == "validation":
            raise ValueError(
                "AE reconstruction supports --mode select; final forecasting requires dynamics"
            )
        validate_dependency(checkpoint, config, stats, stage)
        model = make_model(checkpoint["config"], stage, device)
        model.load_state_dict(checkpoint["model"], strict=True)
        ae = None
        if stage == "dynamics":
            ae, _ = load_ae(
                config, stats, ae_checkpoint, device, checkpoint["ae_checkpoint_id"]
            )
        indices = (
            monitor_indices(dataset.splits["validation"])
            if mode == "select"
            else dataset.splits["validation"]
        )
        candidate_dir = output_dir / f"candidate_{number:03d}"
        summary = evaluate_model(
            model,
            config,
            stage,
            dataset,
            stats,
            prepared,
            indices,
            device,
            checkpoint["settings"]["seed"],
            candidate_dir,
            ae,
        )
        record = {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_update": checkpoint["updates"],
            "checkpoint": str(Path(checkpoint_file).resolve()),
            "stage": stage,
            "key": selection_key(summary, checkpoint["updates"]),
            "summary": summary,
            "result_directory": candidate_dir.name,
        }
        candidates.append(record)
    selected = min(candidates, key=lambda row: row["key"])
    result = {
        "protocol": PROTOCOL_VERSION,
        "evaluator": EVALUATOR_VERSION,
        "mode": mode,
        "code": code_identity(),
        "data_identity": dataset.identity(),
        "selected": selected,
        "candidates": candidates,
        "state": "validation_complete_awaiting_test"
        if mode == "validation"
        else "selection_complete",
    }
    write_json(output_dir / "evaluation.json", result)
    return result
