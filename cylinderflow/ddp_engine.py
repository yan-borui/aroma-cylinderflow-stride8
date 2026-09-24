"""Four-rank native baseline training; global batching and rank-local exact recovery."""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import time
import traceback
import uuid
from pathlib import Path

import h5py
import torch
from torch.nn.parallel import DistributedDataParallel

from . import PROTOCOL_VERSION
from .data import TRAIN_FRAMES
from .distributed import Context, capture_rng, restore_rng
from .engine import (
    epoch_groups,
    evaluate_model,
    load_ae,
    open_prepared,
    validate_dependency,
)
from .metrics import selection_key, summarize_trajectories
from .models import make_model, training_loss
from .runtime import (
    ROOT,
    acquire_run_lock,
    append_json,
    autocast,
    clean_json,
    code_identity,
    load_checkpoint,
    monitor_indices,
    peak_memory,
    read_jsonl,
    save_checkpoint,
    seed_everything,
    synchronize,
    write_csv,
    write_json,
)

FORMAT = "cylinderflow.native_ddp.training.v1"


class NativeLoss(torch.nn.Module):
    def __init__(
        self, model, config, stage, dataset, stats, prepared, device, latent_cache
    ):
        super().__init__()
        self.model = model
        self.arguments = (config, stage, dataset, stats, prepared, device, latent_cache)

    def forward(self, units, nodes, samples):
        if not units:
            return sum(p.sum() * 0 for p in self.model.parameters()), {}
        config, stage, dataset, stats, prepared, device, cache = self.arguments
        return training_loss(
            self.model,
            config,
            stage,
            units,
            dataset,
            stats,
            prepared,
            device,
            nodes,
            samples,
            cache,
        )


def freeze_source(output: Path, resume: bool) -> None:
    for directory in ("cylinderflow", "modulus", "Models", "aroma"):
        for original in sorted((ROOT / directory).rglob("*.py")):
            destination = output / "source" / original.relative_to(ROOT)
            if resume:
                if (
                    not destination.is_file()
                    or destination.read_bytes() != original.read_bytes()
                ):
                    raise ValueError(
                        f"resume source changed: {original.relative_to(ROOT)}"
                    )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, destination)


def train_distributed(
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
    ctx = Context(config["distributed"]["world_size"], str(device))
    device, lock, cache, attempt_dir = ctx.device, None, None, None
    epoch = cursor = updates = examples = 0
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if min(microbatch, accumulation) < 1:
            raise ValueError(
                "positive per-rank microbatch and accumulation are required"
            )
        effective_batch = ctx.world * microbatch * accumulation
        options = config["training"][stage]
        expected = options.get("effective_batch")
        if expected is not None and expected != effective_batch:
            raise ValueError("configured effective batch differs from rank allocation")
        stats = open_prepared(dataset, config, prepared)
        settings = {
            "seed": seed,
            "precision": precision,
            "microbatch": microbatch,
            "accumulation": accumulation,
            "effective_batch": effective_batch,
            "world_size": ctx.world,
            "stage": stage,
        }

        def prepare_run():
            nonlocal lock
            if resume is None:
                output_dir.mkdir(parents=True, exist_ok=False)
            lock = acquire_run_lock(output_dir)
            freeze_source(output_dir, resume is not None)
            attempt = (
                output_dir
                / f"attempt_{len(list(output_dir.glob('attempt_*'))) + 1:03d}"
            )
            attempt.mkdir()
            return str(attempt)

        attempt_dir = Path(ctx.primary_call(prepare_run))
        seed_everything(seed)
        model = make_model(config, stage, device)
        optimizer_class = (
            torch.optim.AdamW if config["method"] == "aroma" else torch.optim.Adam
        )
        optimizer = optimizer_class(
            model.parameters(), lr=options["learning_rate"], weight_decay=0
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, options["epochs"], eta_min=options["min_lr"]
            )
            if config["method"] == "aroma"
            else None
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and precision == "fp16"
        )
        ae, ae_id = None, None
        if stage == "dynamics":
            if ae_checkpoint is None:
                raise ValueError("dynamics requires the frozen selected AE")
            ae, ae_id = load_ae(config, stats, ae_checkpoint, device)
            cache = h5py.File(prepared / "train_latents.h5", "r")
            cache_id = json.loads(cache.attrs["identity"])
            if (
                cache_id["ae_checkpoint_id"] != ae_id
                or cache_id["statistics"] != stats
                or cache_id["frames"] != TRAIN_FRAMES
            ):
                raise ValueError("latent cache/AE/Train75 dependency mismatch")
        wrapper = NativeLoss(
            model, config, stage, dataset, stats, prepared, device, cache
        )
        initial_state = (
            {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            if config.get("preflight_max_graph")
            else None
        )
        largest = max(
            dataset.splits["train"], key=lambda i: len(dataset.static(i)["points"])
        )
        ddp = DistributedDataParallel(
            wrapper,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        seed_everything(seed + ctx.rank * 1000003)
        run_id = ctx.broadcast(str(uuid.uuid4()) if ctx.primary else None)
        best, pending_evaluation = None, False
        elapsed_prior = 0.0
        costs = {"train_seconds": 0.0, "validation_seconds": 0.0}
        if resume is not None:
            saved = load_checkpoint(resume)
            validate_dependency(saved, config, stats, stage)
            if (
                saved.get("format") != FORMAT
                or saved["config"] != config
                or saved["settings"] != settings
                or saved["ae_checkpoint_id"] != ae_id
            ):
                raise ValueError("resume training config/world/dependency mismatch")
            manifest = json.loads((output_dir / "run_manifest.json").read_text())
            if (
                manifest["run_id"] != saved["run_id"]
                or manifest["data"] != dataset.identity()
            ):
                raise ValueError("resume run/data identity mismatch")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            if scheduler is not None:
                scheduler.load_state_dict(saved["scheduler"])
            epoch, cursor, updates, examples = (
                saved["epoch"],
                saved["cursor"],
                saved["updates"],
                saved["examples_seen"],
            )
            best, run_id, elapsed_prior = (
                saved["best"],
                saved["run_id"],
                saved["elapsed_seconds"],
            )
            pending_evaluation = saved.get("pending_evaluation", False)
            costs.update(saved.get("costs", {}))
            restore_rng(saved["rank_states"][ctx.rank]["rng"], device)
        else:
            ctx.primary_call(
                lambda: write_json(
                    output_dir / "run_manifest.json",
                    {
                        "run_id": run_id,
                        "code": code_identity(),
                        "config": config,
                        "data": dataset.identity(),
                        "settings": settings,
                        "ae_checkpoint_id": ae_id,
                        "mgn_lr_clock": "global_examples_seen"
                        if config["method"] == "mgn"
                        else None,
                    },
                )
            )
        started = time.perf_counter()

        def checkpoint(destination):
            states = ctx.gather(
                {"rng": capture_rng(device), "memory": peak_memory(device)}
            )

            def save():
                payload = {
                    "format": FORMAT,
                    "protocol": PROTOCOL_VERSION,
                    "code": code_identity(),
                    "run_id": run_id,
                    "checkpoint_id": f"{run_id}:{updates}",
                    "stage": stage,
                    "config": config,
                    "settings": settings,
                    "statistics": stats,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    "scaler": scaler.state_dict(),
                    "rank_states": states,
                    "epoch": epoch,
                    "cursor": cursor,
                    "updates": updates,
                    "examples_seen": examples,
                    "best": best,
                    "pending_evaluation": pending_evaluation,
                    "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
                    "costs": costs,
                    "ae_checkpoint_id": ae_id,
                }
                save_checkpoint(destination, payload)

            ctx.primary_call(save)

        def validation():
            nonlocal best, pending_evaluation
            preserved, mode = capture_rng(device), model.training
            begin = time.perf_counter()
            output = output_dir / "monitor" / f"update_{updates:09d}"
            indices = monitor_indices(dataset.splits["validation"])
            try:

                def evaluate_shard():
                    rows = []
                    for index in indices[ctx.rank :: ctx.world]:
                        case_dir = output / f"trajectory_{index:04d}"
                        completed = case_dir / "completed.json"
                        if completed.exists():
                            record = json.loads(completed.read_text())
                            if record["checkpoint_id"] != f"{run_id}:{updates}":
                                raise ValueError(
                                    "physical evaluation checkpoint identity mismatch"
                                )
                            rows.extend(record["rows"])
                            continue
                        case_dir.mkdir(parents=True, exist_ok=True)
                        attempt = (
                            case_dir
                            / f"attempt_{len(list(case_dir.glob('attempt_*'))) + 1:03d}"
                        )
                        evaluate_model(
                            model,
                            config,
                            stage,
                            dataset,
                            stats,
                            prepared,
                            (index,),
                            device,
                            seed,
                            output_dir=attempt,
                            autoencoder=ae,
                            provenance={
                                "checkpoint_id": f"{run_id}:{updates}",
                                "update": updates,
                            },
                            fail_on_runtime_error=True,
                        )
                        case_rows = read_jsonl(attempt / "case_metrics.jsonl")
                        for row in case_rows:
                            if "prediction_file" in row:
                                row["prediction_file"] = str(
                                    (attempt / row["prediction_file"]).relative_to(
                                        output
                                    )
                                )
                        write_json(
                            completed,
                            {"checkpoint_id": f"{run_id}:{updates}", "rows": case_rows},
                        )
                        rows.extend(case_rows)
                    return rows

                shards = ctx.all_call(evaluate_shard)

                def merge():
                    rows = sorted(
                        [r for shard in shards for r in shard],
                        key=lambda row: (row["trajectory_index"], row["seed"]),
                    )
                    count = len(indices) * (
                        3 if stage != "ae" and config["method"] == "aroma" else 1
                    )
                    identities = {(r["trajectory_index"], r["seed"]) for r in rows}
                    if len(rows) != count or len(identities) != count:
                        raise ValueError(
                            "missing or duplicated baseline Validation samples"
                        )
                    summary = summarize_trajectories(rows)
                    write_json(output / "summary.json", summary)
                    write_json(output / "cases.json", {"rows": rows})
                    write_csv(output / "case_metrics.csv", rows)
                    write_csv(
                        output / "trajectory_metrics.csv", summary["trajectory_metrics"]
                    )
                    write_json(
                        output / "failures.json",
                        {"failures": [r for r in rows if not r["finite"]]},
                    )
                    event = {
                        "event": "validation",
                        "stage": stage,
                        "update": updates,
                        "epoch": epoch,
                        "examples_seen": examples,
                        "failed_clips": summary["failed_clips"],
                        "uv_relative_rmse": summary["selection_uv_relative_rmse"],
                    }
                    write_json(output / "event.json", event)
                    print(json.dumps(clean_json(event)), flush=True)
                    return summary

                summary = ctx.primary_call(merge)
                key = selection_key(summary, updates)
                improved = best is None or key < tuple(best["key"])
                if improved:
                    best = {
                        "key": list(key),
                        "epoch": epoch,
                        "update": updates,
                        "summary": summary,
                    }
                pending_evaluation = False
            finally:
                model.train(mode)
                restore_rng(preserved, device)
                costs["validation_seconds"] += time.perf_counter() - begin
            if improved:
                checkpoint(output_dir / "best.pt")
                ctx.primary_call(
                    lambda: write_json(
                        output_dir / "selector.json",
                        {
                            "checkpoint_id": f"{run_id}:{updates}",
                            **best,
                        },
                    )
                )
            checkpoint(output_dir / f"epoch_{epoch:06d}.pt")
            checkpoint(output_dir / "last.pt")

        if resume is None:
            checkpoint(output_dir / "last.pt")
        if pending_evaluation:
            validation()
        while epoch < options["epochs"] and (
            max_updates is None or updates < max_updates
        ):
            groups = epoch_groups(
                dataset,
                config["method"],
                stage,
                seed,
                epoch,
                effective_batch,
                config.get("dynamics_train_frames", TRAIN_FRAMES),
            )
            if cursor > len(groups):
                raise ValueError("invalid global epoch-group cursor")
            while cursor < len(groups) and (
                max_updates is None or updates < max_updates
            ):
                group = groups[cursor]
                if initial_state is not None:
                    group = [(largest, step) for _, step in group]
                nodes = sum(len(dataset.static(index)["points"]) for index, _ in group)
                synchronize(device)
                begin = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                model.train()
                if config["method"] == "mgn":
                    # Same LR at a given exposure count as the original B1 recipe.
                    for item in optimizer.param_groups:
                        item["lr"] = options["learning_rate"] * 0.9999991**examples
                used_lr = optimizer.param_groups[0]["lr"]
                local_units = group[ctx.rank :: ctx.world]
                passes = math.ceil(len(group) / (ctx.world * microbatch))
                loss_sum = torch.zeros((), device=device)
                for micro in range(passes):
                    units = local_units[micro * microbatch : (micro + 1) * microbatch]
                    synchronize_gradients = micro == passes - 1
                    with (
                        contextlib.nullcontext()
                        if synchronize_gradients
                        else ddp.no_sync()
                    ):
                        with autocast(device, precision):
                            loss, _ = ddp(units, nodes, len(group))
                        if not all(ctx.gather(bool(torch.isfinite(loss)))):
                            raise FloatingPointError("nonfinite native training loss")
                        # Native loss is already divided by global sample/node count.
                        scaler.scale(loss * ctx.world).backward()
                    loss_sum += loss.detach()
                scaler.unscale_(optimizer)
                norm = torch.sqrt(
                    sum(
                        torch.sum(p.grad.float() ** 2)
                        for p in model.parameters()
                        if p.grad is not None
                    )
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite native training gradient")
                scaler.step(optimizer)
                scaler.update()
                updates += 1
                examples += len(group)
                cursor += 1
                synchronize(device)
                costs["train_seconds"] += time.perf_counter() - begin
                if updates == 1 or updates % config["log_every_updates"] == 0:
                    global_loss = float(ctx.mean(loss_sum)) * ctx.world
                    rank_memory = ctx.gather({"rank": ctx.rank, **peak_memory(device)})
                    row = {
                        "event": "train",
                        "stage": stage,
                        "update": updates,
                        "epoch": epoch,
                        "examples_seen": examples,
                        "loss": global_loss,
                        "gradient_norm": float(norm),
                        "learning_rate": used_lr,
                        "effective_batch": effective_batch,
                        "elapsed_seconds": elapsed_prior
                        + time.perf_counter()
                        - started,
                        "rank_metrics": rank_memory,
                        **costs,
                        **peak_memory(device),
                    }

                    def log():
                        append_json(output_dir / "metrics.jsonl", row)
                        write_json(
                            output_dir / "status.json", {**row, "state": "running"}
                        )
                        print(json.dumps(clean_json(row)), flush=True)

                    ctx.primary_call(log)
                if updates % config["checkpoint_every_updates"] == 0:
                    checkpoint(output_dir / "last.pt")
            if cursor == len(groups):
                epoch += 1
                cursor = 0
                if scheduler is not None:
                    scheduler.step()
                pending_evaluation = (
                    epoch % options["validate_every_epochs"] == 0
                    or epoch == options["epochs"]
                )
                checkpoint(output_dir / "last.pt")
                if pending_evaluation:
                    validation()
        if initial_state is not None:

            def preflight():
                changes = [
                    float((value.detach().cpu() - initial_state[name]).abs().max())
                    for name, value in model.state_dict().items()
                    if value.is_floating_point()
                ]
                if not all(math.isfinite(v) for v in changes) or max(changes) <= 0:
                    raise RuntimeError(
                        "preflight did not update model parameters finitely"
                    )
                rng, mode = capture_rng(device), model.training
                try:
                    summary = evaluate_model(
                        model,
                        config,
                        stage,
                        dataset,
                        stats,
                        prepared,
                        (largest,),
                        device,
                        seed,
                        output_dir=attempt_dir / f"rank_{ctx.rank:03d}_maximum_graph",
                        autoencoder=ae,
                        fail_on_runtime_error=True,
                    )
                    if summary["failed_clips"]:
                        raise FloatingPointError(
                            "preflight physical output is nonfinite"
                        )
                finally:
                    model.train(mode)
                    restore_rng(rng, device)
                return {
                    "rank": ctx.rank,
                    "largest_train_graph": largest,
                    "maximum_parameter_change": max(changes),
                    "physical_summary": summary,
                    **peak_memory(device),
                }

            checked = ctx.all_call(preflight)
            ctx.primary_call(
                lambda: write_json(output_dir / "preflight.json", {"ranks": checked})
            )
        checkpoint(output_dir / "last.pt")

        def finish():
            elapsed = elapsed_prior + time.perf_counter() - started
            result = {
                "state": "training_complete"
                if epoch == options["epochs"]
                else "stopped_at_requested_update_limit",
                "epoch": epoch,
                "cursor": cursor,
                "updates": updates,
                "examples_seen": examples,
                "best": best,
                "elapsed_seconds": elapsed,
                "settings": settings,
                **costs,
                "allocated_gpu_hours": ctx.world * elapsed / 3600
                if device.type == "cuda"
                else None,
                "training_gpu_hours": ctx.world * costs["train_seconds"] / 3600
                if device.type == "cuda"
                else None,
                **peak_memory(device),
            }
            write_json(output_dir / "summary.json", result)
            write_json(output_dir / "status.json", result)
            write_json(
                output_dir / "checkpoint_inventory.json",
                {
                    "last_checkpoint": "last.pt",
                    "best_physical_checkpoint": "best.pt" if best else None,
                    "best_update": best["update"] if best else None,
                    "termination_reason": result["state"],
                },
            )
            return result

        return ctx.primary_call(finish)
    except BaseException as error:
        if attempt_dir is not None:
            write_json(
                attempt_dir / f"rank_{ctx.rank:03d}_failure.json",
                {
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "update": updates,
                },
            )
            if ctx.primary:
                write_json(
                    output_dir / "status.json",
                    {"state": "failed", "update": updates, "error": str(error)},
                )
        raise
    finally:
        if cache is not None:
            cache.close()
        if lock is not None:
            lock.close()
        ctx.close()
