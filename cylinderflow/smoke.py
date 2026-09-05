"""Real-model bounded training/resume and complete autoregressive CPU checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import Dataset
from .engine import evaluate, load_ae, open_prepared, prepare, prepare_latents, train
from .fixture import make_fixture
from .models import make_model, rollout, training_loss
from .runtime import load_checkpoint, peak_memory, seed_everything, write_json


def smoke(config, output_dir: Path, device):
    output_dir.mkdir(parents=True, exist_ok=False)
    fixture, manifest = make_fixture(output_dir / "fixture")
    dataset = Dataset(fixture, manifest, debug=True)
    config = copy.deepcopy(config)
    for options in config["training"].values():
        options["epochs"] = 2
        options["validate_every_epochs"] = 1
    config["checkpoint_every_updates"] = 1000
    prepared = output_dir / "prepared"
    prepare(dataset, config, prepared, 123)
    stages = ("ae", "dynamics") if config["method"] == "aroma" else ("main",)
    ae_checkpoint = None
    checks = {}
    for stage in stages:
        # One complete tiny-fixture epoch exercises all native temporal pairs
        # and the epoch scheduler/Validation boundary as well as mid-epoch resume.
        update_cap = 148 if config["method"] == "mgn" or stage == "dynamics" else 2
        run = output_dir / stage
        train(
            dataset,
            config,
            prepared,
            run,
            device,
            123,
            "fp32",
            1,
            1,
            stage,
            ae_checkpoint=ae_checkpoint,
            max_updates=1,
        )
        checkpoint_file = run / "last.pt"
        train(
            dataset,
            config,
            prepared,
            run,
            device,
            123,
            "fp32",
            1,
            1,
            stage,
            resume=checkpoint_file,
            ae_checkpoint=ae_checkpoint,
            max_updates=update_cap,
        )
        reference_run = output_dir / f"{stage}_uninterrupted"
        train(
            dataset,
            config,
            prepared,
            reference_run,
            device,
            123,
            "fp32",
            1,
            1,
            stage,
            ae_checkpoint=ae_checkpoint,
            max_updates=update_cap,
        )
        resumed = load_checkpoint(checkpoint_file)
        reference = load_checkpoint(reference_run / "last.pt")
        for name, value in resumed["model"].items():
            torch.testing.assert_close(value, reference["model"][name], rtol=0, atol=0)
        for key in ("cursor", "epoch", "updates", "examples_seen", "scheduler"):
            assert resumed[key] == reference[key], key
        checks[f"{stage}_exact_resume"] = True
        if stage == "ae":
            ae_checkpoint = checkpoint_file
            prepare_latents(dataset, config, prepared, ae_checkpoint, device)
            with h5py.File(prepared / "train_latents.h5") as handle:
                assert all(value.shape == (75, 32, 8) for value in handle.values())
            checks["posterior_mean_cache_train75"] = True
    final_file = output_dir / stages[-1] / "last.pt"
    selection = evaluate(
        dataset,
        config,
        prepared,
        [final_file],
        output_dir / "selection",
        device,
        "select",
        ae_checkpoint,
    )
    assert (
        selection["selected"]["checkpoint_id"]
        == load_checkpoint(final_file)["checkpoint_id"]
    )
    checks["fixed_validation_selection"] = True
    result = evaluate(
        dataset,
        config,
        prepared,
        [final_file],
        output_dir / "evaluation",
        device,
        "validation",
        ae_checkpoint,
    )
    assert result["selected"]["summary"]["failed_clips"] == 0
    checks["full_64_step_rollout"] = True
    stats = open_prepared(dataset, config, prepared)
    model = make_model(config, stages[-1], device)
    model.load_state_dict(load_checkpoint(final_file)["model"])
    ae = load_ae(config, stats, ae_checkpoint, device)[0] if ae_checkpoint else None
    seed_everything(901)
    first = rollout(model, config, dataset.initial(2), stats, prepared, device, ae)[0]
    with h5py.File(fixture, "r+") as handle:
        field = handle["trajectory_0002/uvp"]
        field[1:] = field[1:] + 500
    seed_everything(901)
    second = rollout(model, config, dataset.initial(2), stats, prepared, device, ae)[0]
    np.testing.assert_array_equal(first, second)
    checks["future_truth_independence"] = True
    from .media import plot_curves, render

    prediction_file = next(
        (output_dir / "evaluation" / "candidate_000" / "predictions").glob("*.npz")
    )
    if config["method"] == "eagle":
        with np.load(prediction_file) as bundle:
            assert bundle["eagle_state_uvpq"].shape == (
                *bundle["prediction"].shape[:2],
                4,
            )
            np.testing.assert_array_equal(
                bundle["prediction"], bundle["eagle_state_uvpq"][..., :3]
            )
        checks["native_four_channel_artifact"] = True
    render([prediction_file], output_dir / "media", [config["method"]], every=16)
    plot_curves([output_dir / stages[-1]], output_dir / "curves")
    checks["png_gif_mp4_curves"] = True
    result = {
        "method": config["method"],
        "device": str(device),
        "checks": checks,
        "quality_claim": "synthetic correctness acceptance only; no trained-model quality claim",
        "gpu_maximum_graph_acceptance": "pending",
    }
    write_json(output_dir / "smoke.json", result)
    return result


def largest_graph(config, dataset, prepared, output_dir, device, args):
    """Explicit target-cluster acceptance; one update, no long training launch."""
    output_dir.mkdir(parents=True, exist_ok=False)
    stage = args.stage or ("ae" if config["method"] == "aroma" else "main")
    stats = open_prepared(dataset, config, prepared)
    index = max(
        dataset.splits["train"], key=lambda item: len(dataset.static(item)["points"])
    )
    seed_everything(args.seed)
    model = make_model(config, stage, device)
    ae = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cache = None
    if stage == "dynamics":
        ae, _ = load_ae(config, stats, args.ae_checkpoint, device)
        cache = h5py.File(prepared / "train_latents.h5")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"][stage]["learning_rate"]
    )
    n = len(dataset.static(index)["points"])
    from .runtime import autocast

    try:
        with autocast(device, args.precision):
            loss, parts = training_loss(
                model,
                config,
                stage,
                [(index, 0)],
                dataset,
                stats,
                prepared,
                device,
                n,
                1,
                cache,
            )
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite largest-graph loss")
        result = {
            "trajectory_index": index,
            "nodes": n,
            "loss": float(loss),
            "parts": parts,
            "stage": stage,
            "device": str(device),
            "precision": args.precision,
            "microbatch": 1,
            "data_identity": dataset.identity(),
        }
        if stage != "ae":
            prediction, _, timings = rollout(
                model, config, dataset.initial(index), stats, prepared, device, ae
            )
            result.update(
                {
                    "prediction_shape": list(prediction.shape),
                    "finite": bool(np.isfinite(prediction).all()),
                    **timings,
                }
            )
            if not result["finite"]:
                raise FloatingPointError("nonfinite largest-graph rollout")
        result.update(peak_memory(device))
        write_json(output_dir / "preflight.json", result)
        return result
    finally:
        if cache is not None:
            cache.close()
