"""CPU/Gloo checks of native DDP batching and recovery using a controlled model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from . import PROTOCOL_VERSION
from . import ddp_engine as engine
from .metrics import summarize_trajectories
from .runtime import append_json, load_checkpoint, write_json


class Fixture:
    splits = {"train": tuple(range(6)), "validation": (6,)}

    def static(self, index):
        return {"points": np.zeros((3 + index, 2))}

    def identity(self):
        return {"synthetic_ddp_control": True, "test_accessed": False}


def model_factory(*args):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.25)
    return model


def controlled_loss(
    model,
    config,
    stage,
    units,
    dataset,
    stats,
    prepared,
    device,
    nodes,
    samples,
    cache=None,
):
    loss = torch.zeros((), device=device)
    for index, _ in units:
        x = (torch.arange(1, 4 + index, device=device).float() / (3 + index))[:, None]
        target = 0.1 + 0.05 * torch.randn_like(x)
        loss = loss + (model(x) - target).square().sum() / nodes
    return loss, {"node_mse": float(loss.detach())}


def controlled_evaluation(
    model,
    config,
    stage,
    dataset,
    stats,
    prepared,
    indices,
    device,
    seed,
    output_dir=None,
    **kwargs,
):
    model.eval()
    torch.randn(
        73
    )  # Deliberately consume the training PRNG unless evaluation isolates it.
    rows = [
        {"trajectory_index": index, "seed": 0, "finite": True, "uv_relative_rmse": 0.5}
        for index in indices
    ]
    output_dir.mkdir(parents=True)
    for row in rows:
        append_json(output_dir / "case_metrics.jsonl", row)
    return summarize_trajectories(rows)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def worker(rank, directory):
    root = Path(directory)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").resolve().as_uri(),
        rank=rank,
        world_size=2,
    )
    try:
        engine.make_model = model_factory
        engine.open_prepared = lambda *args: {}
        engine.training_loss = controlled_loss
        engine.evaluate_model = controlled_evaluation
        config = {
            "protocol": PROTOCOL_VERSION,
            "method": "eagle",
            "distributed": {"world_size": 2},
            "log_every_updates": 1,
            "checkpoint_every_updates": 1,
            "training": {
                "main": {
                    "epochs": 2,
                    "validate_every_epochs": 1,
                    "learning_rate": 1e-4,
                    "effective_batch": 4,
                }
            },
        }
        dataset = Fixture()
        common = (
            dataset,
            config,
            root,
            None,
            torch.device("cpu"),
            123,
            "fp32",
            1,
            2,
            "main",
        )

        def train(folder, cap, resume=False):
            values = list(common)
            values[3] = root / folder
            return engine.train_distributed(
                *values,
                resume=root / folder / "last.pt" if resume else None,
                max_updates=cap,
            )

        train("continuous", 4)
        train("resumed", 1)
        if rank == 0:
            first = load_checkpoint(root / "resumed/last.pt")
            group = engine.epoch_groups(dataset, "eagle", "main", 123, 0, 4)[0]
            reference = model_factory()
            nodes = sum(3 + index for index, _ in group)
            xs, targets = [], []
            for peer in range(2):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(123 + peer * 1000003)
                    for index, _ in group[peer::2]:
                        x = (torch.arange(1, 4 + index).float() / (3 + index))[:, None]
                        xs.append(x)
                        targets.append(0.1 + 0.05 * torch.randn_like(x))
            expected_loss = (
                (reference(torch.cat(xs)) - torch.cat(targets)).square().mean()
            )
            expected_loss.backward()
            saved_moment = next(iter(first["optimizer"]["state"].values()))["exp_avg"]
            torch.testing.assert_close(
                saved_moment, reference.weight.grad * 0.1, rtol=1e-5, atol=1e-8
            )
        dist.barrier()
        original = engine.evaluate_model

        def fault(*args, **kwargs):
            result = original(*args, **kwargs)
            raise RuntimeError("injected native evaluator failure")

        engine.evaluate_model = fault
        failed = False
        try:
            train("resumed", 2, True)
        except RuntimeError as error:
            assert "injected native evaluator failure" in str(error)
            failed = True
        finally:
            engine.evaluate_model = original
        assert failed
        train("resumed", 4, True)
        if rank == 0:
            first, second = [
                load_checkpoint(root / folder / "last.pt")
                for folder in ("continuous", "resumed")
            ]
            for key in (
                "model",
                "optimizer",
                "rank_states",
                "epoch",
                "cursor",
                "examples_seen",
                "scheduler",
            ):
                equal(first[key], second[key])
            assert first["examples_seen"] == 12
            write_json(
                root / "acceptance.json",
                {
                    "native_engine_global_node_weighted_gradient": True,
                    "gradient_accumulation_and_partial_batch": True,
                    "continuous_resume_exact": True,
                    "pending_epoch_evaluation_recovered": True,
                    "evaluation_rng_isolation": True,
                    "backend": "cpu/gloo",
                    "world_size": 2,
                    "model": "controlled linear fixture",
                    "native_backbones_verified": False,
                    "formal_gpu_verified": False,
                },
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    mp.spawn(worker, args=(str(args.output_dir.resolve()),), nprocs=2)


if __name__ == "__main__":
    main()
