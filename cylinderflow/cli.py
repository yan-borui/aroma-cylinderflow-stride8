"""Offline, single-device entry points for the matched CylinderFlow task."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import torch

from .data import Dataset
from .engine import config_from_file, evaluate, prepare, prepare_latents, train
from .runtime import ROOT, clean_json, code_identity, write_json


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "resume", "evaluate", "doctor", "smoke"):
        child = commands.add_parser(name)
        child.add_argument(
            "--config", type=Path, default=ROOT / "cylinderflow_config.json"
        )
        child.add_argument("--device", default="cpu")
        child.add_argument("--threads", type=int, default=2)
        child.add_argument(
            "--dataset",
            type=Path,
            required=name in {"prepare", "train", "resume", "evaluate"},
        )
        child.add_argument(
            "--manifest",
            type=Path,
            required=name in {"prepare", "train", "resume", "evaluate"},
        )
        child.add_argument(
            "--debug-data",
            action="store_true",
            help="allow tiny fixtures; recorded as nonformal",
        )
        child.add_argument("--output-dir", type=Path, required=name not in {"doctor"})
        child.add_argument("--seed", type=int, default=123)
        if name in {"train", "resume", "evaluate", "smoke", "prepare"}:
            child.add_argument("--prepared", type=Path)
            child.add_argument("--ae-checkpoint", type=Path)
        if name in {"train", "resume", "smoke"}:
            child.add_argument("--stage", choices=["main", "ae", "dynamics"])
            child.add_argument(
                "--precision", choices=["fp32", "bf16", "fp16"], default="fp32"
            )
            child.add_argument("--microbatch", type=int)
            child.add_argument("--accumulation", type=int)
            child.add_argument(
                "--max-updates",
                type=int,
                help="explicit bounded run; total updates, including resume",
            )
        if name == "prepare":
            child.add_argument(
                "--latents",
                action="store_true",
                help="cache Train posterior means using --ae-checkpoint",
            )
        if name == "resume":
            child.add_argument("--checkpoint", type=Path, required=True)
        if name == "evaluate":
            child.add_argument("--checkpoint", type=Path, nargs="+", required=True)
            child.add_argument(
                "--mode", choices=["select", "validation"], default="select"
            )
        if name == "smoke":
            child.add_argument(
                "--largest-graph",
                action="store_true",
                help="one real training update and full rollout on the largest Train mesh",
            )
    render = commands.add_parser("render")
    render.add_argument("--inputs", type=Path, nargs="+", required=True)
    render.add_argument("--labels", nargs="+")
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--every", type=int, default=1)
    render.add_argument("--fps", type=float, default=12.5)
    render.add_argument(
        "--scales", type=Path, help="reuse the scales.json from another rendering"
    )
    curves = commands.add_parser("plot-curves")
    curves.add_argument("--run", type=Path, nargs="+", required=True)
    curves.add_argument("--output-dir", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--inputs", type=Path, nargs="+", required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    return result


def doctor(config, device, dataset=None):
    from .models import make_model

    packages = ["torch", "numpy", "h5py", "matplotlib", "imageio", "imageio-ffmpeg"]
    packages += {
        "aroma": ["diffusers", "einops", "timm"],
        "mgn": ["dgl"],
        "eagle": ["torch-scatter", "numba"],
    }[config["method"]]
    stages = ["ae", "dynamics"] if config["method"] == "aroma" else ["main"]
    sizes = {}
    for stage in stages:
        model = make_model(config, stage, device)
        sizes[stage] = sum(p.numel() for p in model.parameters())
        del model
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "parameter_counts": sizes,
        "code": code_identity(),
        "wandb": "disabled; never imported",
        "gpu_maximum_graph_acceptance": "pending until smoke --largest-graph is run on target GPU",
    }
    if device.type == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(device)
    if dataset:
        result["data"] = dataset.identity()
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.time()
    destination = args.output_dir
    if destination is not None and destination.exists() and args.command != "resume":
        print(
            f"output directory already exists: {destination}; choose a new directory",
            file=sys.stderr,
        )
        return 2
    try:
        if args.command == "score":
            from .score import score

            result = score(args.inputs, destination)
        elif args.command == "render":
            from .media import render

            result = render(
                args.inputs, destination, args.labels, args.every, args.fps, args.scales
            )
        elif args.command == "plot-curves":
            from .media import plot_curves

            result = plot_curves(args.run, destination)
        else:
            torch.set_num_threads(args.threads)
            device = torch.device(args.device)
            if device.type not in {"cpu", "cuda"}:
                raise ValueError(
                    "supported devices are cpu or an explicitly assigned cuda device"
                )
            if device.type == "cuda":
                torch.cuda.set_device(device)
            config = config_from_file(args.config)
            dataset = (
                Dataset(args.dataset, args.manifest, args.debug_data)
                if args.dataset and args.manifest
                else None
            )
            if args.command == "doctor":
                result = doctor(config, device, dataset)
            elif args.command == "prepare":
                if args.latents:
                    if (
                        not args.prepared
                        or not args.ae_checkpoint
                        or config["method"] != "aroma"
                    ):
                        raise ValueError(
                            "prepare --latents requires AROMA, --prepared and --ae-checkpoint"
                        )
                    destination.mkdir(parents=True, exist_ok=False)
                    result = prepare_latents(
                        dataset, config, args.prepared, args.ae_checkpoint, device
                    )
                else:
                    result = prepare(dataset, config, destination, args.seed)
            elif args.command in {"train", "resume"}:
                if not args.prepared:
                    raise ValueError("--prepared is required")
                stage = args.stage or ("ae" if config["method"] == "aroma" else "main")
                options = config["training"][stage]
                result = train(
                    dataset,
                    config,
                    args.prepared,
                    destination,
                    device,
                    args.seed,
                    args.precision,
                    options["microbatch"]
                    if args.microbatch is None
                    else args.microbatch,
                    options["accumulation"]
                    if args.accumulation is None
                    else args.accumulation,
                    stage,
                    getattr(args, "checkpoint", None),
                    args.ae_checkpoint,
                    args.max_updates,
                )
            elif args.command == "evaluate":
                if not args.prepared:
                    raise ValueError("--prepared is required")
                result = evaluate(
                    dataset,
                    config,
                    args.prepared,
                    args.checkpoint,
                    destination,
                    device,
                    args.mode,
                    args.ae_checkpoint,
                )
            else:
                from .smoke import smoke, largest_graph

                if args.largest_graph:
                    if dataset is None or not args.prepared:
                        raise ValueError(
                            "--largest-graph requires --dataset, --manifest, --prepared"
                        )
                    result = largest_graph(
                        config, dataset, args.prepared, destination, device, args
                    )
                else:
                    result = smoke(config, destination, device)
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            if args.command == "doctor":
                write_json(destination / "doctor.json", result)
            write_json(
                destination / "exit.json",
                {
                    "command": args.command,
                    "exit_code": 0,
                    "started_unix": started,
                    "elapsed_seconds": time.time() - started,
                },
            )
        print(json.dumps(clean_json(result), indent=2), flush=True)
        return 0
    except Exception as error:
        if destination is not None and destination.is_dir():
            from .runtime import append_json

            payload = {
                "command": args.command,
                "exit_code": 1,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "started_unix": started,
                "elapsed_seconds": time.time() - started,
            }
            append_json(destination / "command_failures.jsonl", payload)
            write_json(destination / "exit.json", payload)
        traceback.print_exc()
        return 1
