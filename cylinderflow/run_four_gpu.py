"""Run the native four-GPU recipe inside one explicit GPU allocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("dataset", "manifest", "prepared", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "cylinderflow_config_4gpu.json"
    )
    parser.add_argument(
        "--gpus", help="exactly four allocated GPU IDs/UUIDs; omit inside Slurm"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--stage", choices=("main", "ae", "dynamics"))
    parser.add_argument("--ae-checkpoint", type=Path)
    args = parser.parse_args()
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
    }
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    assigned = env.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (
        len(assigned) != 4
        or any(not item.strip() for item in assigned)
        or len(set(assigned)) != 4
    ):
        raise ValueError("supply exactly four explicitly allocated visible GPU IDs")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError("choose a new output directory or use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from .runtime import acquire_run_lock, write_json

    lock = acquire_run_lock(args.output_dir)
    try:
        config = json.loads(args.config.read_text())
        if config.get("distributed", {}).get("world_size") != 4:
            raise ValueError("this launcher requires a four-rank configuration")
        if args.preflight:
            config["preflight_max_graph"] = True
        config_file = args.output_dir.resolve() / "config.json"
        if config_file.exists() and json.loads(config_file.read_text()) != config:
            raise ValueError("launcher resume config mismatch")
        write_json(config_file, config)
        logs = args.output_dir.resolve() / "launcher_logs"
        logs.mkdir(exist_ok=True)

        def run(label, command):
            attempt = len(list(logs.glob(label + "_*.log"))) + 1
            log = logs / f"{label}_{attempt:03d}.log"
            begin = time.time()
            with log.open("x", encoding="utf-8") as output:
                rc = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT
                ).returncode
            write_json(
                log.with_suffix(".json"),
                {
                    "command": command,
                    "exit_code": rc,
                    "started_unix": begin,
                    "elapsed_seconds": time.time() - begin,
                },
            )
            if rc:
                write_json(
                    args.output_dir / "status.json",
                    {
                        "state": "failed",
                        "stage": label,
                        "exit_code": rc,
                        "log": str(log),
                    },
                )
                raise RuntimeError(f"{label} exited {rc}; see {log}")

        prepared = args.prepared.resolve()
        if config["method"] == "aroma":
            local_prepared = args.output_dir.resolve() / "prepared"
            local_prepared.mkdir(exist_ok=True)
            for name in ("ready.json", "normalization.json", "data_identity.json"):
                destination = local_prepared / name
                if not destination.exists():
                    shutil.copyfile(prepared / name, destination)
                elif destination.read_bytes() != (prepared / name).read_bytes():
                    raise ValueError("prepared Train statistics changed")
            if args.stage == "dynamics":
                if args.ae_checkpoint is None:
                    raise ValueError(
                        "a separate dynamics stage requires --ae-checkpoint"
                    )
                source_cache = prepared / "train_latents.h5"
                target_cache = local_prepared / "train_latents.h5"
                if not target_cache.exists():
                    if not source_cache.is_file():
                        raise FileNotFoundError(
                            "prepare the selected AE's Train latent cache first"
                        )
                    temporary = (
                        local_prepared / f"train_latents.{uuid.uuid4().hex}.partial.h5"
                    )
                    try:
                        os.link(source_cache, temporary)
                    except OSError:
                        shutil.copyfile(source_cache, temporary)
                    os.replace(temporary, target_cache)
            prepared = local_prepared
        stages = (
            [args.stage]
            if args.stage
            else (["ae", "dynamics"] if config["method"] == "aroma" else ["main"])
        )
        if args.preflight and not args.stage:
            stages = stages[:1]
        ae = args.ae_checkpoint.resolve() if args.ae_checkpoint else None
        common = [
            "--config",
            str(config_file),
            "--dataset",
            str(args.dataset.resolve()),
            "--manifest",
            str(args.manifest.resolve()),
            "--prepared",
            str(prepared),
            "--device",
            "cuda:0",
        ]
        for stage in stages:
            folder = args.output_dir.resolve() / stage
            checkpoint = folder / "last.pt"
            summary_file = folder / "summary.json"
            complete = (
                summary_file.exists()
                and json.loads(summary_file.read_text()).get("state")
                == "training_complete"
            )
            if not complete:
                training = (
                    ["resume", "--checkpoint", str(checkpoint)]
                    if args.resume and checkpoint.exists()
                    else ["train"]
                )
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=4",
                    "--max-restarts=0",
                    "--module",
                    "cylinderflow",
                    *training,
                    *common,
                    "--stage",
                    stage,
                    "--output-dir",
                    str(folder),
                ]
                if ae is not None:
                    command += ["--ae-checkpoint", str(ae)]
                if args.preflight:
                    command += ["--max-updates", "8"]
                run(stage, command)
            if stage == "ae" and not args.preflight:
                ae = folder / "best.pt"
                if not ae.is_file():
                    raise RuntimeError("AE training produced no selected checkpoint")
                if not (prepared / "train_latents.h5").exists():
                    cache_run = args.output_dir.resolve() / "cache_build"
                    if cache_run.exists():
                        cache_run = (
                            args.output_dir.resolve()
                            / f"cache_build_{int(time.time())}"
                        )
                    run(
                        "cache",
                        [
                            sys.executable,
                            "-m",
                            "cylinderflow",
                            "prepare",
                            "--latents",
                            *common,
                            "--ae-checkpoint",
                            str(ae),
                            "--output-dir",
                            str(cache_run),
                        ],
                    )
        write_json(
            args.output_dir / "status.json",
            {
                "state": "preflight_complete"
                if args.preflight
                else "training_complete",
                "stages": stages,
                "world_size": 4,
                "test_accessed": False,
            },
        )
    finally:
        lock.close()


if __name__ == "__main__":
    main()
