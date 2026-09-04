"""Small analytic variable-node meshes for CPU acceptance, never research data."""

from pathlib import Path

import h5py
import numpy as np

from .data import FORMAT
from .runtime import write_json


def make_fixture(directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    data_file = directory / "fixture.h5"
    with h5py.File(data_file, "w") as handle:
        for index, nx in enumerate((4, 5, 4)):
            xs = np.linspace(0, 1.6, nx, dtype=np.float32) ** 1.2
            points = np.array(
                [(x, y) for y in (0, 0.3, 0.7) for x in xs], dtype=np.float32
            )
            cells = []
            for y in range(2):
                for x in range(nx - 1):
                    a = y * nx + x
                    cells.extend([[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]])
            cells = np.asarray(cells, dtype=np.int64)
            types = np.zeros(len(points), dtype=np.int64)
            types[:nx] = 6
            types[::nx] = 4
            types[nx - 1 :: nx] = 5
            t = np.arange(75, dtype=np.float32)[:, None] * 0.08
            x, y = points[:, 0][None], points[:, 1][None]
            u = 1 + 0.15 * np.sin(2 * np.pi * t) + 0.2 * y + np.zeros_like(x)
            v = 0.1 * np.cos(2 * np.pi * t) - 0.15 * x + np.zeros_like(y)
            pressure = 0.5 * x - 0.2 * y + 0.1 * np.sin(t)
            field = np.stack([u, v, pressure], axis=-1).astype(np.float32)
            fixed = np.isin(types, [4, 6])
            field[:, fixed, :2] = field[0, fixed, :2]
            group = handle.create_group(f"trajectory_{index:04d}")
            group.create_dataset("uvp", data=field)
            group.create_dataset("mesh_pos", data=points)
            group.create_dataset("cells", data=cells)
            group.create_dataset("node_type", data=types)
    manifest_file = directory / "manifest.json"
    write_json(
        manifest_file,
        {
            "format": FORMAT,
            "frames": 75,
            "temporal_stride": 8,
            "frame_dt": 0.08,
            "raw_frame_dt": 0.01,
            "test_accessed": False,
            "splits": {"train": [0, 1], "validation": [2]},
        },
    )
    return data_file, manifest_file
