import copy
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from cylinderflow.data import Dataset, compute_statistics
from cylinderflow.engine import epoch_groups
from cylinderflow.fixture import make_fixture
from cylinderflow.metrics import (
    compute_metrics,
    node_area_weights,
    selection_key,
    summarize_trajectories,
    triangle_vorticity_divergence,
)
from cylinderflow.runtime import monitor_indices


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        data_file, manifest = make_fixture(Path(self.temporary.name) / "fixture")
        self.dataset = Dataset(data_file, manifest, debug=True)

    def tearDown(self):
        self.temporary.cleanup()

    def test_prefix_only_statistics_samples_and_targets(self):
        before = compute_statistics(self.dataset)
        samples = {i: self.dataset.read(i)["field"] for i in (0, 1, 2)}
        with h5py.File(self.dataset.dataset, "r+") as handle:
            for group in handle.values():
                group["uvp"][65:] = np.nan
        self.assertEqual(before, compute_statistics(self.dataset))
        for i, sample in samples.items():
            np.testing.assert_array_equal(sample, self.dataset.read(i)["field"])
        with self.assertRaises(ValueError):
            self.dataset.read(0, 0, 66)
        self.assertNotEqual(samples[0].shape[1], samples[1].shape[1])

    def test_split_and_time_contract(self):
        with self.assertRaises(ValueError):
            Dataset(self.dataset.dataset, self.dataset.manifest_file)
        original = json.loads(self.dataset.manifest_file.read_text())
        for key, value in [
            ("frame_dt", 0.64),
            ("temporal_stride", 64),
            ("splits", {"train": [0], "validation": [0]}),
            ("splits", {"train": [0], "validation": [2], "test": [1]}),
        ]:
            changed = dict(original)
            changed[key] = value
            self.dataset.manifest_file.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                Dataset(self.dataset.dataset, self.dataset.manifest_file, debug=True)

    def test_native_sampling_exposure(self):
        for method, stage in [("mgn", "main"), ("aroma", "dynamics")]:
            groups = epoch_groups(self.dataset, method, stage, 123, 0, 2)
            self.assertEqual(
                sorted(unit for group in groups for unit in group),
                [(i, t) for i in (0, 1) for t in range(64)],
            )
        for method, stage, maximum in [("aroma", "ae", 65), ("eagle", "main", 60)]:
            for epoch in range(20):
                units = [
                    unit
                    for group in epoch_groups(
                        self.dataset, method, stage, 123, epoch, 2
                    )
                    for unit in group
                ]
                self.assertEqual(sorted(i for i, _ in units), [0, 1])
                self.assertTrue(all(0 <= t < maximum for _, t in units))
        selected = monitor_indices(tuple(range(1000, 1100)))
        self.assertEqual(len(set(selected)), 24)
        self.assertEqual((selected[0], selected[-1]), (1000, 1099))

    def test_metric_gauge_area_derivatives_and_failure(self):
        sample = self.dataset.read(2)
        target = sample["field"].astype(np.float64)
        prediction = target.copy()
        prediction[1:, :, 2] += 7
        metrics = compute_metrics(
            prediction,
            target,
            sample["points"],
            sample["cells"],
            sample["node_type"],
            0.08,
        )
        self.assertAlmostEqual(metrics["pressure_raw_rmse"], 7)
        self.assertAlmostEqual(metrics["pressure_gauge_free_rmse"], 0)
        weights = node_area_weights(sample["points"], sample["cells"])
        self.assertGreater(np.std(weights), 0)
        prediction = target.copy()
        prediction[1:, 0, 0] += 1
        metrics = compute_metrics(
            prediction,
            target,
            sample["points"],
            sample["cells"],
            sample["node_type"],
            0.08,
        )
        expected = np.sqrt(
            64 * weights[0] / np.sum(weights[None, :, None] * target[1:, :, :2] ** 2)
        )
        self.assertAlmostEqual(metrics["uv_relative_rmse"], expected)
        xy = sample["points"].astype(np.float64)
        velocity = np.stack(
            [2 * xy[:, 0] + 3 * xy[:, 1], 5 * xy[:, 0] - 2 * xy[:, 1]], axis=-1
        )
        omega, div, _ = triangle_vorticity_divergence(
            velocity[None], xy, sample["cells"]
        )
        np.testing.assert_allclose(omega, 2, atol=1e-6)
        np.testing.assert_allclose(div, 0, atol=1e-6)
        prediction[3, 0, 0] = np.nan
        self.assertFalse(
            compute_metrics(
                prediction, target, xy, sample["cells"], sample["node_type"], 0.08
            )["finite"]
        )

    def test_seed_aggregation_and_selection(self):
        rows = [
            {"trajectory_index": i, "seed": s, "finite": True, "uv_relative_rmse": v}
            for i, s, v in [(0, 0, 1), (0, 1, 3), (0, 2, 5), (1, 0, 9)]
        ]
        summary = summarize_trajectories(rows)
        self.assertEqual(summary["selection_uv_relative_rmse"], 6)
        failed = copy.deepcopy(rows)
        failed[0]["finite"] = False
        summary_failed = summarize_trajectories(failed)
        self.assertEqual(summary_failed["failed_clips"], 1)
        self.assertEqual(summary_failed["failed_trajectories"], 1)
        self.assertEqual(summary_failed["trajectory_count"], 2)
        self.assertLess(selection_key(summary, 100), selection_key(summary_failed, 1))
        self.assertLess(selection_key(summary, 1), selection_key(summary, 2))


if __name__ == "__main__":
    unittest.main()
