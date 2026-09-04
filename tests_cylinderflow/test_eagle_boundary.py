import json
import tempfile
import unittest
from pathlib import Path

import torch

from cylinderflow.data import Dataset
from cylinderflow.engine import prepare
from cylinderflow.fixture import make_fixture
from cylinderflow.models import eagle_inputs, make_model
from cylinderflow.runtime import ROOT, seed_everything


CONFIG = json.loads((ROOT / "cylinderflow_config.json").read_text())


@unittest.skipUnless(
    CONFIG["method"] == "eagle", "EAGLE-only native four-channel boundary path"
)
class EagleBoundaryTests(unittest.TestCase):
    def test_native_future_state_cannot_change_any_predicted_channel(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_file, manifest = make_fixture(root / "data")
            dataset = Dataset(data_file, manifest, True)
            stats = prepare(dataset, CONFIG, root / "prepared", 123)
            device = torch.device("cpu")
            seed_everything(123)
            model = make_model(CONFIG, "main", device).eval()
            args = list(
                eagle_inputs(dataset.read(2, 0, 6), stats, root / "prepared", device)
            )
            fixed_boundary = args[2][:, 0].clone()
            with torch.no_grad():
                expected = model(
                    *args, boundary_values=fixed_boundary, boundary_channels=(0, 1)
                )[0]
                args[2] = args[2].clone()
                args[2][:, 1:, :, :] += 1234
                changed = model(
                    *args, boundary_values=fixed_boundary, boundary_channels=(0, 1)
                )[0]
            torch.testing.assert_close(expected, changed, rtol=0, atol=0)
            self.assertEqual(expected.shape[-1], 4)
            self.assertFalse(
                torch.equal(expected[:, 1:, :-1, 2:], args[2][:, 1:, :-1, 2:])
            )


if __name__ == "__main__":
    unittest.main()
