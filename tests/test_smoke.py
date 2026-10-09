"""Small, data-free checks for the PB-MMAE package."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised before installation
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ModelSmokeTest(unittest.TestCase):
    def test_model_reconstructs_all_families(self) -> None:
        from pbmmae.model import FAMILY_NAMES, PBMMAELinear64, split_into_families

        model = PBMMAELinear64().eval()
        batch = torch.from_numpy(np.zeros((2, 14, 13, 13), dtype=np.float32))
        with torch.no_grad():
            reconstructions, fused, _, _ = model(split_into_families(batch))
        self.assertEqual(tuple(fused.shape), (2, 64))
        self.assertEqual(tuple(reconstructions), FAMILY_NAMES)
        self.assertEqual(tuple(reconstructions["magnetic"].shape), (2, 1, 13, 13))
        self.assertEqual(tuple(reconstructions["structural"].shape), (2, 4, 13, 13))


class ConfigurationAndPipelineSmokeTest(unittest.TestCase):
    def test_frozen_configuration_and_dry_run(self) -> None:
        from pbmmae.cli import main
        from pbmmae.config import load_config
        from pbmmae.runner import run_pipeline

        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "paper1.example.yaml")
        self.assertEqual(config.train.evaluation_folds, (1, 2, 4, 5))
        self.assertEqual(config.train.development_fold, 3)
        self.assertEqual(len(run_pipeline(config, dry_run=True)), 14)
        self.assertEqual(main(["run", "--config", str(root / "configs" / "paper1.example.yaml"), "--dry-run"]), 0)
