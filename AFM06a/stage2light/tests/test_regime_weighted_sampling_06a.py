from __future__ import annotations

import unittest

import numpy as np

from AFM06a.stage1pluslight.config import default_config
from AFM06a.stage1pluslight.data import (
    dataset_paths,
    load_dataset,
    prepare_window,
    transition_resolved_source_indices,
)


class AFM06aSamplingTests(unittest.TestCase):
    def test_transition_resolved_grid_preserves_regular_and_switch_samples(self) -> None:
        times = 0.1 * np.arange(12, dtype=float)
        contact = np.asarray(
            [False, False, False, False, True, True, True, True, False, False, False, False]
        )
        selected = transition_resolved_source_indices(
            times,
            contact,
            start=0,
            stride=4,
            transition_half_width_s=0.11,
        )

        np.testing.assert_array_equal(selected, np.asarray([0, 3, 4, 7, 8, 11]))

    def test_default_grid_is_regime_weighted_with_fixed_count(self) -> None:
        config = default_config()
        paths = dataset_paths(config.dataset_root, config.error_level)
        missing = [path for name, path in paths.items() if name != "data_dir" and not path.is_file()]
        if missing:
            self.skipTest("generated AFM06a dataset is intentionally not bundled")
        dataset = load_dataset(config.dataset_root, config.error_level)
        window = prepare_window(config, dataset)
        times_all = np.asarray(dataset.table["t"], dtype=float)

        self.assertEqual(config.sampling_policy, "regime_weighted_fixed_count")
        self.assertEqual(config.transition_sampling_weight, 10.0)
        self.assertEqual(config.contact_sampling_weight, 5.0)
        self.assertEqual(config.noncontact_sampling_weight, 1.0)
        self.assertEqual(window.times.size, 394)
        self.assertEqual(window.train_idx.size, 315)
        self.assertEqual(window.val_idx.size, 79)
        self.assertAlmostEqual(float(window.times[0]), 19.474944e-3, places=15)
        self.assertAlmostEqual(float(window.times[-1]), 20.525056e-3, places=15)
        self.assertTrue(np.all(np.diff(window.times) > 0.0))
        self.assertEqual(np.unique(window.source_idx).size, window.source_idx.size)
        self.assertGreater(int(np.max(np.diff(window.source_idx))), int(config.sample_stride))
        self.assertLess(int(np.min(np.diff(window.source_idx))), int(config.sample_stride))

    def test_window_rhs_residual_uses_dataset_manifest_settings(self) -> None:
        config = default_config()
        paths = dataset_paths(config.dataset_root, config.error_level)
        missing = [path for name, path in paths.items() if name != "data_dir" and not path.is_file()]
        if missing:
            self.skipTest("generated AFM06a dataset is intentionally not bundled")
        dataset = load_dataset(config.dataset_root, config.error_level)
        window = prepare_window(config, dataset)

        np.testing.assert_allclose(
            window.bar_fts_rhs_residual,
            window.bar_fts,
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg="AFM06a RHS residual must use the dataset manifest physics settings",
        )


if __name__ == "__main__":
    unittest.main()
