from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from AFM06a.stage1pluslight.config import default_config
from AFM06a.stage1pluslight.kan_backend import (
    LOCAL_FORCE_OUTPUT_POLICY,
    NORMALIZED_INPUT_POLICY,
    build_kan_runtime,
    normalized_support_from_raw_inputs,
    training_state_normalizer,
)
from AFM06a.stage2light.config import default_config as stage2_default_config
from AFM06a.stage2light.kan_backend import restore_exact_stage1_model


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class AFM06aKANInputPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = np.asarray(
            [
                [-2.0e-9, -3.0e-4],
                [1.0e-9, 2.0e-4],
                [4.0e-9, -1.0e-4],
                [2.0e-9, 1.0e-4],
            ],
            dtype=float,
        )
        self.global_mean = np.asarray([0.5e-9, -0.25e-4], dtype=float)
        self.global_scale = np.asarray([2.0e-9, 2.5e-4], dtype=float)

    def _build(self):
        return build_kan_runtime(
            replace(default_config(_repo_root()), adaptive_grid_enabled=False),
            nn_init_seed=20260719,
            training_states=self.states,
            gain_reference=np.zeros(self.states.shape[0], dtype=float),
            global_state_mean=self.global_mean,
            global_state_scale=self.global_scale,
        )

    def test_formal_defaults_match_the_archived_single_input_baseline(self) -> None:
        config = default_config(_repo_root())
        self.assertEqual(config.kan_width, (1, 3, 1))
        self.assertEqual(config.nn_seed_bank_size, 2000)
        self.assertEqual(config.normalizer_policy, NORMALIZED_INPUT_POLICY)
        self.assertEqual(config.force_output_policy, LOCAL_FORCE_OUTPUT_POLICY)
        self.assertEqual(
            config.loss_policy,
            "force_scaled_dynamics_residual_true_x1_residual_smooth",
        )
        self.assertEqual(config.smoothness_loss_weight, 1.0)
        self.assertFalse(config.gain_enabled)
        self.assertFalse(config.soft_mask_enabled)
        self.assertEqual(config.readiness_gaps(), [])

    def test_training_window_x1_transform_and_support_are_exact(self) -> None:
        model = self._build()
        states = torch.as_tensor(self.states, dtype=torch.float64)
        mean, scale = training_state_normalizer(self.states)
        expected = (states[:, [0]] - mean[0]) / scale[0]
        torch.testing.assert_close(model.kan_inputs(states), expected)
        np.testing.assert_allclose(
            model.initial_grid_support.detach().cpu().numpy(),
            normalized_support_from_raw_inputs(
                self.states,
                mean,
                scale,
            ),
        )
        self.assertEqual(model.metadata()["construction"]["width"], [1, 3, 1])

    def test_stage2_restores_the_exact_stage1_function(self) -> None:
        source = self._build()
        metadata = source.metadata()
        endpoint = SimpleNamespace(
            warmstart={
                "kan_construction": metadata["construction"],
                "state_mean": metadata["state_mean"],
                "state_scale": metadata["state_scale"],
                "force_output_policy": metadata["force_output_policy"],
                "force_mean": metadata["force_mean"],
                "force_scale": metadata["force_scale"],
                "initial_grid_support": metadata["initial_grid_support"],
                "kan_state_dict": source.frozen_state_dict(),
                "kan_state_dict_sha256": metadata["state_dict_sha256"],
            }
        )
        restored = restore_exact_stage1_model(stage2_default_config(_repo_root()), endpoint)
        states = torch.as_tensor(self.states, dtype=torch.float64)
        torch.testing.assert_close(restored(states), source(states), rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
