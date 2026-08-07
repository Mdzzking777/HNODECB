from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import torch

from AFM06a.stage1pluslight.config import default_config
from AFM06a.stage1pluslight.kan_backend import (
    GLOBAL_NORMALIZED_INPUT_POLICY,
    IDENTITY_FORCE_OUTPUT_POLICY,
    build_kan_runtime,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class AFM06aSingleInputKANTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = np.asarray(
            [
                [-9.0e-9, -2.0e-4],
                [-8.0e-9, 1.0e-4],
                [-7.0e-9, -1.0e-4],
                [-6.0e-9, 2.0e-4],
            ],
            dtype=float,
        )
        config = replace(
            default_config(_repo_root()),
            normalizer_policy=GLOBAL_NORMALIZED_INPUT_POLICY,
            force_output_policy=IDENTITY_FORCE_OUTPUT_POLICY,
            adaptive_grid_enabled=False,
        )
        self.global_mean = np.asarray([-7.5e-9, 0.0], dtype=float)
        self.global_scale = np.asarray([1.25e-9, 1.0], dtype=float)
        self.model = build_kan_runtime(
            config,
            nn_init_seed=20260726,
            training_states=self.states,
            gain_reference=np.zeros(self.states.shape[0], dtype=float),
            global_state_mean=self.global_mean,
            global_state_scale=self.global_scale,
        )

    def test_architecture_is_single_input_1_3_1(self) -> None:
        self.assertEqual(self.model.construction["width"], [1, 3, 1])
        self.assertEqual(self.model.construction["routing_policy"], "single_x1_branch")
        self.assertEqual(tuple(self.model.kan.act_fun[0].mask.shape), (1, 3))

    def test_only_x1_enters_the_kan(self) -> None:
        states = torch.as_tensor(self.states, dtype=torch.float64)
        expected = (states[:, [0]] - self.global_mean[0]) / self.global_scale[0]
        torch.testing.assert_close(self.model.kan_inputs(states), expected)
        changed_x2 = states.clone()
        changed_x2[:, 1] = torch.tensor([9.0, -8.0, 7.0, -6.0], dtype=torch.float64)
        torch.testing.assert_close(
            self.model(changed_x2),
            self.model(states),
            rtol=0.0,
            atol=0.0,
        )

    def test_all_six_edges_are_trainable(self) -> None:
        states = torch.as_tensor(self.states, dtype=torch.float64)
        self.model.zero_grad(set_to_none=True)
        torch.sum(self.model(states)).backward()
        for layer in self.model.kan.act_fun:
            self.assertIsNotNone(layer.coef.grad)
            self.assertTrue(torch.all(torch.isfinite(layer.coef.grad)))


if __name__ == "__main__":
    unittest.main()
