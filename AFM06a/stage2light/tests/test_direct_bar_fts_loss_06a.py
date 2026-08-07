from __future__ import annotations

import unittest

import torch

from AFM06a.stage1pluslight.losses import (
    SUPPORTED_POLICY,
    dynamics_residual_torch,
    evaluate_dynamics_residual_loss_with_terms,
    residual_smoothness_torch,
)
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (
    AFM06aHardSampleInputs,
)


class AFM06aDynamicsResidualLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AFM06aHardSampleInputs()
        self.times = torch.tensor([0.0100, 0.0101, 0.0102], dtype=torch.float64)
        self.true_states = torch.tensor(
            [[-8.7e-9, -8.8e-9, -8.9e-9], [2.0e-5, -1.0e-5, 3.0e-5]],
            dtype=torch.float64,
        )
        self.true_x2dot = torch.tensor([0.7, -0.9, 1.1], dtype=torch.float64)

    def _exact_fts(self) -> torch.Tensor:
        zero_force = torch.zeros_like(self.true_x2dot)
        residual_without_force = dynamics_residual_torch(
            predicted_fts=zero_force,
            true_states=self.true_states,
            true_x2dot=self.true_x2dot,
            times=self.times,
            settings=self.settings,
        )
        return residual_without_force * float(self.settings.mass_kg)

    def test_policy_names_zero_target_dynamics_residual(self) -> None:
        self.assertEqual(
            SUPPORTED_POLICY,
            "force_scaled_dynamics_residual_true_x1_residual_smooth",
        )

    def test_exact_noiseless_rhs_has_zero_loss(self) -> None:
        exact_force = self._exact_fts()
        total, parts, terms = evaluate_dynamics_residual_loss_with_terms(
            predicted_fts=exact_force,
            true_states=self.true_states,
            true_x2dot=self.true_x2dot,
            times=self.times,
            indices=torch.arange(3),
            residual_scale=torch.tensor(2.0, dtype=torch.float64),
            settings=self.settings,
        )

        self.assertEqual(set(terms), {"dynamics_residual", "smoothness"})
        self.assertAlmostEqual(float(total), 0.0, places=15)
        self.assertAlmostEqual(parts.dynamics_residual, 0.0, places=15)
        self.assertAlmostEqual(parts.smoothness, 0.0, places=15)
        self.assertLess(parts.residual_rmse, 1.0e-12)

    def test_residual_is_normalized_by_dataset_force_scale(self) -> None:
        force_scale = torch.tensor(2.0, dtype=torch.float64)
        predicted_force = self._exact_fts() - force_scale * float(self.settings.mass_kg)
        total, parts, _terms = evaluate_dynamics_residual_loss_with_terms(
            predicted_fts=predicted_force,
            true_states=self.true_states,
            true_x2dot=self.true_x2dot,
            times=self.times,
            indices=torch.arange(3),
            residual_scale=force_scale,
            settings=self.settings,
        )

        self.assertAlmostEqual(float(total), 1.0, places=12)
        self.assertAlmostEqual(parts.dynamics_residual, 1.0, places=12)
        self.assertAlmostEqual(parts.residual_scale, 2.0, places=14)

    def test_residual_backpropagates_only_through_predicted_force(self) -> None:
        predicted_force = torch.zeros(
            3,
            dtype=torch.float64,
            requires_grad=True,
        )
        total, _parts, _terms = evaluate_dynamics_residual_loss_with_terms(
            predicted_fts=predicted_force,
            true_states=self.true_states,
            true_x2dot=self.true_x2dot,
            times=self.times,
            indices=torch.arange(3),
            residual_scale=torch.tensor(2.0, dtype=torch.float64),
            settings=self.settings,
        )
        total.backward()

        self.assertIsNotNone(predicted_force.grad)
        self.assertGreater(float(torch.linalg.vector_norm(predicted_force.grad)), 0.0)

    def test_zero_residual_has_zero_smoothness(self) -> None:
        residual = torch.zeros(6, dtype=torch.float64, requires_grad=True)
        smoothness = residual_smoothness_torch(
            normalized_residual=residual,
            times=torch.tensor(
                [0.0, 0.8, 2.1, 3.0, 4.7, 7.0],
                dtype=torch.float64,
            ),
            indices=torch.arange(6),
            contact_indicator=torch.tensor(
                [False, False, False, True, True, True],
            ),
        )

        self.assertEqual(float(smoothness.detach()), 0.0)
        smoothness.backward()
        torch.testing.assert_close(residual.grad, torch.zeros_like(residual))

    def test_residual_smoothness_does_not_cross_contact_transition(self) -> None:
        smoothness = residual_smoothness_torch(
            normalized_residual=torch.tensor(
                [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
                dtype=torch.float64,
            ),
            times=torch.arange(6, dtype=torch.float64),
            indices=torch.arange(6),
            contact_indicator=torch.tensor(
                [False, False, False, True, True, True],
            ),
        )

        self.assertAlmostEqual(float(smoothness), 0.0, places=15)

    def test_oscillatory_residual_has_positive_smoothness(self) -> None:
        residual = torch.tensor(
            [0.0, 1.0, -1.0, 1.0, -1.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        smoothness = residual_smoothness_torch(
            normalized_residual=residual,
            times=torch.arange(5, dtype=torch.float64),
            indices=torch.arange(5),
            contact_indicator=torch.zeros(5, dtype=torch.bool),
        )

        self.assertGreater(float(smoothness.detach()), 0.0)
        smoothness.backward()
        self.assertGreater(float(torch.linalg.vector_norm(residual.grad)), 0.0)


if __name__ == "__main__":
    unittest.main()
