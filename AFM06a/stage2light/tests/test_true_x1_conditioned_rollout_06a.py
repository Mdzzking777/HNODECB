from __future__ import annotations

import unittest

import torch
from torch import nn

from AFM06a.stage1pluslight.rollout import AFM06aKnownRHS
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (
    AFM06aHardSampleInputs,
)


class RecordingForce(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_states: torch.Tensor | None = None

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        self.last_states = states.detach().clone()
        return torch.zeros(states.shape[0], dtype=states.dtype, device=states.device)


class AFM06aTrueX1ConditionedRolloutTests(unittest.TestCase):
    def test_rhs_force_input_uses_interpolated_observed_x1(self) -> None:
        force = RecordingForce()
        times = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        observed_x1 = torch.tensor([-3.0, -2.0, -1.0], dtype=torch.float64)
        rhs = AFM06aKnownRHS(
            force,
            AFM06aHardSampleInputs(),
            observed_times=times,
            observed_x1=observed_x1,
        )

        rhs(
            torch.tensor(0.5, dtype=torch.float64),
            torch.tensor([99.0, 7.0], dtype=torch.float64),
        )

        self.assertIsNotNone(force.last_states)
        self.assertAlmostEqual(float(force.last_states[0, 0]), -2.5, places=15)
        self.assertAlmostEqual(float(force.last_states[0, 1]), 0.0, places=15)


if __name__ == "__main__":
    unittest.main()
