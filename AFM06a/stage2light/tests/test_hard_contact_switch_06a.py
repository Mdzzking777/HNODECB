from __future__ import annotations

import math
import unittest

import torch

from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_functions import (
    bar_f_ts_from_distance,
    effective_damping_from_distance,
)


class HardContactSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a0 = 2.0
        self.omega0 = 3.0
        self.c1 = -0.5
        self.c2 = 4.0
        self.eta = 5.0
        self.beta = 123.0
        self.ca = self.c1 * self.omega0**2 * self.eta**3
        self.ch = self.c2 * self.omega0**2 / math.sqrt(self.eta)

    def force(self, s):
        return bar_f_ts_from_distance(
            s,
            omega0=self.omega0,
            c1=self.c1,
            c2=self.c2,
            eta_star_length=self.eta,
            a0=self.a0,
            beta=self.beta,
        )

    def test_scalar_force_uses_exact_noncontact_and_contact_branches(self) -> None:
        noncontact_s = 3.0
        contact_s = 1.5
        self.assertAlmostEqual(self.force(noncontact_s), self.ca / noncontact_s**2)
        self.assertAlmostEqual(
            self.force(contact_s),
            self.ca / self.a0**2 + self.ch * (self.a0 - contact_s) ** 1.5,
        )
        self.assertAlmostEqual(self.force(self.a0), self.ca / self.a0**2)

    def test_tensor_force_matches_scalar_hard_switch(self) -> None:
        values = torch.tensor([3.0, 2.0, 1.5], dtype=torch.float64)
        actual = self.force(values)
        expected = torch.tensor(
            [self.force(float(value)) for value in values],
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual, expected)

    def test_damping_switches_at_contact_boundary(self) -> None:
        d1 = 0.1
        d2 = 2.0
        self.assertEqual(
            effective_damping_from_distance(
                self.a0 + 1.0e-6,
                d1=d1,
                d2=d2,
                a0=self.a0,
                beta=self.beta,
            ),
            d1,
        )
        self.assertEqual(
            effective_damping_from_distance(
                self.a0,
                d1=d1,
                d2=d2,
                a0=self.a0,
                beta=self.beta,
            ),
            d2,
        )


if __name__ == "__main__":
    unittest.main()
