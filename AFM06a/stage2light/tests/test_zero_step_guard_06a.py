from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from AFM06a.stage2light.train import (
    _kan_active_hidden_nodes,
    _kan_node_sparsification_penalty,
    _lbfgs_zero_step_action,
    _make_lbfgs,
    _mask_hidden_node,
    _model_state_snapshot,
    _restore_model_state,
    _select_prunable_hidden_node,
    _trainable_parameter_delta_stats,
)


def _config(lr: float) -> SimpleNamespace:
    return SimpleNamespace(
        lbfgs_lr=lr,
        lbfgs_max_iter=1,
        lbfgs_max_eval=10,
        lbfgs_tolerance_grad=1.0e-12,
        lbfgs_tolerance_change=1.0e-15,
        lbfgs_history_size=100,
        lbfgs_ys_threshold=1.0e-15,
    )


def _optimizer_step(model: torch.nn.Linear, optimizer) -> None:
    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.square(model.weight.reshape(()) - 1.0)
        loss.backward()
        return loss

    optimizer.step(closure)


class _FakeNumericLayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.coef = torch.nn.Parameter(
            torch.ones(in_dim, out_dim, 4, dtype=torch.float64)
        )
        self.scale_base = torch.nn.Parameter(
            torch.ones(in_dim, out_dim, dtype=torch.float64)
        )
        self.scale_sp = torch.nn.Parameter(
            torch.ones(in_dim, out_dim, dtype=torch.float64)
        )
        self.mask = torch.nn.Parameter(
            torch.ones(in_dim, out_dim, dtype=torch.float64),
            requires_grad=False,
        )


class _FakeSymbolicLayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mask = torch.nn.Parameter(
            torch.ones(out_dim, in_dim, dtype=torch.float64),
            requires_grad=False,
        )


class _FakeKAN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act_fun = torch.nn.ModuleList(
            [_FakeNumericLayer(1, 3), _FakeNumericLayer(3, 1)]
        )
        self.symbolic_fun = torch.nn.ModuleList(
            [_FakeSymbolicLayer(1, 3), _FakeSymbolicLayer(3, 1)]
        )


class _FakeForceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kan = _FakeKAN()


class AFM06aZeroStepGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.model.weight.data.zero_()

    def test_accepted_lbfgs_step_is_not_classified_as_zero(self) -> None:
        config = _config(1.0)
        snapshot = _model_state_snapshot(self.model)
        optimizer = _make_lbfgs(config, list(self.model.parameters()))

        _optimizer_step(self.model, optimizer)

        _, max_abs = _trainable_parameter_delta_stats(self.model, snapshot)
        self.assertGreater(max_abs, config.lbfgs_tolerance_change)

    def test_persistent_zero_step_is_restored_after_fresh_restart(self) -> None:
        config = _config(0.0)
        snapshot = _model_state_snapshot(self.model)
        optimizer = _make_lbfgs(config, list(self.model.parameters()))

        for attempt in range(2):
            _optimizer_step(self.model, optimizer)
            _, max_abs = _trainable_parameter_delta_stats(self.model, snapshot)
            self.assertLessEqual(max_abs, config.lbfgs_tolerance_change)
            _restore_model_state(self.model, snapshot)
            if attempt == 0:
                optimizer = _make_lbfgs(config, list(self.model.parameters()))

        self.assertEqual(float(self.model.weight.detach()), 0.0)

    def test_zero_step_policy_has_three_global_actions(self) -> None:
        self.assertEqual(_lbfgs_zero_step_action(1), "prune_and_restart")
        self.assertEqual(
            _lbfgs_zero_step_action(1, pruning_enabled=False),
            "restart_only",
        )
        self.assertEqual(_lbfgs_zero_step_action(2), "restart_only")
        self.assertEqual(_lbfgs_zero_step_action(3), "early_stop")
        self.assertEqual(_lbfgs_zero_step_action(4), "early_stop")

    def test_sparsification_penalty_is_finite_and_differentiable(self) -> None:
        model = _FakeForceModel()
        penalty = _kan_node_sparsification_penalty(
            model,
            entropy_weight=2.0,
            epsilon=1.0e-12,
        )
        self.assertTrue(bool(torch.isfinite(penalty)))

        penalty.backward()

        self.assertIsNotNone(model.kan.act_fun[0].scale_base.grad)
        self.assertGreater(
            float(torch.linalg.vector_norm(model.kan.act_fun[0].scale_base.grad)),
            0.0,
        )

    def test_mask_prunes_only_selected_hidden_path(self) -> None:
        model = _FakeForceModel()

        _mask_hidden_node(model, 1)

        self.assertEqual(_kan_active_hidden_nodes(model), [0, 2])
        self.assertTrue(
            bool(torch.all(model.kan.act_fun[0].mask[:, 1] == 0.0))
        )
        self.assertTrue(
            bool(torch.all(model.kan.act_fun[1].mask[1, :] == 0.0))
        )
        self.assertTrue(
            bool(torch.all(model.kan.act_fun[0].mask[:, 0] == 1.0))
        )
        self.assertTrue(
            bool(torch.all(model.kan.act_fun[1].mask[2, :] == 1.0))
        )

    def test_selector_uses_lowest_finite_active_attribution(self) -> None:
        selected = _select_prunable_hidden_node(
            [0.8, 0.2, 0.5],
            [0, 1, 2],
            min_hidden_nodes=1,
        )
        self.assertEqual(selected, 1)
        self.assertIsNone(
            _select_prunable_hidden_node(
                [0.2],
                [0],
                min_hidden_nodes=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
