"""Command-line entrypoint for AFM06a stage2light."""

from __future__ import annotations

import argparse
import json

from AFM06a.stage2light.config import default_config
from AFM06a.stage2light.archive import archive_completed_stage2light_run
from AFM06a.stage2light.data import load_stage1_endpoint
from AFM06a.stage2light.train import backward_smoke, replay_stage1_endpoint, run_stage2light


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="AFM06a stage2light")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--backward-smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--rank", type=int, default=None)
    args = parser.parse_args()
    config = default_config()
    if args.rank is not None:
        from dataclasses import replace

        config = replace(config, input_rank=int(args.rank))
    if args.smoke:
        _print(replay_stage1_endpoint(config))
    elif args.backward_smoke:
        _print(backward_smoke(config))
    elif args.run:
        training = run_stage2light(config)
        archive = archive_completed_stage2light_run(config)
        _print({"training": training, "archive": archive})
    else:
        config.validate()
        endpoint = load_stage1_endpoint(config)
        _print(
            {
                "ready": True,
                "input_rank": endpoint.rank,
                "trial_id": endpoint.record["params"]["trial_id"],
                "kan_width": endpoint.record["params"]["kan_width"],
                "entry_policy": config.entry_policy,
                "epoch0_preparation": config.epoch0_preparation,
                "epochs": config.epochs,
                "adam_epochs": config.adam_epochs,
                "lbfgs_epochs": config.lbfgs_epochs,
                "lbfgs_sparsification_enabled": config.lbfgs_sparsification_enabled,
                "lbfgs_sparsification_lambda": config.lbfgs_sparsification_lambda,
                "prune_on_first_zero_step": config.prune_on_first_zero_step,
                "lbfgs_zero_step_max_events": config.lbfgs_zero_step_max_events,
                "adaptive_grid_enabled": config.adaptive_grid_enabled,
                "agu_update_every_adam_epoch": config.agu_update_every_adam_epoch,
            }
        )


if __name__ == "__main__":
    main()
