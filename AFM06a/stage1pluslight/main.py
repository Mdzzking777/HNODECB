"""CLI entry point for the AFM06a stage1pluslight framework."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from AFM06a.stage1pluslight.config import default_config
from AFM06a.stage1pluslight.kan_rs_trial import execute_seed_trial
from AFM06a.stage1pluslight.runner import Stage1Logger, readiness_report, run_shard_mainloop, timestamp_compact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AFM06a stage1pluslight framework")
    parser.add_argument("--check-only", action="store_true", help="Inspect data/contracts without running trials")
    parser.add_argument("--run", action="store_true", help="Request a formal run after all policies are finalized")
    args = parser.parse_args(argv)
    if args.check_only == args.run:
        parser.error("choose exactly one of --check-only or --run")
    config = default_config()
    if args.check_only:
        print(json.dumps(readiness_report(config), indent=2))
        return 0
    configured_log = os.environ.get("HNODECB_AFM06a_STAGE1_LOG_PATH", "").strip()
    if configured_log:
        log_path = Path(configured_log)
    elif config.shard_count > 1:
        log_path = config.log_dir / (
            f"afm06a_stage1pluslight_local_p{config.shard_index}of{config.shard_count}_{timestamp_compact()}.txt"
        )
    else:
        log_path = config.log_dir / f"afm06a_stage1pluslight_{timestamp_compact()}.txt"
    echo_log = os.environ.get("HNODECB_AFM06a_STAGE1_LOG_ECHO", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    _, summary = run_shard_mainloop(
        config,
        executor=execute_seed_trial,
        logger=Stage1Logger(log_path, echo=echo_log),
    )
    print(
        json.dumps(
            {
                "total_trials": summary["total_trials"],
                "viable_trials": summary["viable_trials"],
                "best_loss": summary["best_loss"],
                "result_path": str(config.result_path),
                "log_path": str(log_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
