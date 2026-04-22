"""CLI entrypoint for AFM04 stage1pluslight single-shard execution."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from AFM04.stage1pluslight.config import default_config
from AFM04.stage1pluslight.runner import Stage1Logger, make_shard_log_path, run_shard_mainloop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one AFM04 stage1pluslight shard.")
    parser.add_argument("--auto-generate-dataset", action="store_true", help="Generate AFM04 dataset if missing.")
    args = parser.parse_args(argv)

    config = default_config()
    log_path_env = os.environ.get("HNODECB_AFM04_STAGE1_LOG_PATH", "").strip()
    log_path = Path(log_path_env) if log_path_env else make_shard_log_path(
        config.log_dir, shard_index=config.shard_index, shard_count=config.shard_count, run_tag=config.run_tag
    )
    logger = Stage1Logger(log_path, echo=(log_path_env == ""))
    logger.log(
        "main start | "
        f"shard={config.shard_index}/{config.shard_count} "
        f"window_mode={config.window_mode} "
        "backend=kan_rs_w1 "
        f"result={config.result_path}"
    )
    try:
        _context, summary = run_shard_mainloop(config, auto_generate_dataset=args.auto_generate_dataset, logger=logger)
    except Exception as err:
        logger.log(f"main failed | error={err}")
        raise
    logger.log(
        "main complete | "
        f"best_loss={summary['best_loss']:.6e} "
        f"total_trials={summary['total_trials']} "
        f"viable={summary['viable_trials']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
