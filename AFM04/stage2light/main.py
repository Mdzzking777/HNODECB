"""CLI entrypoint for the standalone AFM04 Python stage2light library."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from AFM04.stage2light.archive import archive_completed_stage2light_run
from AFM04.stage2light.config import default_config
from AFM04.stage2light.train import merge_stage2light_results, run_stage2light_shard


def _timestamp_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _driver_log_path(cfg) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg.log_dir / f"log2_04_step2a_stage2light_local_driver_{stamp}.txt"


def _shard_log_path(cfg, shard_index: int) -> Path:
    return cfg.shard_log_dir / f"log2_04_step2a_stage2light_local_p{shard_index}.txt"


def _log_line(log, message: str) -> None:
    line = f"[{_timestamp_human()}] {message}"
    log.write(line + "\n")
    log.flush()
    print(line, flush=True)


def run_driver(cfg, *, auto_generate_dataset: bool) -> dict[str, object]:
    driver_log = _driver_log_path(cfg)
    driver_log.parent.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, subprocess.Popen[str], Path]] = []
    base_env = dict(os.environ)
    base_env["HNODECB_AFM04_STAGE2LIGHT_SHARD_COUNT"] = str(cfg.shard_count)
    if auto_generate_dataset:
        base_env["HNODECB_AFM04_STAGE2LIGHT_AUTOGEN_DATASET"] = "1"

    with driver_log.open("w", encoding="utf-8") as log:
        _log_line(log, f"AFM04 stage2light driver start | shard_count={cfg.shard_count}")
        _log_line(log, f"Logs -> {cfg.shard_log_dir}")
        _log_line(log, f"Results -> {cfg.result_dir}")
        for shard_index in range(1, cfg.shard_count + 1):
            shard_log = _shard_log_path(cfg, shard_index)
            env = dict(base_env)
            env["HNODECB_AFM04_STAGE2LIGHT_SHARD_INDEX"] = str(shard_index)
            proc = subprocess.Popen(
                [sys.executable, "-m", "AFM04.stage2light.main", "--shard-index", str(shard_index)],
                cwd=str(cfg.repo_root),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            procs.append((shard_index, proc, shard_log))
            _log_line(log, f"launched shard {shard_index}/{cfg.shard_count} | pid={proc.pid} | log={shard_log}")

        exit_codes: dict[int, int] = {}
        try:
            for shard_index, proc, shard_log in procs:
                code = proc.wait()
                exit_codes[shard_index] = int(code)
                _log_line(log, f"shard {shard_index}/{cfg.shard_count} exited | code={code} | log={shard_log}")
        finally:
            for _shard_index, proc, _shard_log in procs:
                if proc.poll() is None:
                    proc.kill()

        failed = {idx: code for idx, code in exit_codes.items() if code != 0}
        if failed:
            _log_line(log, f"driver failed | failed_shards={failed}")
            raise RuntimeError(f"AFM04 stage2light driver failed: {failed}")

        merged = merge_stage2light_results(cfg)
        _log_line(log, f"driver merge complete | best_val={merged['best_val_loss']:.6e}")
        archive_summary = archive_completed_stage2light_run(cfg, log=log)
        return {
            "driver_log": str(driver_log),
            "exit_codes": exit_codes,
            "merged_summary": merged,
            "archive_summary": archive_summary,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", action="store_true")
    parser.add_argument("--auto-generate-dataset", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    args = parser.parse_args()

    cfg = default_config()
    if args.driver:
        result = run_driver(cfg, auto_generate_dataset=args.auto_generate_dataset)
    else:
        result = run_stage2light_shard(cfg, shard_index=args.shard_index)
    print("stage2light completed")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
