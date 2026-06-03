"""CLI entrypoint for the isolated AFM04 KAN full functional test."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from AFM04.KAN_full_test.config import default_config
from AFM04.KAN_full_test.prerun import run_prerun_driver, run_prerun_shard
from AFM04.KAN_full_test.random_search import (
    merge_random_search_results,
    run_random_search_shard,
    write_random_search_seed_bank,
)
from AFM04.KAN_full_test.train import merge_shard_results, run_full_test_shard


def _timestamp_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _driver_log_path(cfg) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg.log_dir / f"log2_04_step2a_kan_full_test_local_driver_{stamp}.txt"


def _window_mode_norm(cfg) -> str:
    return str(cfg.window_mode).strip().lower()


def _is_modified_w0(cfg) -> bool:
    return _window_mode_norm(cfg) in ("modified_w0", "modified-w0", "shifted_w0", "shifted-w0")


def _is_w0(cfg) -> bool:
    return _window_mode_norm(cfg) in ("w0", "stage2_w0", "stage2-w0", "first_contact", "first-contact")


def _selected_window_text(cfg) -> str:
    if _is_modified_w0(cfg):
        return "modified W0"
    if _is_w0(cfg):
        return "W0"
    if cfg.random_search_window_index > 0:
        return f"W{cfg.random_search_window_index}"
    return f"{cfg.shard_count} selected windows"


def _shard_log_path(cfg, shard_index: int) -> Path:
    if _is_modified_w0(cfg):
        return cfg.shard_log_dir / "log2_04_step2a_kan_full_test_local_modified_w0.txt"
    if _is_w0(cfg):
        return cfg.shard_log_dir / "log2_04_step2a_kan_full_test_local_w0.txt"
    return cfg.shard_log_dir / f"log2_04_step2a_kan_full_test_local_p{shard_index}.txt"


def _random_search_driver_log_path(cfg) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg.random_search_log_dir / f"log2_04_step2a_kan_full_test_random_search_local_driver_{stamp}.txt"


def _random_search_shard_log_path(cfg, subshard_index: int) -> Path:
    return cfg.random_search_shard_log_dir / f"log2_04_step2a_kan_full_test_random_search_local_s{subshard_index}.txt"


def _log_line(log, message: str) -> None:
    line = f"[{_timestamp_human()}] {message}"
    log.write(line + "\n")
    log.flush()
    print(line, flush=True)


def run_driver(
    cfg,
    *,
    auto_generate_dataset: bool,
    warmstart_from_random_search: bool = False,
    warmstart_fallback_random: bool = False,
) -> dict[str, object]:
    driver_log = _driver_log_path(cfg)
    driver_log.parent.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, subprocess.Popen[str], Path]] = []
    base_env = dict(os.environ)
    base_env["HNODECB_AFM04_KAN_TEST_SHARD_COUNT"] = str(cfg.shard_count)
    base_env["HNODECB_AFM04_KAN_TEST_WARMSTART_FROM_RANDOM_SEARCH"] = "1" if warmstart_from_random_search else "0"
    base_env["HNODECB_AFM04_KAN_TEST_WARMSTART_FALLBACK_RANDOM"] = "1" if warmstart_fallback_random else "0"
    if auto_generate_dataset:
        base_env["HNODECB_AFM04_KAN_TEST_AUTOGEN_DATASET"] = "1"

    with driver_log.open("w", encoding="utf-8") as log:
        _log_line(log, f"AFM04 KAN full test driver start | shard_count={cfg.shard_count}")
        _log_line(log, f"Logs -> {cfg.shard_log_dir}")
        _log_line(log, f"Results -> {cfg.result_dir}")
        for shard_index in range(1, cfg.shard_count + 1):
            shard_log = _shard_log_path(cfg, shard_index)
            env = dict(base_env)
            env["HNODECB_AFM04_KAN_TEST_SHARD_INDEX"] = str(shard_index)
            proc = subprocess.Popen(
                [sys.executable, "-m", "AFM04.KAN_full_test.main", "--shard-index", str(shard_index)],
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
            raise RuntimeError(f"AFM04 KAN full test driver failed: {failed}")

        merged = merge_shard_results(cfg)
        _log_line(log, f"driver merge complete | best_val={merged['best_val_loss']:.6e}")
        return {
            "driver_log": str(driver_log),
            "exit_codes": exit_codes,
            "merged_summary": merged,
        }


def run_random_search_driver(cfg, *, auto_generate_dataset: bool) -> dict[str, object]:
    driver_log = _random_search_driver_log_path(cfg)
    driver_log.parent.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, int, subprocess.Popen[str], Path]] = []
    base_env = dict(os.environ)
    base_env["HNODECB_AFM04_KAN_TEST_SHARD_COUNT"] = str(cfg.shard_count)
    base_env["HNODECB_AFM04_KAN_TEST_WARMSTART_FROM_RANDOM_SEARCH"] = "0"
    base_env["HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_SUBSHARD_COUNT"] = str(cfg.random_search_subshard_count)
    base_env["HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_WINDOW_INDEX"] = str(cfg.random_search_window_index)
    if auto_generate_dataset:
        base_env["HNODECB_AFM04_KAN_TEST_AUTOGEN_DATASET"] = "1"

    with driver_log.open("w", encoding="utf-8") as log:
        total_workers = cfg.random_search_subshard_count
        selected_window_text = _selected_window_text(cfg)
        _log_line(
            log,
            "AFM04 KAN random-search driver start | "
            f"windows={selected_window_text} compute_shards={cfg.random_search_subshard_count} "
            f"workers={total_workers} trials_per_window={cfg.random_search_trials}",
        )
        seed_bank = write_random_search_seed_bank(cfg)
        _log_line(log, f"Logs -> {cfg.random_search_shard_log_dir}")
        _log_line(log, f"Results -> {cfg.random_search_result_dir}")
        _log_line(
            log,
            "Trial seeds -> "
            f"mode={'true-random' if seed_bank['true_random'] else 'deterministic-bank'} "
            f"| count={len(seed_bank['trial_seeds'])} | path={seed_bank['path']}",
        )
        for subshard_index in range(1, cfg.random_search_subshard_count + 1):
            shard_log = _random_search_shard_log_path(cfg, subshard_index)
            env = dict(base_env)
            env["HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_SUBSHARD_INDEX"] = str(subshard_index)
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "AFM04.KAN_full_test.main",
                    "--random-search-shard",
                    "--random-search-subshard-index",
                    str(subshard_index),
                ],
                cwd=str(cfg.repo_root),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            procs.append((subshard_index, proc, shard_log))
            _log_line(
                log,
                "launched random-search worker "
                f"S{subshard_index}/{cfg.random_search_subshard_count} "
                f"| pid={proc.pid} | log={shard_log}",
            )

        exit_codes: dict[int, int] = {}
        try:
            for subshard_index, proc, shard_log in procs:
                code = proc.wait()
                exit_codes[subshard_index] = int(code)
                _log_line(
                    log,
                    "random-search worker exited | "
                    f"S{subshard_index}/{cfg.random_search_subshard_count} "
                    f"| code={code} | log={shard_log}",
                )
        finally:
            for _subshard_index, proc, _shard_log in procs:
                if proc.poll() is None:
                    proc.kill()

        failed = {idx: code for idx, code in exit_codes.items() if code != 0}
        if failed:
            _log_line(log, f"random-search driver failed | failed_shards={failed}")
            raise RuntimeError(f"AFM04 KAN random-search driver failed: {failed}")

        merged = merge_random_search_results(cfg)
        _log_line(
            log,
            "random-search merge complete | "
            f"best_rank({merged.get('ranking_metric', 'val_loss')})={merged['best_ranking_loss']:.6e}",
        )
        return {
            "driver_log": str(driver_log),
            "exit_codes": exit_codes,
            "merged_summary": merged,
        }


def run_pipeline_driver(cfg, *, auto_generate_dataset: bool) -> dict[str, object]:
    rs = run_random_search_driver(cfg, auto_generate_dataset=auto_generate_dataset)
    train = run_driver(
        cfg,
        auto_generate_dataset=auto_generate_dataset,
        warmstart_from_random_search=True,
        warmstart_fallback_random=False,
    )
    return {"random_search": rs, "train": train}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", action="store_true")
    parser.add_argument("--with-random-search", action="store_true")
    parser.add_argument("--random-search-driver", action="store_true")
    parser.add_argument("--random-search-shard", action="store_true")
    parser.add_argument("--prerun-driver", action="store_true")
    parser.add_argument("--prerun-shard", action="store_true")
    parser.add_argument("--prerun-layer", type=str, default="layer_a")
    parser.add_argument("--prerun-assignment-path", type=str, default=None)
    parser.add_argument("--auto-generate-dataset", action="store_true")
    parser.add_argument("--warmstart-from-random-search", action="store_true")
    parser.add_argument("--warmstart-fallback-random", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--random-search-window-index", type=int, default=None)
    parser.add_argument("--random-search-subshard-index", type=int, default=None)
    args = parser.parse_args()

    cfg = default_config()
    if args.random_search_window_index is not None:
        cfg = replace(cfg, random_search_window_index=max(0, int(args.random_search_window_index)))
    if args.random_search_subshard_index is not None:
        cfg = replace(cfg, random_search_subshard_index=max(1, int(args.random_search_subshard_index)))
    if args.random_search_driver:
        result = run_random_search_driver(cfg, auto_generate_dataset=args.auto_generate_dataset)
    elif args.random_search_shard:
        result = run_random_search_shard(
            cfg,
            subshard_index=args.random_search_subshard_index,
        )
    elif args.prerun_driver:
        result = run_prerun_driver(cfg)
    elif args.prerun_shard:
        result = run_prerun_shard(
            cfg,
            shard_index=args.shard_index,
            layer_name=args.prerun_layer,
            assignment_path=args.prerun_assignment_path,
        )
    elif args.driver and args.with_random_search:
        result = run_pipeline_driver(cfg, auto_generate_dataset=args.auto_generate_dataset)
    elif args.driver:
        result = run_driver(
            cfg,
            auto_generate_dataset=args.auto_generate_dataset,
            warmstart_from_random_search=args.warmstart_from_random_search,
            warmstart_fallback_random=args.warmstart_fallback_random,
        )
    else:
        result = run_full_test_shard(cfg, shard_index=args.shard_index)
    print("KAN full test completed")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
