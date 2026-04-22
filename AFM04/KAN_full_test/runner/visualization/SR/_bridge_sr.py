from __future__ import annotations

import importlib
import sys
from pathlib import Path


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
SR_RESULT_DIR = REPO_ROOT / "AFM04" / "KAN_smooth_reg" / "results"
SR_LOG_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "smoothness regularization" / "window_per_shard"
SR_OUT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "visualization" / "SR"


def _patched_out_path(filename: str, out_dir: Path = SR_OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def run_target(module_basename: str) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    module_name = f"AFM04.KAN_smooth_reg.runner.visualization.{module_basename}"
    common = importlib.import_module("AFM04.KAN_smooth_reg.runner.visualization._common")
    module = importlib.import_module(module_name)

    if hasattr(module, "DEFAULT_LOG_DIR"):
        module.DEFAULT_LOG_DIR = SR_LOG_DIR
    if hasattr(module, "DEFAULT_OUT_DIR"):
        module.DEFAULT_OUT_DIR = SR_OUT_DIR
    if hasattr(module, "load_result_payloads"):
        module.load_result_payloads = lambda: common.load_result_payloads(SR_RESULT_DIR)
    if hasattr(module, "out_path"):
        module.out_path = _patched_out_path

    argv = [sys.argv[0]]
    if len(sys.argv) >= 2:
        argv.extend(sys.argv[1:])
    elif hasattr(module, "DEFAULT_LOG_DIR"):
        argv.extend([str(SR_LOG_DIR), str(SR_OUT_DIR)])
    sys.argv = argv
    module.main()
