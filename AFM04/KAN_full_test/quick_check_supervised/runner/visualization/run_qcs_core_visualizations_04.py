from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    "plot_qcs_losses_04.py",
    "plot_qcs_x1x3_error_04.py",
]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for path in [here.parent, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repository root from {here}")


def _clean_env_for_command(cmd: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__"):
        env.pop(key, None)
    exe_path = Path(cmd[0])
    if exe_path.exists() and exe_path.name.lower() == "python.exe":
        prefix = exe_path.parent
        extra = [str(prefix), str(prefix / "DLLs"), str(prefix / "Library" / "bin"), str(prefix / "Scripts")]
        path_parts = [p for p in extra if Path(p).exists()]
        path_parts.append(env.get("PATH", ""))
        env["PATH"] = ";".join(part for part in path_parts if part)
    return env


def _candidate_commands(repo_root: Path) -> list[list[str]]:
    candidates: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(cmd: list[str]) -> None:
        key = tuple(cmd)
        if key not in seen:
            seen.add(key)
            candidates.append(cmd)

    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.is_file():
        add([str(venv_python)])
    add([sys.executable])
    python_on_path = shutil.which("python")
    if python_on_path:
        add([python_on_path])
    return candidates


def _supports_required_modules(cmd: list[str]) -> bool:
    try:
        result = subprocess.run(
            [*cmd, "-c", "import numpy, matplotlib; print('OK')"],
            check=False,
            capture_output=True,
            text=True,
            env=_clean_env_for_command(cmd),
        )
    except OSError:
        return False
    return result.returncode == 0


def _select_python_command(repo_root: Path) -> list[str]:
    for cmd in _candidate_commands(repo_root):
        if _supports_required_modules(cmd):
            return cmd
    tried = [" ".join(cmd) for cmd in _candidate_commands(repo_root)]
    raise RuntimeError(f"No usable Python interpreter with numpy/matplotlib was found. Tried: {tried}")


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = _repo_root()
    python_cmd = _select_python_command(repo_root)
    env = _clean_env_for_command(python_cmd)
    env["HNODECB_AFM04_KFT_QCS_VIS_ALLOW_LOG_FALLBACK"] = "1"
    print(f"[python] using {' '.join(python_cmd)}")

    for script_name in SCRIPT_ORDER:
        script_path = here / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Missing visualization script: {script_path}")
        print(f"[run] {script_name}")
        subprocess.run([*python_cmd, str(script_path)], check=True, env=env)

    print("[done] KFT quick-check supervised core visualizations generated")


if __name__ == "__main__":
    main()

