from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    ("plot_afm_kan_full_test_losses_04.py", "log"),
    ("plot_afm_kan_full_test_grad_norm_04.py", "log"),
    ("plot_afm_kan_full_test_q_init_04.py", "log"),
    ("plot_afm_kan_full_test_recon_nn_04.py", "log"),
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

    py_launcher = shutil.which("py")
    if py_launcher:
        for version in ("-3.13", "-3.12", "-3.11", "-3"):
            add([py_launcher, version])
    return candidates


def _supports_required_modules(cmd: list[str]) -> bool:
    try:
        result = subprocess.run(
            [*cmd, "-c", "import matplotlib; print('OK')"],
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
    raise RuntimeError(f"No usable Python interpreter with matplotlib was found. Tried: {tried}")


def main() -> None:
    repo_root = _repo_root()
    here = Path(__file__).resolve().parent
    log_dir = repo_root / "AFM04" / "KAN_full_test" / "logs" / "window_per_shard"
    out_dir = repo_root / "AFM04" / "KAN_full_test" / "logs" / "visualization"
    python_cmd = _select_python_command(repo_root)
    env = _clean_env_for_command(python_cmd)

    print(f"[python] using {' '.join(python_cmd)}")
    print(f"[logs] {log_dir}")

    for script_name, mode in SCRIPT_ORDER:
        script_path = here / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Missing visualization script: {script_path}")
        print(f"[run] {script_name}")
        args = [*python_cmd, str(script_path)]
        if mode == "log":
            args.extend([str(log_dir), str(out_dir)])
        subprocess.run(args, check=True, env=env)

    print("[done] AFM04 KAN full-test core visualizations generated")


if __name__ == "__main__":
    main()
