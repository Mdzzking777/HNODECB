from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    ("plot_afm_mlp_full_test_losses_04.py", "log"),
    ("plot_afm_mlp_full_test_recon_nn_04.py", "log_or_result"),
    ("plot_afm_mlp_full_test_mech_04.py", "result"),
    ("plot_afm_mlp_full_test_state_x2dot_04.py", "result"),
    ("plot_afm_mlp_full_test_fcontact_traj_04.py", "result"),
    ("plot_afm_mlp_full_test_x1x3_traj_04.py", "result"),
    ("plot_afm_mlp_full_test_x2_x2dot_traj_04.py", "result"),
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

    where_exe = shutil.which("where.exe") or shutil.which("where")
    if where_exe:
        try:
            result = subprocess.run([where_exe, "python"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    path = line.strip()
                    if path:
                        add([path])
        except OSError:
            pass

    py_launcher = shutil.which("py")
    if py_launcher:
        for version in ("-3.13", "-3.12", "-3.11", "-3"):
            add([py_launcher, version])
    return candidates


def _supports_required_modules(cmd: list[str]) -> bool:
    try:
        result = subprocess.run(
            [*cmd, "-c", "import numpy, matplotlib, torch; print('OK')"],
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
    raise RuntimeError(f"No usable Python interpreter with numpy/matplotlib/torch was found. Tried: {tried}")


def _has_log_inputs(repo_root: Path) -> bool:
    log_dir = repo_root / "AFM04" / "MLP_full_test" / "logs" / "window_per_shard"
    return (log_dir / "log2_04_step2a_MLP_full_test_local_w0.txt").is_file()


def _has_result_inputs(repo_root: Path) -> bool:
    result_dir = repo_root / "AFM04" / "MLP_full_test" / "results"
    return (
        (result_dir / "MLP_full_test_result_w0.viz.pkl").is_file()
        or (result_dir / "MLP_full_test_result_w0.pt").is_file()
    )


def _inputs_available(kind: str, *, has_logs: bool, has_results: bool) -> bool:
    if kind == "log":
        return has_logs
    if kind == "result":
        return has_results
    if kind == "log_or_result":
        return has_logs or has_results
    raise ValueError(f"unknown visualization input kind: {kind}")


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = _repo_root()
    python_cmd = _select_python_command(repo_root)
    env = _clean_env_for_command(python_cmd)
    has_logs = _has_log_inputs(repo_root)
    has_results = _has_result_inputs(repo_root)
    print(f"[python] using {' '.join(python_cmd)}")
    print(f"[inputs] logs={'yes' if has_logs else 'no'} results={'yes' if has_results else 'no'}")

    ran_any = False
    for script_name, input_kind in SCRIPT_ORDER:
        script_path = here / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Missing visualization script: {script_path}")
        if not _inputs_available(input_kind, has_logs=has_logs, has_results=has_results):
            print(f"[skip] {script_name} -- missing {input_kind} inputs")
            continue
        print(f"[run] {script_name}")
        subprocess.run([*python_cmd, str(script_path)], check=True, env=env)
        ran_any = True

    if ran_any:
        print("[done] AFM04 MLP full-test all visualizations generated")
    else:
        print("[done] no AFM04 MLP full-test visualization inputs found; nothing generated")


if __name__ == "__main__":
    main()
