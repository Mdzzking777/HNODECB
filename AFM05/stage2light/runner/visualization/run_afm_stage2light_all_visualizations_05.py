from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    "plot_afm_stage2light_losses_05.py",
    "plot_afm_stage2light_grad_norm_05.py",
    "plot_afm_stage2light_mech_05.py",
    "plot_afm_stage2light_recon_nn_05.py",
    "plot_afm_stage2light_state_x2dot_05.py",
    "plot_afm_stage2light_x2_x2dot_traj_05.py",
]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for path in [here.parent, *here.parents]:
        if (path / "AFM05" / "stage2light").is_dir():
            return path
    raise RuntimeError(f"Could not locate repository root from {here}")


def _candidate_commands() -> list[list[str]]:
    candidates: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(cmd: list[str]) -> None:
        key = tuple(cmd)
        if key in seen:
            return
        seen.add(key)
        candidates.append(cmd)

    venv_python = _repo_root() / ".venv" / "Scripts" / "python.exe"
    if venv_python.is_file():
        add([str(venv_python)])

    add([sys.executable])

    python_on_path = shutil.which("python")
    if python_on_path:
        add([python_on_path])

    where_exe = shutil.which("where.exe") or shutil.which("where")
    if where_exe:
        try:
            result = subprocess.run(
                [where_exe, "python"],
                check=False,
                capture_output=True,
                text=True,
            )
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
    probe = (
        "import numpy, matplotlib, torch; "
        "print('OK')"
    )
    try:
        result = subprocess.run(
            [*cmd, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            env=_clean_env_for_command(cmd),
        )
    except OSError:
        return False
    return result.returncode == 0


def _clean_env_for_command(cmd: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "PYTHONEXECUTABLE",
        "__PYVENV_LAUNCHER__",
    ):
        env.pop(key, None)

    exe_path = Path(cmd[0])
    if exe_path.exists() and exe_path.name.lower() == "python.exe":
        prefix = exe_path.parent
        extra = [
            str(prefix),
            str(prefix / "DLLs"),
            str(prefix / "Library" / "bin"),
            str(prefix / "Scripts"),
        ]
        path_parts = [p for p in extra if Path(p).exists()]
        path_parts.append(env.get("PATH", ""))
        env["PATH"] = ";".join(part for part in path_parts if part)

    return env


def _select_python_command() -> list[str]:
    for cmd in _candidate_commands():
        if _supports_required_modules(cmd):
            return cmd
    joined = [" ".join(cmd) for cmd in _candidate_commands()]
    raise RuntimeError(
        "No usable Python interpreter with numpy/matplotlib/torch was found. "
        f"Tried: {joined}"
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    python_cmd = _select_python_command()
    print(f"[python] using {' '.join(python_cmd)}")

    for script_name in SCRIPT_ORDER:
        script_path = here / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Missing visualization script: {script_path}")
        print(f"[run] {script_name}")
        subprocess.run(
            [*python_cmd, str(script_path)],
            check=True,
            env=_clean_env_for_command(python_cmd),
        )

    print("[done] AFM05 stage2light all visualizations generated")


if __name__ == "__main__":
    main()
