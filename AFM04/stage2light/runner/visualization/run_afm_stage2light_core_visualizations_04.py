from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    "plot_afm_stage2light_losses_04.py",
    "plot_afm_stage2light_mech_04.py",
    "plot_afm_stage2light_recon_nn_04.py",
    "plot_afm_stage2light_state_x2dot_04.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    python_exe = sys.executable

    for script_name in SCRIPT_ORDER:
        script_path = here / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Missing visualization script: {script_path}")
        print(f"[run] {script_name}")
        subprocess.run([python_exe, str(script_path)], check=True)

    print("[done] AFM04 stage2light core visualizations generated")


if __name__ == "__main__":
    main()
