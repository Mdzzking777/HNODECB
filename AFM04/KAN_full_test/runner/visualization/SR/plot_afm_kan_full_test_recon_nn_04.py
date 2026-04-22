from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from AFM04.KAN_full_test.runner.visualization.SR._bridge_sr import run_target


if __name__ == "__main__":
    run_target("plot_afm_kan_full_test_recon_nn_04")
