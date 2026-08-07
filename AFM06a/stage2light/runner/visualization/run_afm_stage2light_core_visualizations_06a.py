"""Generate AFM06a stage2light core optimization diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_repo_root() -> Path:
    here = Path(__file__).resolve()
    for path in [here.parent, *here.parents]:
        if (path / "AFM06a" / "stage2light").is_dir():
            root = str(path)
            if root not in sys.path:
                sys.path.insert(0, root)
            return path
    raise RuntimeError(f"Could not locate HNODECB root from {here}")


REPO_ROOT = _bootstrap_repo_root()

from AFM06a.stage2light.runner.visualization._common import load_visualization_context
from AFM06a.stage2light.runner.visualization.plots import (
    plot_grad_norm,
    plot_losses,
    plot_reconstruction,
    plot_state_bar_fts_losses,
)


def main() -> None:
    context = load_visualization_context()
    plot_losses(context)
    plot_grad_norm(context)
    plot_reconstruction(context)
    plot_state_bar_fts_losses(context)
    print("[done] AFM06a stage2light core visualizations generated")


if __name__ == "__main__":
    main()
