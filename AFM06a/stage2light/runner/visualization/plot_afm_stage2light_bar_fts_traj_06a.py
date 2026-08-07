from pathlib import Path
import sys


def _bootstrap_repo_root() -> None:
    here = Path(__file__).resolve()
    for path in [here.parent, *here.parents]:
        if (path / "AFM06a" / "stage2light").is_dir():
            root = str(path)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise RuntimeError(f"Could not locate HNODECB root from {here}")


_bootstrap_repo_root()

from AFM06a.stage2light.runner.visualization.plots import plot_final_bar_fts

if __name__ == "__main__":
    plot_final_bar_fts()
