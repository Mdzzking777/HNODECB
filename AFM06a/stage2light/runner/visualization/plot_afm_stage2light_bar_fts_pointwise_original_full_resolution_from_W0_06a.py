from pathlib import Path
import sys


for _path in Path(__file__).resolve().parents:
    if (_path / "AFM06a" / "stage2light").is_dir():
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
        break
else:
    raise RuntimeError("Could not locate the HNODECB repository root")

from AFM06a.stage2light.runner.visualization.plots import (
    plot_bar_fts_pointwise_full_resolution_from_w0,
)


if __name__ == "__main__":
    plot_bar_fts_pointwise_full_resolution_from_w0()
