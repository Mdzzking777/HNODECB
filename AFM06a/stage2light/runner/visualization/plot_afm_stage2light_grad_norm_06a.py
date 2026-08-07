from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM06a.stage2light.runner.visualization.plots import plot_grad_norm

if __name__ == "__main__":
    plot_grad_norm()
