from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
RESULT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "results"
OUT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "visualization"


def load_result_payloads(result_dir: Path = RESULT_DIR) -> list[dict]:
    payloads: list[dict] = []
    for shard_index in (1, 2, 3):
        viz_path = result_dir / f"kan_full_test_result_p{shard_index}.viz.pkl"
        pt_path = result_dir / f"kan_full_test_result_p{shard_index}.pt"
        if viz_path.is_file():
            with viz_path.open("rb") as f:
                payload = pickle.load(f)
        elif pt_path.is_file():
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Visualization needs either a '.viz.pkl' sidecar or a Python env with 'torch' "
                    f"to load legacy result file: {pt_path}"
                ) from exc
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        else:
            continue
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected payload type for shard {shard_index}")
        payloads.append(payload)
    if not payloads:
        raise FileNotFoundError(f"No shard results found under: {result_dir}")
    return payloads


def window_title(payload: dict) -> str:
    meta = payload.get("window_meta", {})
    return str(meta.get("title", meta.get("role", "window")))


def stage_title(payload: dict) -> str:
    meta = payload.get("window_meta", {})
    role = str(meta.get("role", ""))
    if role == "first_contact":
        return "W1"
    if role == "max_x1_pp_change":
        return "W2"
    if role == "tail_stable":
        return "W3"
    return role or "W?"


def out_path(filename: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def finalize_and_save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {path}")
    return path
