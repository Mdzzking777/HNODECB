from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
PREST2_ROOT = REPO_ROOT / "AFM04" / "prestage2"
DEFAULT_DRIVER_LOG = PREST2_ROOT / "logs" / "log2_04_prest2_local_driver_20260427_101556.txt"
DEFAULT_LAYER_A_RESULT = PREST2_ROOT / "results" / "afm_prest2_04_top_mech_winners_a.pkl"
DEFAULT_FINAL_RESULT = PREST2_ROOT / "results" / "afm_prest2_04_candidates_b.pkl"
DEFAULT_OUT_DIR = PREST2_ROOT / "visualization"
DEFAULT_TOP_STAGE1 = 100
DEFAULT_TOP_NEWRANK = 100
DEFAULT_TOP_CANDIDATE = 20
SHRINK_STAGE1_PT = 4.2
SHRINK_NEWRANK_PT = 4.2
SHRINK_CANDIDATE_PT = 4.4


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected payload type for {path}: {type(payload)!r}")
    return payload


def _records(payload: dict[str, Any], *, prefer_final: bool) -> list[dict[str, Any]]:
    primary = "candidate_b_records" if prefer_final else "top_mech_winner_a_records"
    secondary = "candidate_records" if prefer_final else "newrank_a_records"
    records = payload.get(primary, [])
    if isinstance(records, list) and records:
        return [rec for rec in records if isinstance(rec, dict)]
    records = payload.get(secondary, [])
    return [rec for rec in records if isinstance(rec, dict)]


def _driver_label(path: Path) -> str:
    return path.name if path.is_file() else "prest2 driver log (not found)"


def _edge_line(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    shrink_a: float,
    shrink_b: float,
    linewidth: float,
    alpha: float,
    color: str,
    zorder: float,
) -> None:
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-",
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        mutation_scale=1.0,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)


def build_flow_plot(
    *,
    driver_log_path: Path,
    layer_a_result_path: Path,
    final_result_path: Path,
    out_dir: Path,
    top_stage1: int,
    top_newrank: int,
    top_candidate: int,
) -> Path:
    layer_a_payload = _load_pickle(layer_a_result_path)
    final_payload = _load_pickle(final_result_path)

    layer_a_records = _records(layer_a_payload, prefer_final=False)
    final_records = _records(final_payload, prefer_final=True)
    if not layer_a_records:
        raise RuntimeError(f"No Layer-A records found in: {layer_a_result_path}")
    if not final_records:
        raise RuntimeError(f"No final candidate records found in: {final_result_path}")

    mech_to_top_a = {
        int(rec["source_mech_winner"]): int(rec.get("top_mech_winner_a", rec.get("newrank_a", 0)))
        for rec in layer_a_records
        if "source_mech_winner" in rec and ("top_mech_winner_a" in rec or "newrank_a" in rec)
    }
    mech_to_candidate_b = {
        int(rec["source_mech_winner"]): int(rec.get("candidate_b", rec.get("candidate", 0)))
        for rec in final_records
        if "source_mech_winner" in rec and ("candidate_b" in rec or "candidate" in rec)
    }

    tracked_mech_winners = list(range(1, top_stage1 + 1))
    tracked_top_a = list(range(1, top_newrank + 1))
    tracked_candidates_b = list(range(1, top_candidate + 1))
    finalist_mech_winners = sorted(
        mech for mech, candidate in mech_to_candidate_b.items() if mech <= top_stage1 and candidate <= top_candidate
    )
    non_finalist_mech_winners = [mech for mech in tracked_mech_winners if mech not in set(finalist_mech_winners)]

    ymax = max(top_stage1, top_newrank, top_candidate)
    if ymax <= 0:
        raise RuntimeError("No valid mech winner -> top mech winner A mapping found for plotting.")

    fig, ax = plt.subplots(figsize=(15.5, 13.5))

    for mech_winner in tracked_mech_winners:
        top_a = mech_to_top_a.get(mech_winner)
        if top_a is None:
            continue
        candidate = mech_to_candidate_b.get(mech_winner)
        is_finalist = candidate is not None and candidate <= top_candidate
        _edge_line(
            ax,
            0.0,
            mech_winner,
            10.0,
            top_a,
            shrink_a=SHRINK_STAGE1_PT,
            shrink_b=SHRINK_NEWRANK_PT,
            linewidth=0.85,
            alpha=0.90 if is_finalist else 0.72,
            color="#dc2626" if is_finalist else "black",
            zorder=10,
        )
        if is_finalist:
            _edge_line(
                ax,
                10.0,
                top_a,
                20.0,
                candidate,
                shrink_a=SHRINK_NEWRANK_PT,
                shrink_b=SHRINK_CANDIDATE_PT,
                linewidth=0.95,
                alpha=0.88,
                color="black",
                zorder=10,
            )

    ax.scatter(
        [0.0] * len(non_finalist_mech_winners),
        non_finalist_mech_winners,
        s=34,
        marker="o",
        color="#1a8f3f",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    if finalist_mech_winners:
        ax.scatter(
            [0.0] * len(finalist_mech_winners),
            finalist_mech_winners,
            s=34,
            marker="o",
            facecolors="#1a8f3f",
            edgecolors="#dc2626",
            linewidths=0.9,
            zorder=5,
        )
        for mech_winner in finalist_mech_winners:
            ax.text(
                -0.55,
                mech_winner,
                f"{mech_winner}",
                color="#b91c1c",
                fontsize=6.8,
                fontweight="bold",
                ha="right",
                va="center",
                zorder=6,
            )
    ax.scatter(
        [10.0] * len(tracked_top_a),
        tracked_top_a,
        s=34,
        marker="o",
        color="#2563eb",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    ax.scatter(
        [20.0] * len(tracked_candidates_b),
        tracked_candidates_b,
        s=38,
        marker="o",
        color="#dc2626",
        edgecolors="white",
        linewidths=0.55,
        zorder=4,
    )

    ax.axvline(0.0, color="#b9c0c7", linewidth=1.0, linestyle="--", zorder=0)
    ax.axvline(10.0, color="#b9c0c7", linewidth=1.0, linestyle="--", zorder=0)
    ax.axvline(20.0, color="#b9c0c7", linewidth=1.0, linestyle="--", zorder=0)

    ax.set_xlim(-4.0, 24.0)
    ax.set_ylim(0, ymax + 2)
    ax.set_xticks([10.0, 20.0], labels=["10", "20"])
    ax.set_xlabel("epochs")
    ax.set_ylabel("placements")
    ax.grid(True, axis="y", alpha=0.20)

    title = "AFM04 Gradient-Based Optimization Preliminary Test"
    fig.suptitle(title, y=0.992)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1a8f3f", markeredgecolor="white", markersize=8, label=f"mech winner 1-{top_stage1}"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1a8f3f", markeredgecolor="#dc2626", markeredgewidth=1.2, markersize=8, label=f"mech winners reaching candidate B 1-{top_candidate}"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2563eb", markeredgecolor="white", markersize=8, label=f"top mech winner A 1-{top_newrank}"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#dc2626", markeredgecolor="white", markersize=8, label=f"candidate B 1-{top_candidate}"),
        Line2D([0], [0], color="black", linewidth=1.0, label="mech winner flow"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95)

    ax.text(0.0, 1.01, "mech winner", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)
    ax.text(10.0, 1.01, "top mech winner A", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)
    ax.text(20.0, 1.01, "candidate B", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "afm_prest2_mech_winner_flow_04.png"
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    driver_log_path = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_DRIVER_LOG
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    out_path = build_flow_plot(
        driver_log_path=driver_log_path,
        layer_a_result_path=DEFAULT_LAYER_A_RESULT.resolve(),
        final_result_path=DEFAULT_FINAL_RESULT.resolve(),
        out_dir=out_dir,
        top_stage1=DEFAULT_TOP_STAGE1,
        top_newrank=DEFAULT_TOP_NEWRANK,
        top_candidate=DEFAULT_TOP_CANDIDATE,
    )
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
