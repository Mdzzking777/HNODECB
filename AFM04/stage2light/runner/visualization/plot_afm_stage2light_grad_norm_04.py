from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.stage2light.runner.visualization._common import (
    discover_log_paths,
    finalize_and_save,
    load_result_payloads,
    out_path,
    plot_phase_series,
)
from AFM04.stage2light.runner.visualization.plot_afm_stage2light_losses_04 import (
    sample_points,
    shard_title,
)


EPOCH_RE = re.compile(r"KAN epoch (\d+) train=")
LBFGS_EPOCH_RE = re.compile(r"KAN LBFGS step \d+ epoch (\d+) train=")
GRAD_RE = re.compile(r"\bgrad_norm=([0-9eE+\-.]+|NaN|Inf|-Inf)\b")
GRAD_NN_RE = re.compile(r"\bgrad_norm_nn=([0-9eE+\-.]+|NaN|Inf|-Inf)\b")
GRAD_MECH_RE = re.compile(r"\bgrad_norm_mech=([0-9eE+\-.]+|NaN|Inf|-Inf)\b")
ROLE_RE = re.compile(r"role=([^|]+)")
LABEL_RE = re.compile(r"label=([^|]+)")


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
DEFAULT_LOG_DIR = REPO_ROOT / "AFM04" / "stage2light" / "logs" / "window_per_shard"
DEFAULT_RESULT_DIR = REPO_ROOT / "AFM04" / "stage2light" / "results"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "stage2light" / "logs" / "visualization"
DEFAULT_OUT_FILE = "afm_param_stage2light_04_grad_norm_triptych.png"
DEFAULT_SAMPLE_EVERY = 1
HISTORY_RE = re.compile(r"stage2light_history_p(\d+)\.json$")
LOG_RE = re.compile(r"log2_04_step2a_stage2light_local_p(\d+)\.txt$")

SERIES = [
    ("grad_norm", "grad_norm", "purple"),
    ("grad_norm_mech", "grad_norm_mech", "seagreen"),
    ("grad_norm_nn", "grad_norm_nn", "firebrick"),
]


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def shard_index(path: Path, pattern: re.Pattern[str]) -> int | None:
    match = pattern.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def infer_result_dir(log_dir: Path) -> Path:
    if log_dir.name == "window_per_shard" and log_dir.parent.name == "logs":
        return log_dir.parent.parent / "results"
    if log_dir.name == "logs":
        archive_result_dir = log_dir.parent / "result"
        if archive_result_dir.exists():
            return archive_result_dir
        sibling_results_dir = log_dir.parent / "results"
        if sibling_results_dir.exists():
            return sibling_results_dir
    return DEFAULT_RESULT_DIR


def discover_history_paths(result_dir: Path) -> list[Path]:
    paths: list[tuple[int, Path]] = []
    for path in result_dir.glob("stage2light_history_p*.json"):
        index = shard_index(path, HISTORY_RE)
        if index is not None:
            paths.append((index, path))
    return [path for _, path in sorted(paths, key=lambda item: item[0])]


def log_identity(log_path: Path | None) -> tuple[str, str]:
    if log_path is None or not log_path.exists():
        return "", ""

    role = ""
    label = log_path.name
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not role:
            match = ROLE_RE.search(line)
            if match is not None:
                role = match.group(1).strip()
        if label == log_path.name:
            match = LABEL_RE.search(line)
            if match is not None:
                label = match.group(1).strip()
        if role and label != log_path.name:
            break
    return role, label


def phase_from_history(record: dict) -> str:
    optimizer = str(record.get("optimizer", "")).lower()
    phase = str(record.get("phase", "")).lower()
    if optimizer == "lbfgs" or "lbfgs" in phase:
        return "lbfgs"
    return "adam"


def parse_history_records_grad_norms(
    raw_history: list,
    *,
    path: str,
    label: str,
    role: str,
    source: str,
) -> dict:
    if not isinstance(raw_history, list):
        raise RuntimeError(f"Unexpected history format in: {path}")

    values_by_name: dict[str, dict[int, float]] = {name: {} for name, _, _ in SERIES}
    phases_by_epoch: dict[int, str] = {}

    for record in raw_history:
        if not isinstance(record, dict) or record.get("epoch") is None:
            continue
        epoch = int(record["epoch"])
        phases_by_epoch[epoch] = phase_from_history(record)
        for name, _, _ in SERIES:
            value = record.get(name)
            if value is not None:
                values_by_name[name][epoch] = float(value)

    series: dict[str, dict[str, list]] = {}
    for name, _, _ in SERIES:
        epochs = sorted(values_by_name[name])
        series[name] = {
            "epochs": epochs,
            "values": [values_by_name[name][epoch] for epoch in epochs],
            "phases": [phases_by_epoch.get(epoch, "adam") for epoch in epochs],
        }

    return {
        "path": path,
        "label": label,
        "role": role if role else label,
        "series": series,
        "source": source,
    }


def parse_history_grad_norms(history_path: Path, log_path: Path | None = None) -> dict:
    raw_history = json.loads(history_path.read_text(encoding="utf-8"))

    role, label = log_identity(log_path)
    if not label:
        label = history_path.name
    return parse_history_records_grad_norms(
        raw_history,
        path=str(history_path),
        label=label,
        role=role if role else label,
        source="history",
    )


def parse_result_payload_grad_norms(payload: dict, log_path: Path | None = None) -> dict | None:
    raw_history = payload.get("history")
    if not isinstance(raw_history, list) or not raw_history:
        return None

    source_path = str(payload.get("_source_path", "stage2light_result_payload"))
    role, label = log_identity(log_path)
    window_meta = payload.get("window_meta")
    if not isinstance(window_meta, dict):
        window_meta = {}
    if not role:
        role = str(window_meta.get("role", "")).strip()
    if not label:
        label = str(window_meta.get("label", "")).strip()
    if not label:
        label = Path(source_path).name

    return parse_history_records_grad_norms(
        raw_history,
        path=source_path,
        label=label,
        role=role if role else label,
        source="result",
    )


def parse_result_grad_norms_all(result_dir: Path, log_dir: Path) -> list[dict]:
    try:
        payloads = load_result_payloads(result_dir)
    except FileNotFoundError:
        return []

    log_by_index: dict[int, Path] = {}
    for log_path in discover_log_paths(log_dir):
        index = shard_index(log_path, LOG_RE)
        if index is not None:
            log_by_index[index] = log_path

    recs: list[dict] = []
    for payload in payloads:
        source_path = Path(str(payload.get("_source_path", "")))
        index = shard_index(source_path, re.compile(r"stage2light_result_p(\d+)(?:\.viz\.pkl|\.pt)$"))
        rec = parse_result_payload_grad_norms(payload, log_by_index.get(index))
        if rec is not None:
            recs.append(rec)
    return recs


def parse_history_grad_norms_all(result_dir: Path, log_dir: Path) -> list[dict]:
    history_paths = discover_history_paths(result_dir)
    if not history_paths:
        return []

    log_by_index: dict[int, Path] = {}
    for log_path in discover_log_paths(log_dir):
        index = shard_index(log_path, LOG_RE)
        if index is not None:
            log_by_index[index] = log_path

    recs: list[dict] = []
    for history_path in history_paths:
        index = shard_index(history_path, HISTORY_RE)
        recs.append(parse_history_grad_norms(history_path, log_by_index.get(index)))
    return recs


def parse_grad_norms(log_path: Path) -> dict:
    values_by_name: dict[str, dict[int, float]] = {name: {} for name, _, _ in SERIES}
    phases_by_epoch: dict[int, str] = {}
    role = ""
    label = log_path.name
    current_epoch: int | None = None
    current_phase = "adam"

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not role:
            match = ROLE_RE.search(line)
            if match is not None:
                role = match.group(1).strip()
        if label == log_path.name:
            match = LABEL_RE.search(line)
            if match is not None:
                label = match.group(1).strip()

        match = EPOCH_RE.search(line)
        if match is not None:
            current_epoch = int(match.group(1))
            current_phase = "adam"
            continue

        match = LBFGS_EPOCH_RE.search(line)
        if match is not None:
            current_epoch = int(match.group(1))
            current_phase = "lbfgs"
            continue

        if current_epoch is None:
            continue

        total_match = GRAD_RE.search(line)
        nn_match = GRAD_NN_RE.search(line)
        mech_match = GRAD_MECH_RE.search(line)

        if total_match is not None:
            values_by_name["grad_norm"][current_epoch] = parse_float(total_match.group(1))
            phases_by_epoch[current_epoch] = current_phase

        if nn_match is not None:
            values_by_name["grad_norm_nn"][current_epoch] = parse_float(nn_match.group(1))
            phases_by_epoch[current_epoch] = current_phase

        if mech_match is not None:
            values_by_name["grad_norm_mech"][current_epoch] = parse_float(mech_match.group(1))
            phases_by_epoch[current_epoch] = current_phase

        if total_match is not None or nn_match is not None or mech_match is not None:
            if mech_match is not None:
                current_epoch = None
            continue

    series: dict[str, dict[str, list]] = {}
    for name, _, _ in SERIES:
        epochs = sorted(values_by_name[name])
        series[name] = {
            "epochs": epochs,
            "values": [values_by_name[name][epoch] for epoch in epochs],
            "phases": [phases_by_epoch.get(epoch, "adam") for epoch in epochs],
        }

    return {
        "path": str(log_path),
        "label": label,
        "role": role if role else label,
        "series": series,
        "source": "log",
    }


def build_subplot(ax, rec: dict, series_name: str, ylabel: str, color: str, *, sample_every: int) -> None:
    series = rec["series"][series_name]
    epochs = series["epochs"]
    values = series["values"]
    ax.set_title(f"{shard_title(rec['role'], rec['label'])} | {ylabel}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    if not epochs:
        ax.text(0.5, 0.5, f"no {ylabel} data", transform=ax.transAxes, ha="center", va="center", color="darkred")
        return

    sample_epoch, sample_value, sample_phase = sample_points(
        epochs,
        values,
        sample_every,
        series.get("phases"),
    )
    y_values = [value for value in sample_value if is_finite(value) and value > 0.0]
    y_min = min(y_values) / 1.8 if y_values else 1.0e-12
    y_max = max(y_values) * 1.8 if y_values else 1.0

    plot_phase_series(
        ax,
        sample_epoch,
        sample_value,
        sample_phase,
        label=ylabel,
        color=color,
        linewidth=2,
        marker="o",
        markersize=3,
    )

    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, epochs[-1] + 1)
    ax.legend(loc="best")


def run_one(log_dir: Path, out_dir: Path, sample_every: int) -> Path:
    result_dir = infer_result_dir(log_dir)
    recs = parse_result_grad_norms_all(result_dir, log_dir)
    if recs:
        source = f"result payload under {result_dir}"
    else:
        log_paths = discover_log_paths(log_dir)
        if log_paths:
            recs = [parse_grad_norms(path) for path in log_paths]
            source = f"log text under {log_dir}"
        else:
            recs = parse_history_grad_norms_all(result_dir, log_dir)
            if recs:
                source = f"history JSON under {result_dir}"
            else:
                raise FileNotFoundError(
                    f"No stage2light result payload under {result_dir}, shard logs under {log_dir}, "
                    f"or history JSON under {result_dir}"
                )

    ncols = max(1, len(recs))
    fig, axes = plt.subplots(len(SERIES), ncols, figsize=(6.4 * ncols, 4.0 * len(SERIES)), sharex=False, squeeze=False)

    for col_idx, rec in enumerate(recs):
        for row_idx, (series_name, ylabel, color) in enumerate(SERIES):
            build_subplot(axes[row_idx][col_idx], rec, series_name, ylabel, color, sample_every=sample_every)

    out_file = out_path(DEFAULT_OUT_FILE, out_dir)
    finalize_and_save(fig, out_file)

    print(f"Source: {source}")
    print(f"Sampled every {sample_every} epochs.")
    for rec in recs:
        title = shard_title(rec["role"], rec["label"])
        parts: list[str] = []
        for series_name, ylabel, _ in SERIES:
            epochs = rec["series"][series_name]["epochs"]
            if epochs:
                parts.append(f"{ylabel}: {epochs[0]}-{epochs[-1]} ({len(epochs)} pts)")
            else:
                parts.append(f"{ylabel}: none")
        print(f"{title}: " + " | ".join(parts))
    return out_file


def main() -> None:
    log_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_LOG_DIR
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    sample_every = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_SAMPLE_EVERY
    run_one(log_dir, out_dir, sample_every)


if __name__ == "__main__":
    main()
