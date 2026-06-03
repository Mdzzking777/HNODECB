"""Build the AFM04 stage1pluslight "mech winner" leaderboard.

The report groups stage1pluslight trials by the active mechanical grid node
`(ks_node_idx, cs_node_idx)`. The grid shape and NN seed count are inferred from
the merged stage1pluslight result whenever possible, so the script works for
10x10x200, 10x10x1000, or later grid sizes without hard-coded coverage rules.
The primary rank is the mean finite/viable loss within a mechanical group, with
viability statistics shown next to it.

This script only reads existing shard logs; it does not rerun experiments.
Use comma-separated tags, or `--tag all`, when a run was continued across
multiple timestamped log sets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from AFM04.stage1pluslight.result_validation import validate_complete_stage1plus_payload

STAGE_ROOT = ROOT / "AFM04" / "stage1pluslight"
DEFAULT_LOG_DIR = STAGE_ROOT / "logs"
DEFAULT_OUT_DIR = STAGE_ROOT / "logs"
DEFAULT_RESULT_PATH = STAGE_ROOT / "results_afm" / "afm_param_stage1pluslight_04.pkl"
DEFAULT_EXPECTED_GROUPS = 0
DEFAULT_EXPECTED_SEEDS = 0

TRIAL_RE = re.compile(
    r"trial\s+(?P<trial_id>\d+)\s+\|\s+"
    r"node_(?P<ks_node>\d+)\*(?P<cs_node>\d+)\s+\|\s+"
    r"seedbank=(?P<seedbank>\d+)\s+initseed=(?P<initseed>\d+)\s+\|\s+"
    r"train=(?P<train>\S+)\s+val=(?P<val>\S+)\s+loss=(?P<loss>\S+)\s+\|\s+"
    r"ks=(?P<ks>\S+)\s+\((?P<ks_err_pct>[^)%]+)%\)\s+"
    r"cs=(?P<cs>\S+)\s+\((?P<cs_err_pct>[^)%]+)%\)\s+\|\s+"
    r"x1_rec=(?P<x1_rec>\S+)%\s+x3_rec=(?P<x3_rec>\S+)%\s+nn=(?P<nn_rec>\S+)%\s+\|\s+"
    r"viable=(?P<viable>True|False)\s+failed=(?P<failed>True|False)"
    r"(?:\s+\|\s+reason=(?P<reason>[^|]+))?"
)


@dataclass(frozen=True)
class TrialRecord:
    trial_id: int
    ks_node: int
    cs_node: int
    seedbank: int
    initseed: int
    train: float
    loss: float
    ks: float
    cs: float
    ks_err_pct: float
    cs_err_pct: float
    x1_rec: float
    nn_rec: float
    viable: bool
    failed: bool
    reason: str
    source_log: str

    @property
    def node_label(self) -> str:
        return f"node_{self.ks_node}*{self.cs_node}"


def _parse_float(token: str) -> float:
    text = token.strip().rstrip(",")
    lower = text.lower()
    if lower in {"nan", "+nan", "-nan"}:
        return float("nan")
    if lower in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if lower in {"-inf", "-infinity"}:
        return float("-inf")
    return float(text)


def _fmt_float(value: float) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.9e}"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def discover_latest_tag(log_dir: Path) -> str:
    tag_re = re.compile(r"afm04_stage1pluslight_local_p\d+of\d+_(?P<tag>\d{8}_\d{6})\.txt$")
    tags: dict[str, float] = {}
    for path in log_dir.glob("afm04_stage1pluslight_local_p*of*_*.txt"):
        match = tag_re.match(path.name)
        if match:
            tag = match.group("tag")
            tags[tag] = max(tags.get(tag, 0.0), path.stat().st_mtime)
    if not tags:
        raise FileNotFoundError(f"No stage1pluslight shard logs found under {log_dir}")
    return sorted(tags.items(), key=lambda item: (item[1], item[0]))[-1][0]


def discover_all_tags(log_dir: Path) -> list[str]:
    tag_re = re.compile(r"afm04_stage1pluslight_local_p\d+of\d+_(?P<tag>\d{8}_\d{6})\.txt$")
    tags: dict[str, float] = {}
    for path in log_dir.glob("afm04_stage1pluslight_local_p*of*_*.txt"):
        match = tag_re.match(path.name)
        if match:
            tag = match.group("tag")
            tags[tag] = max(tags.get(tag, 0.0), path.stat().st_mtime)
    if not tags:
        raise FileNotFoundError(f"No stage1pluslight shard logs found under {log_dir}")
    return [tag for tag, _ in sorted(tags.items(), key=lambda item: (item[1], item[0]))]


def resolve_tags(log_dir: Path, tag_arg: str) -> tuple[list[str], str]:
    tag_text = tag_arg.strip()
    if tag_text == "latest":
        tag = discover_latest_tag(log_dir)
        return [tag], tag
    if tag_text == "all":
        tags = discover_all_tags(log_dir)
        return tags, "all"
    tags = [part.strip() for part in tag_text.split(",") if part.strip()]
    if not tags:
        raise ValueError("--tag must be 'latest', 'all', or a comma-separated tag list")
    return tags, "__".join(tags)


def load_records(log_dir: Path, tags: list[str]) -> tuple[list[TrialRecord], int]:
    records_by_trial_id: dict[int, TrialRecord] = {}
    duplicate_trial_count = 0

    for tag in tags:
        paths = sorted(log_dir.glob(f"afm04_stage1pluslight_local_p*of*_{tag}.txt"))
        if not paths:
            raise FileNotFoundError(f"No stage1pluslight shard logs found for tag {tag!r} under {log_dir}")

        for path in paths:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if " trial " not in line:
                        continue
                    match = TRIAL_RE.search(line)
                    if not match:
                        continue
                    reason = (match.group("reason") or "").strip()
                    rec = TrialRecord(
                        trial_id=int(match.group("trial_id")),
                        ks_node=int(match.group("ks_node")),
                        cs_node=int(match.group("cs_node")),
                        seedbank=int(match.group("seedbank")),
                        initseed=int(match.group("initseed")),
                        train=_parse_float(match.group("train")),
                        loss=_parse_float(match.group("loss")),
                        ks=_parse_float(match.group("ks")),
                        cs=_parse_float(match.group("cs")),
                        ks_err_pct=_parse_float(match.group("ks_err_pct")),
                        cs_err_pct=_parse_float(match.group("cs_err_pct")),
                        x1_rec=_parse_float(match.group("x1_rec")),
                        nn_rec=_parse_float(match.group("nn_rec")),
                        viable=match.group("viable") == "True",
                        failed=match.group("failed") == "True",
                        reason=reason,
                        source_log=path.name,
                    )
                    if rec.trial_id in records_by_trial_id:
                        duplicate_trial_count += 1
                    records_by_trial_id[rec.trial_id] = rec

    records = [records_by_trial_id[key] for key in sorted(records_by_trial_id)]

    if not records:
        raise RuntimeError(f"No trial records parsed from logs for tags {tags!r}")
    return records, duplicate_trial_count


def load_result_payload(result_path: Path) -> dict[str, Any]:
    with result_path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected stage1pluslight result payload type: {type(payload)!r}")
    validate_complete_stage1plus_payload(payload, path=result_path)
    return payload


def load_records_from_payload(payload: dict[str, Any], *, source_log: str) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    for rec in payload["trial_parameters"]:
        params = rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}
        val_parts = rec.get("val_parts", {}) if isinstance(rec.get("val_parts"), dict) else {}
        records.append(
            TrialRecord(
                trial_id=int(params.get("trial_id", 0)),
                ks_node=int(params.get("ks_node_idx", 0)),
                cs_node=int(params.get("cs_node_idx", 0)),
                seedbank=int(params.get("nn_seed_bank_idx", 0)),
                initseed=int(params.get("nn_init_seed", 0)),
                train=float(rec.get("train_loss", float("nan"))),
                loss=float(rec.get("loss", float("inf"))),
                ks=float(rec.get("ks_hat", params.get("ks0", float("nan")))),
                cs=float(rec.get("cs_hat", params.get("cs0", float("nan")))),
                ks_err_pct=float(rec.get("ks_err_pct", float("nan"))),
                cs_err_pct=float(rec.get("cs_err_pct", float("nan"))),
                x1_rec=float(val_parts.get("x1_rec", float("nan"))),
                nn_rec=float(rec.get("val_nn_err", float("nan"))),
                viable=bool(rec.get("is_viable", False)),
                failed=bool(rec.get("trial_failed", False)),
                reason=str(rec.get("failure_reason", rec.get("train_reason", "")) or ""),
                source_log=source_log,
            )
        )
    return records


def load_records_from_result(result_path: Path) -> list[TrialRecord]:
    payload = load_result_payload(result_path)
    return load_records_from_payload(payload, source_log=result_path.name)


def _positive_int(value: Any) -> int:
    try:
        out = int(value)
    except Exception:
        return 0
    return out if out > 0 else 0


def _infer_expected_counts(records: list[TrialRecord], payload: dict[str, Any] | None = None) -> tuple[int, int]:
    expected_groups = 0
    expected_seeds = 0
    expected_total = 0

    if payload is not None:
        ks_nodes = _positive_int(payload.get("stage1plus_grid_ks_nodes"))
        cs_nodes = _positive_int(payload.get("stage1plus_grid_cs_nodes"))
        if ks_nodes > 0 and cs_nodes > 0:
            expected_groups = ks_nodes * cs_nodes
        expected_seeds = _positive_int(payload.get("stage1plus_grid_nn_seeds_per_node"))
        expected_total = _positive_int(payload.get("stage1plus_total_trials")) or _positive_int(payload.get("searches_per_candidate"))

        if expected_groups > 0 and expected_seeds <= 0 and expected_total > 0 and expected_total % expected_groups == 0:
            expected_seeds = expected_total // expected_groups
        if expected_seeds > 0 and expected_groups <= 0 and expected_total > 0 and expected_total % expected_seeds == 0:
            expected_groups = expected_total // expected_seeds

    if expected_groups <= 0:
        expected_groups = len({(rec.ks_node, rec.cs_node) for rec in records})

    if expected_seeds <= 0:
        counts: Counter[tuple[int, int]] = Counter((rec.ks_node, rec.cs_node) for rec in records)
        max_group_count = max(counts.values()) if counts else 0
        max_seedbank = max((rec.seedbank for rec in records), default=0)
        expected_seeds = max(max_group_count, max_seedbank)

    if expected_groups <= 0 or expected_seeds <= 0:
        raise RuntimeError("Could not infer stage1pluslight mechanical grid coverage from result/log records")

    return int(expected_groups), int(expected_seeds)


def summarize_group(records: list[TrialRecord], expected_seeds: int) -> dict[str, Any]:
    first = min(records, key=lambda rec: rec.trial_id)
    viable_records = [rec for rec in records if rec.viable and math.isfinite(rec.loss)]
    finite_records = [rec for rec in records if math.isfinite(rec.loss)]
    losses = [rec.loss for rec in viable_records]
    trains = _finite([rec.train for rec in viable_records])
    x1_recs = _finite([rec.x1_rec for rec in viable_records])
    nn_recs = _finite([rec.nn_rec for rec in viable_records])
    reason_counts = Counter(rec.reason or "none" for rec in records if rec.failed)

    best = min(finite_records, key=lambda rec: rec.loss) if finite_records else None
    n_total = len(records)
    n_viable = len(viable_records)
    n_failed = sum(1 for rec in records if rec.failed)
    n_finite = len(finite_records)

    return {
        "node_label": first.node_label,
        "ks_node_idx": first.ks_node,
        "cs_node_idx": first.cs_node,
        "ks": first.ks,
        "cs": first.cs,
        "ks_err_pct": first.ks_err_pct,
        "cs_err_pct": first.cs_err_pct,
        "n_total": n_total,
        "expected_seeds": expected_seeds,
        "coverage_ok": n_total == expected_seeds,
        "viable_count": n_viable,
        "failed_count": n_failed,
        "finite_count": n_finite,
        "viable_rate": n_viable / n_total if n_total else float("nan"),
        "failed_rate": n_failed / n_total if n_total else float("nan"),
        "mean_loss_viable": mean(losses) if losses else float("inf"),
        "median_loss_viable": median(losses) if losses else float("inf"),
        "std_loss_viable": pstdev(losses) if len(losses) > 1 else 0.0 if losses else float("nan"),
        "p10_loss_viable": _percentile(losses, 0.10),
        "p90_loss_viable": _percentile(losses, 0.90),
        "best_loss": best.loss if best else float("inf"),
        "best_trial_id": best.trial_id if best else 0,
        "best_seedbank": best.seedbank if best else 0,
        "best_initseed": best.initseed if best else 0,
        "mean_train_viable": mean(trains) if trains else float("inf"),
        "mean_x1_rec_viable_pct": mean(x1_recs) if x1_recs else float("nan"),
        "mean_nn_rec_viable_pct": mean(nn_recs) if nn_recs else float("nan"),
        "failure_reasons": dict(sorted(reason_counts.items())),
    }


def build_leaderboard(records: list[TrialRecord], expected_seeds: int) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[TrialRecord]] = defaultdict(list)
    for rec in records:
        groups[(rec.ks_node, rec.cs_node)].append(rec)

    rows = [summarize_group(group_records, expected_seeds) for group_records in groups.values()]
    rows.sort(
        key=lambda row: (
            row["mean_loss_viable"],
            -row["viable_rate"],
            row["best_loss"],
            row["ks_node_idx"],
            row["cs_node_idx"],
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["mech_winner"] = idx
    return rows


def validate_complete(
    rows: list[dict[str, Any]],
    *,
    total_records: int,
    expected_groups: int,
    expected_seeds: int,
) -> None:
    expected_total = expected_groups * expected_seeds
    problems: list[str] = []
    if total_records != expected_total:
        problems.append(f"unique trial records={total_records}, expected={expected_total}")
    if len(rows) != expected_groups:
        problems.append(f"mechanistic groups={len(rows)}, expected={expected_groups}")
    incomplete = [
        f"{row['node_label']}:{row['n_total']}/{expected_seeds}"
        for row in rows
        if int(row["n_total"]) != expected_seeds
    ]
    if incomplete:
        preview = ", ".join(incomplete[:10])
        suffix = "" if len(incomplete) <= 10 else f", ... ({len(incomplete)} groups total)"
        problems.append(f"incomplete groups: {preview}{suffix}")
    if problems:
        raise RuntimeError(
            "Refusing to build mech winner from incomplete stage1pluslight trials. "
            + " | ".join(problems)
            + " | Use --allow-incomplete only for diagnostics, not final ranking."
        )


CSV_FIELDS = [
    "mech_winner",
    "node_label",
    "ks_node_idx",
    "cs_node_idx",
    "ks",
    "cs",
    "ks_err_pct",
    "cs_err_pct",
    "n_total",
    "expected_seeds",
    "coverage_ok",
    "viable_count",
    "failed_count",
    "finite_count",
    "viable_rate",
    "failed_rate",
    "mean_loss_viable",
    "median_loss_viable",
    "std_loss_viable",
    "p10_loss_viable",
    "p90_loss_viable",
    "best_loss",
    "best_trial_id",
    "best_seedbank",
    "best_initseed",
    "mean_train_viable",
    "mean_x1_rec_viable_pct",
    "mean_nn_rec_viable_pct",
    "failure_reasons",
]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["failure_reasons"] = json.dumps(out["failure_reasons"], sort_keys=True)
            writer.writerow(out)


def write_json(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    tag: str,
    source_tags: list[str],
    total_records: int,
    expected_groups: int,
    expected_seeds: int,
    duplicate_trial_count: int,
) -> None:
    payload = {
        "name": "mech winner",
        "stage": "AFM04 stage1pluslight",
        "tag": tag,
        "source_tags": source_tags,
        "ordering_rule": "mean_loss_viable ascending; tie: viable_rate descending, best_loss ascending",
        "total_unique_trial_records": total_records,
        "expected_groups": expected_groups,
        "expected_seeds_per_group": expected_seeds,
        "expected_total_trials": expected_groups * expected_seeds,
        "duplicate_trial_records_removed": duplicate_trial_count,
        "mechanistic_group_count": len(rows),
        "rows": rows,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=True)


def write_text(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    tag: str,
    source_tags: list[str],
    total_records: int,
    expected_groups: int,
    expected_seeds: int,
    duplicate_trial_count: int,
    top: int,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"AFM04 stage1pluslight mech winner leaderboard | average behavior of mech grid | tag={tag}\n")
        fh.write(f"Source tags: {', '.join(source_tags)}\n")
        fh.write("Primary ordering: mean_loss_viable ascending; tie: viable_rate descending, best_loss ascending\n")
        fh.write(f"Total unique parsed trials: {total_records}\n")
        fh.write(f"Expected full coverage: {expected_groups} groups * {expected_seeds} seeds = {expected_groups * expected_seeds}\n")
        fh.write(f"Duplicate trial records removed: {duplicate_trial_count}\n")
        fh.write(f"Mechanistic groups: {len(rows)}\n")
        fh.write(f"Showing top {min(top, len(rows))}\n\n")
        header = (
            "mech_winner  node      ks            ks_err%   cs            cs_err%   viable  failed  "
            "viable_rate  mean_loss      median_loss    best_loss      best_trial  best_seed\n"
        )
        fh.write(header)
        fh.write("-" * (len(header) - 1) + "\n")
        for row in rows[:top]:
            fh.write(
                f"{row['mech_winner']:>11}  "
                f"{row['node_label']:<8}  "
                f"{_fmt_float(row['ks']):>11}  "
                f"{row['ks_err_pct']:>7.2f}  "
                f"{_fmt_float(row['cs']):>11}  "
                f"{row['cs_err_pct']:>7.2f}  "
                f"{row['viable_count']:>6}/{row['n_total']:<3}  "
                f"{row['failed_count']:>6}  "
                f"{row['viable_rate'] * 100:>9.2f}%  "
                f"{_fmt_float(row['mean_loss_viable']):>11}  "
                f"{_fmt_float(row['median_loss_viable']):>11}  "
                f"{_fmt_float(row['best_loss']):>11}  "
                f"{row['best_trial_id']:>10}  "
                f"{row['best_seedbank']:>9}\n"
            )


def build_mechanistic_winner_report(
    *,
    result_path: Path = DEFAULT_RESULT_PATH,
    log_dir: Path = DEFAULT_LOG_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    source: str = "result",
    tag: str = "latest",
    expected_groups: int | None = None,
    expected_seeds: int | None = None,
    allow_incomplete: bool = False,
    top: int | None = None,
) -> dict[str, Any]:
    result_path = result_path.resolve()
    log_dir = log_dir.resolve()
    out_dir = out_dir.resolve()

    payload: dict[str, Any] | None = None
    if source == "result":
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing stage1pluslight merged result: {result_path}")
        payload = load_result_payload(result_path)
        source_tags = [result_path.name]
        resolved_tag = "merged"
        records = load_records_from_payload(payload, source_log=result_path.name)
        duplicate_trial_count = 0
    elif source == "logs":
        source_tags, resolved_tag = resolve_tags(log_dir, tag)
        records, duplicate_trial_count = load_records(log_dir, source_tags)
    else:
        raise ValueError(f"Unsupported stage1pluslight mech winner source: {source!r}")

    inferred_groups, inferred_seeds = _infer_expected_counts(records, payload)
    expected_groups_final = int(expected_groups) if expected_groups and expected_groups > 0 else inferred_groups
    expected_seeds_final = int(expected_seeds) if expected_seeds and expected_seeds > 0 else inferred_seeds

    rows = build_leaderboard(records, expected_seeds=expected_seeds_final)
    if not allow_incomplete:
        validate_complete(
            rows,
            total_records=len(records),
            expected_groups=expected_groups_final,
            expected_seeds=expected_seeds_final,
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"afm04_stage1pluslight_mechanistic_winners_{resolved_tag}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    txt_path = out_dir / f"{stem}.txt"
    top_final = int(top) if top and top > 0 else len(rows)

    write_csv(rows, csv_path)
    write_json(
        rows,
        json_path,
        tag=resolved_tag,
        source_tags=source_tags,
        total_records=len(records),
        expected_groups=expected_groups_final,
        expected_seeds=expected_seeds_final,
        duplicate_trial_count=duplicate_trial_count,
    )
    write_text(
        rows,
        txt_path,
        tag=resolved_tag,
        source_tags=source_tags,
        total_records=len(records),
        expected_groups=expected_groups_final,
        expected_seeds=expected_seeds_final,
        duplicate_trial_count=duplicate_trial_count,
        top=top_final,
    )

    return {
        "tag": resolved_tag,
        "source_tags": source_tags,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "txt_path": str(txt_path),
        "expected_groups": int(expected_groups_final),
        "expected_seeds": int(expected_seeds_final),
        "total_records": int(len(records)),
        "duplicate_trial_count": int(duplicate_trial_count),
        "rows": rows,
        "best": rows[0] if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="latest",
        help="Run tag, comma-separated tags, 'all', or 'latest'. Use multiple tags when a run was resumed.",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--source", choices=("result", "logs"), default="result")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--expected-groups", type=int, default=DEFAULT_EXPECTED_GROUPS, help="Override inferred mechanical group count")
    parser.add_argument("--expected-seeds", type=int, default=DEFAULT_EXPECTED_SEEDS, help="Override inferred NN seed count per mechanical group")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    report = build_mechanistic_winner_report(
        result_path=args.result_path,
        log_dir=args.log_dir,
        out_dir=args.out_dir,
        source=args.source,
        tag=args.tag,
        expected_groups=args.expected_groups,
        expected_seeds=args.expected_seeds,
        allow_incomplete=args.allow_incomplete,
        top=args.top,
    )
    rows = report["rows"]

    print(f"Wrote {report['csv_path']}")
    print(f"Wrote {report['json_path']}")
    print(f"Wrote {report['txt_path']}")
    if rows:
        best = rows[0]
        print(
            "mech winner: "
            f"mech_winner=1 node={best['node_label']} ks={best['ks']:.9e} cs={best['cs']:.9e} "
            f"mean_loss_viable={best['mean_loss_viable']:.9e} "
            f"viable={best['viable_count']}/{best['n_total']} "
            f"best_trial={best['best_trial_id']}"
        )


if __name__ == "__main__":
    main()
