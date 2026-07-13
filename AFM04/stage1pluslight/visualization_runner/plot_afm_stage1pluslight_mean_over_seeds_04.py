"""AFM04 stage1pluslight mean-over-seeds mechanistic loss surface.

This draws the AFM04 counterpart of the AFM05 mean-over-seeds plot:
one point per mechanistic grid location, where the z value is
log10(mean(train_loss over finite viable NN seeds)).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_helper(repo_root: Path):
    helper_path = (
        repo_root
        / "AFM05"
        / "stage1pluslight"
        / "runner"
        / "visualization"
        / "plot_afm_stage1pluslight_seed_surfaces_05.py"
    )
    spec = importlib.util.spec_from_file_location("afm05_seed_surface_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    script_dir = Path(__file__).resolve().parent
    afm04_root = script_dir.parents[1]
    repo_root = script_dir.parents[2]

    default_result = afm04_root / "stage1pluslight" / "results_afm" / "afm_param_stage1pluslight_04.pkl"
    default_out_dir = afm04_root / "stage1pluslight" / "visualization"
    result_path = Path(argv[0]).resolve() if argv else default_result.resolve()
    out_dir = Path(argv[1]).resolve() if len(argv) > 1 else default_out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root))
    from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import CS, KS

    helper = load_helper(repo_root)
    records = helper.viable_records(helper.load_records(result_path))
    ks_values, cs_values = helper.grid_values(records)
    points = helper.mean_seed_surface(records)

    out_path = out_dir / "afm_param_stage1pluslight_04_mean_over_seeds_ks_logcs_train_loss_3d.html"
    helper.write_surface_html(
        out_path=out_path,
        result_path=result_path,
        title="AFM04 st1pl mean-over-seeds mechanistic loss surface",
        points=points,
        ks_values=ks_values,
        cs_values=cs_values,
        seed_label="mean over finite viable NN seeds at this mech point",
        highlight_lowest_n=30,
        highlight_name="lowest 30 mean-loss points",
        gt_ks=float(KS),
        gt_cs=float(CS),
    )
    print(f"saved {out_path} ({len(points)} grid points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
