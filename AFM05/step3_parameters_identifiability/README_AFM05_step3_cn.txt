AFM05 Step 3: Local Identifiability Analysis
=============================================

Purpose
-------

AFM05 Step 3 analyzes the local practical identifiability of one completed
stage2light candidate at its FINAL epoch. It does not train the model and does
not select a best epoch.

The default observable vector contains only:

  x1, x2, x2dot

AFM05 has no observed x3 or Fts. The archived x3_compat value is used only as
the third component of the rollout initial condition. Predicted x3 and Fts may
be used to verify that the final st2l rollout was reconstructed, but they are
never added to the observable sensitivity vector and are never treated as
ground truth.

Entry point
-----------

The current candidate is selected from:

  AFM05/Archive/st2l/valley

Only this payload is accepted:

  result/stage2light_result_p1.pt

The local point is reconstructed from final_state_dict and
final_mech_state_dict. The exact stride-sampled time grid and initial state are
read from final_snapshot.full. This is essential because window_meta records
the raw index span (6273 points), whereas the actual AFM05 st2l rollout uses
the stride-sampled grid (currently 393 points).

Baseline reproduction gate
--------------------------

Before any finite difference is evaluated, Step 3 rebuilds the final st2l
model and rollout. It compares the reconstructed x1, x2, x2dot, predicted x3,
and predicted Fts trajectories against final_snapshot. The computation stops
if any relative RMS mismatch exceeds the configured tolerance (default 1e-6).

This gate prevents Step 3 from silently analyzing a local point that differs
from the archived st2l endpoint.

Mathematical computation
------------------------

The formal parameter set is:

  active numerical KAN parameters
  log_gnn
  trainable soft-mask parameters
  raw_ks and raw_cs

Inactive symbolic parameters and the initial state are excluded.

For every scalar parameter, central finite differences are evaluated in the
same relative raw-parameter coordinates used by AFM04 Step 3. The normalized
observable sensitivity matrix S is assembled, followed by:

  H_GN = S^T S / N

The eigensystem of H_GN is then used for near-null-space projection and the
groupwise decomposition among KAN, gain, soft mask, and mechanistic parameters.

Main scripts
------------

  set_afm05_step3_entry.py
  compute_afm05_step3_identifiability.py
  post_analyze_afm05_step3_identifiability.py
  visualize_afm05_step3_post_analysis.py

Main runners
------------

Select/list available completed st2l candidates:

  & .venv/Scripts/python.exe `
    AFM05/step3_parameters_identifiability/set_afm05_step3_entry.py `
    --archive-root AFM05/Archive/st2l/valley `
    --list

Run identifiability for the default B01/rank148 candidate:

  & AFM05/step3_parameters_identifiability/runner/run_step3_identifiability_05_local.ps1

Run the complete compute, post-analysis, and visualization pipeline:

  & AFM05/step3_parameters_identifiability/runner/run_step3_full_pipeline_05_local.ps1

Smoke test only (one scalar parameter, one process):

  & AFM05/step3_parameters_identifiability/runner/run_step3_identifiability_05_local.ps1 `
    -RankDir "AFM05/Archive/st2l/valley/B01_rank148" `
    -ParameterLimit 1 `
    -ShardCount 1

Outputs
-------

  results/       sensitivity/Hessian/eigensystem and post-analysis data
  logs/          runner and shard logs
  visualization/ Step 3 figures

The archived stage2light payloads are read-only inputs. Step 3 does not modify
AFM05 stage1pluslight, prestage2, stage2light, or their archives.
