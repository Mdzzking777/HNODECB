AFM04 step3_parameters_identifiability
=====================================

Purpose
-------
This folder is the AFM04-specific step3 implementation.  It is intentionally
separate from the repository-root `step3_parameters_identifiability`, because
the original HNODECB examples are Julia `.jld` workflows, while AFM04 st2l
archives are Python/PyTorch `.pt` payloads.

Current One-Command Runner
--------------------------
Run:

  powershell -ExecutionPolicy Bypass -File ^
    AFM04/step3_parameters_identifiability/runner/run_step3_identifiability_04_local.ps1

Default object:

  rank247 / candidate B11

Default behavior:

  1. Select and validate one completed st2l archive object.
  2. Write `results/current_step3_entry.json`.
  3. In formal `trained` mode, compute sensitivity parameter columns with
     the default `-ShardCount 8`.
  4. Merge shard sensitivity columns.
  5. Run the final Hessian/FIM and eigenvalue/eigenvector calculation.

The runner writes logs to:

  AFM04/step3_parameters_identifiability/logs/

and writes results to:

  AFM04/step3_parameters_identifiability/results/

Post-Analysis Runner
--------------------
After a formal step3 result exists, run:

  powershell -ExecutionPolicy Bypass -File ^
    AFM04/step3_parameters_identifiability/runner/run_step3_post_analysis_04_local.ps1

Default input:

  AFM04/step3_parameters_identifiability/results/afm04_step3_identifiability_trained_latest.json

This runner is post-process only. It does not rerun st2l rollouts and does not
mutate the original eigen/Hessian result.

It writes post-analysis tables to:

  AFM04/step3_parameters_identifiability/results/post_analysis/

and figures to:

  AFM04/step3_parameters_identifiability/visualization/

Main post-analysis outputs:

  - eigen spectrum plot with the AFM04 null threshold
  - null threshold sweep summary
  - null-space projection of mech.raw_ks and mech.raw_cs
  - group decomposition of projection energy:

      KAN numeric / gain / soft-mask / mech.raw_ks / mech.raw_cs

The latest aliases are:

  results/post_analysis/afm04_step3_post_analysis_latest.csv
  results/post_analysis/afm04_step3_post_analysis_latest.json
  results/post_analysis/afm04_step3_post_analysis_latest.txt
  visualization/afm04_step3_post_analysis_latest_eigen_spectrum.png
  visualization/afm04_step3_post_analysis_latest_null_projection_groups.png
  visualization/afm04_step3_post_analysis_latest_threshold_sweep.png

Use a specific historical step3 result:

  powershell -ExecutionPolicy Bypass -File ^
    AFM04/step3_parameters_identifiability/runner/run_step3_post_analysis_04_local.ps1 ^
    -InputResult "AFM04/step3_parameters_identifiability/results/afm04_step3_identifiability_trained_rank247_B11_20260609_163751.json"

Use another threshold sweep:

  -Thresholds "1e-12,1e-11,1e-10,1e-9,1e-8"

Individual Visualization Runners
--------------------------------
The three post-analysis figures also have one-command runners under:

  AFM04/step3_parameters_identifiability/runner/visualization runner/

Eigen spectrum:

  powershell -ExecutionPolicy Bypass -File ^
    "AFM04/step3_parameters_identifiability/runner/visualization runner/run_step3_eigen_spectrum_04_local.ps1"

Null-space projection stacked bar:

  powershell -ExecutionPolicy Bypass -File ^
    "AFM04/step3_parameters_identifiability/runner/visualization runner/run_step3_null_projection_groups_04_local.ps1"

Threshold sweep:

  powershell -ExecutionPolicy Bypass -File ^
    "AFM04/step3_parameters_identifiability/runner/visualization runner/run_step3_threshold_sweep_04_local.ps1"

All three figures:

  powershell -ExecutionPolicy Bypass -File ^
    "AFM04/step3_parameters_identifiability/runner/visualization runner/run_step3_all_post_analysis_visualizations_04_local.ps1"

These visualization runners are timestamped and pass `--no-latest`, so they
do not create duplicate `latest` image aliases.

Implemented Identifiability Calculation
---------------------------------------
AFM04 step3 formal defaults:

  1. Do not include u0 in the parameter vector.
  2. Use x1, x2, and x2dot as the default observable outputs.

Rationale:

  - In the original HNODECB examples, u0 is included because the initial state
    can be part of the local optimum object. In AFM04 st2l, the W0 initial
    condition is supplied by the window data and is not a trained variable, so
    it should not be part of the default identifiability parameter vector.
  - x1 and x2 are observed state channels, and x2dot is the derivative-like
    observable constraint used by AFM04 training. x3 is hidden/oracle-only,
    and force/Fts is model-implied rather than a default direct observation.

The formal default parameter set is:

  trained

It contains the trainable KAN numeric parameters, gain, soft-mask parameters,
and mechanistic raw_ks/raw_cs parameters. It excludes inactive symbolic
parameters and excludes u0.

The fast diagnostic parameter set is:

  theta = [log(ks), log(cs)]

The diagnostic mode is useful for quickly checking the local ks/cs subspace,
but it is not the faithful default analogue of the original step3 workflow.

Both modes use central finite differences, evaluate the stage2light rollout
around the completed st2l local point, build a normalized sensitivity matrix S
over the observable components, form:

  H = S' S / N

and eigendecomposes H.

Default observable components:

  x1, x2, x2dot

`x3` and `force` are intentionally not included by default.

Main output files:

  results/afm04_step3_identifiability_<parameter_set>_<rank_label>_<timestamp>.csv
  results/afm04_step3_identifiability_<parameter_set>_<rank_label>_<timestamp>.json
  results/afm04_step3_identifiability_<parameter_set>_<rank_label>_<timestamp>.txt
  results/afm04_step3_identifiability_<parameter_set>_<rank_label>_<timestamp>.npz

Latest aliases are also refreshed:

  results/afm04_step3_identifiability_<parameter_set>_latest.csv
  results/afm04_step3_identifiability_<parameter_set>_latest.json
  results/afm04_step3_identifiability_<parameter_set>_latest.txt
  results/afm04_step3_identifiability_<parameter_set>_latest.npz

Progress Logging
----------------
The runner transcript captures timestamped progress from the Python step3
compute script.

For formal `trained` mode, the log prints:

  - baseline rollout start/done
  - each scalar parameter index, e.g. `param 37/453`
  - current parameter name
  - sensitivity matrix assembly start/done
  - Hessian build start/done
  - eigendecomposition start/done
  - eigen row summary start/done

Example:

  [2026-06-09 15:40:58] param 1/453 | parameter=model.log_gnn[0]
  [2026-06-09 15:41:05] param 2/453 | parameter=model.soft_mask_s0_raw[0]
  [2026-06-09 16:28:12] sensitivity matrix assembly start
  [2026-06-09 16:28:13] hessian build start
  [2026-06-09 16:28:13] eigendecomposition start

This makes the step3 log usable like the st2l logs for seeing the current
stage during a long full-parameter run. Hessian/eigen is split into the
largest meaningful NumPy-level stages; the eigendecomposition itself is a
single library call, so it cannot provide inner per-eigenpair progress.

Parameter-Column Sharding
-------------------------
Formal `trained` step3 supports parameter-column sharding.

The sensitivity matrix has shape:

  S = N_observations x N_parameters

For AFM04 rank247/B11, this is approximately:

  S ~= 1182 x 453

Each sensitivity column is independent:

  S[:, i] = d observable / d theta_i

Therefore the only stage that should be sharded is the finite-difference
parameter-column construction. With the default `-ShardCount 8`, the runner
splits the 453 parameter columns into 8 contiguous column ranges and computes
these shard outputs separately:

  results/shards/<run_id>/shard_001_of_008.npz
  results/shards/<run_id>/shard_002_of_008.npz
  ...
  results/shards/<run_id>/shard_008_of_008.npz

Important:

  - Each shard computes only its assigned sensitivity columns.
  - Shards do not compute final Hessian/FIM/eigen results.
  - The merge stage concatenates all shard columns back into the original
    parameter order.
  - Only after merge do we compute:

      H = S' S / N_observations

    and then run eigendecomposition.

This is necessary because H contains cross-shard blocks such as S1' S2, so
eigenvalues/eigenvectors from individual shards cannot be merged directly.

Sharding is disabled for the fast `mech` diagnostic mode because it only has
two parameters.

Null-Space Threshold And Perturbation Convention
------------------------------------------------
For AFM04, `abs(eigenvalue)` should not be interpreted with a universal
constant copied from the original examples. The threshold depends on the
observable normalization, the finite-difference epsilon, and the numerical
scale of the local Hessian/FIM:

  H = S' S / N_observations

AFM04 convention:

  1. Use the same threshold rule for all ranks/candidates being compared.
  2. Define the primary null-space threshold as:

       tau_null = 1e-8

     A direction is treated as null/near-null when:

       abs(eigenvalue) <= tau_null

  3. Always report a threshold sweep as a stability check:

       1e-12, 1e-10, 1e-8

     If the qualitative conclusion changes sharply across this sweep, the
     rank/candidate should be described as threshold-sensitive rather than
     assigned a single hard conclusion.

For the current rank247/B11 trained result:

  max(abs(eigenvalue)) ~= 6.65e-3
  tau_null = 1e-8

The current null counts are approximately:

  abs(lambda) <= 1e-12 : 368 / 453
  abs(lambda) <= 1e-10 : 409 / 453
  abs(lambda) <= 1e-8  : 435 / 453

Therefore `1e-8` is the formal AFM04 default from this point onward, while
`1e-12` and `1e-10` are used to check robustness. The original examples often use a
larger absolute threshold such as `1e-5`, but applying that directly to the
current AFM04 spectrum would classify almost the entire 453-dimensional space
as near-null, so it is too broad for the formal AFM04 default.

For qualitative perturbation tests, do not only perturb `ks`.

AFM04 should run the same null-space compensation check for both mechanistic
parameters:

  - perturb ks alone, then perturb ks together with its null-space projection
  - perturb cs alone, then perturb cs together with its null-space projection

The point is to identify which trainable groups compensate each mechanistic
parameter direction. For example, if the `ks` unit direction has a large
projection into the null space and that projection is mostly KAN numeric
parameters, the precise statement is:

  the local `ks` perturbation can be compensated mainly by KAN numeric
  parameters under the default observables x1, x2, x2dot.

The same statement must be checked independently for `cs`.

Switch Entry Object
-------------------
Use candidate B number:

  powershell -ExecutionPolicy Bypass -File ^
    AFM04/step3_parameters_identifiability/runner/run_step3_identifiability_04_local.ps1 ^
    -Candidate B13

Use original rank:

  powershell -ExecutionPolicy Bypass -File ^
    AFM04/step3_parameters_identifiability/runner/run_step3_identifiability_04_local.ps1 ^
    -Rank 141

Use explicit rank directory:

  powershell -ExecutionPolicy Bypass -File ^
    AFM04/step3_parameters_identifiability/runner/run_step3_identifiability_04_local.ps1 ^
    -RankDir "AFM04/archive/st2l/3_10x10x200_W0 new/rank247_B11"

List available completed st2l entries:

  powershell -ExecutionPolicy Bypass -File ^
    AFM04/step3_parameters_identifiability/runner/run_step3_identifiability_04_local.ps1 ^
    -List

Useful Options
--------------
Only select/validate entry, without running eigen computation:

  -SkipIdentifiability

Use another central finite-difference epsilon:

  -FiniteDiffEps 1e-5

Use another observable component set:

  -Components "x1,x2,x2dot,force"

Use the fast ks/cs-only diagnostic mode:

  -ParameterSet mech

Run a tiny formal-parameter smoke test without overwriting the full trained
latest aliases:

  -ParameterSet trained -ParameterLimit 3

`-ParameterLimit` is only for debugging/verification. Full formal step3 uses
`-ParameterSet trained` with `-ParameterLimit 0`.

Entry Payload Kind
------------------
`-EntryPayloadKind result` analyzes the final st2l local point.

`-EntryPayloadKind best` analyzes the best validation checkpoint.

`-EntryPayloadKind checkpoint` analyzes the resumable checkpoint.

No st2l rerun is required for manifest generation or for the current
identifiability calculation, as long as the archive contains a complete st2l
result payload and copied data files.
