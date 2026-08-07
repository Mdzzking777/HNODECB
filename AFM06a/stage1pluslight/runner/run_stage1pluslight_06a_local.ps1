param(
    [int]$Shards = 5,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

Push-Location $RepoRoot
try {
    $ErrorLevel = "e0.0_real"
    $DataDir = Join-Path $RepoRoot ("AFM06a\datasets\" + $ErrorLevel + "\data")
    $RequiredData = @(
        (Join-Path $DataDir "ode_data_afm_dmt_hard.npz"),
        (Join-Path $DataDir "pert_df_afm_dmt_hard.npz")
    )
    $MissingData = @($RequiredData | Where-Object { -not (Test-Path -LiteralPath $_) })

    if ((-not $CheckOnly) -and $MissingData.Count -gt 0) {
        throw "AFM06a e0.0_real dataset is missing required files: $($MissingData -join ', ')"
    }

    $stale = @(Get-ChildItem Env:HNODECB_AFM06a_STAGE1_* -ErrorAction SilentlyContinue)
    foreach ($item in $stale) {
        Remove-Item -LiteralPath ("Env:\{0}" -f $item.Name) -ErrorAction SilentlyContinue
    }

    if ($CheckOnly) {
        & $Python -m AFM06a.stage1pluslight.main --check-only
    }
    else {
        Write-Host "AFM06a stage1pluslight | formal data -> datasets\$ErrorLevel\data; W1 = 10.000-10.013328 ms."
        Write-Host "KAN input -> all 16 ns true x1 values in W1; internal rollout -> disabled."
        Write-Host "Split -> all selected samples are training points; validation split disabled."
        Write-Host "Sampling -> full-resolution 16 ns W1 grid; no weighted resampling."
        Write-Host "Normalizer -> W1 training_mean_std input; Force output -> training_mean_std inverse to physical Fts [N]."
        Write-Host "Loss -> force-scale-normalized r_dyn + residual smoothness (lambda_sm=1)."
        Write-Host "Physics -> omega0/drive settings loaded from the selected dataset manifest."
        $env:HNODECB_AFM06a_ERROR_LEVEL = "$ErrorLevel"
        $env:HNODECB_AFM06a_STAGE1_NN_SEEDS = "2000"
        $env:HNODECB_AFM06a_STAGE1_SHARD_COUNT = "$Shards"
        $env:HNODECB_AFM06a_STAGE1_ALL_TRAIN = "1"
        $env:HNODECB_AFM06a_STAGE1_WINDOW_START_S = "0.010000000"
        $env:HNODECB_AFM06a_STAGE1_WINDOW_STOP_S = "0.010013328"
        Remove-Item Env:HNODECB_AFM06a_STAGE1_SAMPLE_COUNT -ErrorAction SilentlyContinue
        $env:HNODECB_AFM06a_STAGE1_SAMPLE_STRIDE = "1"
        $env:HNODECB_AFM06a_STAGE1_KAN_GRID = "11"
        $env:HNODECB_AFM06a_STAGE1_KAN_GRID_LO = "-1"
        $env:HNODECB_AFM06a_STAGE1_KAN_GRID_HI = "1"
        $env:HNODECB_AFM06a_STAGE1_NORMALIZER_POLICY = "training_mean_std"
        $env:HNODECB_AFM06a_STAGE1_FORCE_OUTPUT_POLICY = "training_mean_std_inverse"
        $env:HNODECB_AFM06a_STAGE1_GAIN = "0"
        $env:HNODECB_AFM06a_STAGE1_GAIN_LEARNABLE = "0"
        $env:HNODECB_AFM06a_STAGE1_GAIN_POLICY = "disabled"
        $env:HNODECB_AFM06a_STAGE1_SAMPLING_POLICY = "full_resolution_all_train"
        $env:HNODECB_AFM06a_STAGE1_TRANSITION_SAMPLING_WEIGHT = "1"
        $env:HNODECB_AFM06a_STAGE1_CONTACT_SAMPLING_WEIGHT = "1"
        $env:HNODECB_AFM06a_STAGE1_NONCONTACT_SAMPLING_WEIGHT = "1"
        $env:HNODECB_AFM06a_STAGE1_TRANSITION_HALF_WIDTH_S = "1.3565187713310766e-07"
        $env:HNODECB_AFM06a_STAGE1_SOFT_MASK = "0"
        $env:HNODECB_AFM06a_STAGE1_SOFT_MASK_TRAINABLE = "0"
        $env:HNODECB_AFM06a_STAGE1_SOFT_MASK_POLICY = "disabled"
        $env:HNODECB_AFM06a_STAGE1_LOSS_POLICY = "force_scaled_dynamics_residual_true_x1_residual_smooth"
        $env:HNODECB_AFM06a_STAGE1_SMOOTHNESS_LOSS_WEIGHT = "1.0"
        $launcher = @"
from AFM06a.stage1pluslight.config import default_config
from AFM06a.stage1pluslight.runner import launch_local_shards

launch_local_shards(default_config(), shard_count=$Shards)
"@
        $launcher | & $Python -
    }
    if ($LASTEXITCODE -ne 0) {
        throw "AFM06a stage1pluslight exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
