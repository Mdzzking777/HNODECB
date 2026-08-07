param(
  [int]$Rank = 1,
  [switch]$CheckOnly,
  [switch]$Smoke,
  [switch]$BackwardSmoke
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
while (-not (Test-Path -LiteralPath (Join-Path $repoRoot "AFM06a\stage2light"))) {
  $parent = Split-Path -Path $repoRoot -Parent
  if ([string]::IsNullOrEmpty($parent) -or $parent -eq $repoRoot) {
    throw "Could not locate HNODECB repository root from $PSScriptRoot"
  }
  $repoRoot = $parent
}

Set-Location -LiteralPath $repoRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
  $pythonExe = "python"
}

$stale = @(Get-ChildItem Env:HNODECB_AFM06a_STAGE2LIGHT_* -ErrorAction SilentlyContinue)
foreach ($item in $stale) {
  Remove-Item -LiteralPath ("Env:\{0}" -f $item.Name) -ErrorAction SilentlyContinue
}
$env:HNODECB_AFM06a_STAGE2LIGHT_INPUT_RANK = "$Rank"
$env:HNODECB_AFM06a_STAGE2LIGHT_RESUME = "1"
$env:HNODECB_AFM06a_STAGE2LIGHT_ADAPTIVE_GRID = "1"
$env:HNODECB_AFM06a_STAGE2LIGHT_AGU_EVERY_ADAM_EPOCH = "1"
$env:HNODECB_AFM06a_STAGE2LIGHT_GAIN = "0"
$env:HNODECB_AFM06a_STAGE2LIGHT_GAIN_LEARNABLE = "0"
$env:HNODECB_AFM06a_STAGE2LIGHT_SOFT_MASK = "0"
$env:HNODECB_AFM06a_STAGE2LIGHT_SOFT_MASK_TRAINABLE = "0"
$env:HNODECB_AFM06a_STAGE2LIGHT_LBFGS_SPARSIFICATION = "0"
$env:HNODECB_AFM06a_STAGE2LIGHT_LBFGS_SPARSIFICATION_LAMBDA = "1e-7"
$env:HNODECB_AFM06a_STAGE2LIGHT_LBFGS_SPARSIFICATION_ENTROPY_WEIGHT = "2.0"
$env:HNODECB_AFM06a_STAGE2LIGHT_PRUNE_ON_FIRST_ZERO_STEP = "0"
$env:HNODECB_AFM06a_STAGE2LIGHT_PRUNING_MAX_VAL_LOSS_INCREASE_FRACTION = "0.05"
$env:HNODECB_AFM06a_STAGE2LIGHT_PRUNING_MIN_HIDDEN_NODES = "1"
$env:HNODECB_AFM06a_STAGE2LIGHT_LBFGS_ZERO_STEP_MAX_EVENTS = "3"

$arguments = @("-m", "AFM06a.stage2light.main", "--run", "--rank", "$Rank")
if ($BackwardSmoke.IsPresent) {
  $arguments[2] = "--backward-smoke"
} elseif ($Smoke.IsPresent) {
  $arguments[2] = "--smoke"
} elseif ($CheckOnly.IsPresent) {
  $arguments[2] = "--check-only"
}

Write-Host "AFM06a stage2light | rank=$Rank | direct complete st1pl warmstart"
Write-Host "Epoch-0 preparation is disabled by design."
Write-Host "Dataset/window/omega0 -> inherited from the selected st1pl endpoint."
Write-Host "Force head -> inherited exactly from st1pl; raw head is mapped by training_mean_std inverse to physical Fts [N]."
Write-Host "KAN -> single-input 1-3-1; normalized x1 is the only KAN input."
Write-Host "Internal rollout -> disabled in st1pl and stage2light; explicit rollout scripts remain separate diagnostics."
Write-Host "Gain -> disabled in st1pl and stage2light."
Write-Host "Sampling -> inherited full-resolution 16 ns W1 grid; no weighted resampling."
Write-Host "Split -> inherited all-training samples; validation split disabled."
Write-Host "Loss -> force-scale-normalized r_dyn + residual smoothness inherited from st1pl."
Write-Host "Soft Mask -> disabled in st1pl and stage2light."
Write-Host "Optimizer -> 10 Adam epochs + 1990 L-BFGS epochs; AGU updates only in Adam epochs."
Write-Host "L-BFGS sparsification -> disabled."
Write-Host "Automatic pruning -> disabled."
Write-Host "Hard zero-step policy -> events 1-2: history reset; event 3: early stop."
Write-Host "Mode -> $($arguments[2])"
Write-Host "Python -> $pythonExe"
& $pythonExe @arguments
if ($LASTEXITCODE -ne 0) {
  throw "AFM06a stage2light command failed with exit code $LASTEXITCODE"
}
