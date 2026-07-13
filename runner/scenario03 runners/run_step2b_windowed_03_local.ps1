param(
  [int]$Threads = 20,
  [int]$BlasThreads = 1
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
while ($true) {
  $hasProject = Test-Path -Path (Join-Path $repoRoot "Project.toml")
  $hasRunner = Test-Path -Path (Join-Path $repoRoot "runner")
  if ($hasProject -and $hasRunner) {
    break
  }

  $parent = Split-Path -Path $repoRoot -Parent
  if ([string]::IsNullOrEmpty($parent) -or $parent -eq $repoRoot) {
    throw "Could not locate repository root from $PSScriptRoot"
  }
  $repoRoot = $parent
}

Set-Location -Path $repoRoot

$projectPath = Join-Path $repoRoot "HPC_env"
if (-not (Test-Path -Path (Join-Path $projectPath "Project.toml"))) {
  throw "Missing HPC_env Project.toml at $projectPath"
}

$env:JULIA_NUM_THREADS = "$Threads"
$env:OPENBLAS_NUM_THREADS = "$BlasThreads"
$env:MKL_NUM_THREADS = "$BlasThreads"
$env:OMP_NUM_THREADS = "$BlasThreads"
$env:VECLIB_MAXIMUM_THREADS = "$BlasThreads"

$env:HNODECB_JULIA_PROJECT = "$projectPath"
$env:JULIA_PROJECT = "$projectPath"

$env:HNODECB_STEP2B_STAGE2_INPUT_BASENAME = "afm_param_stage2light_windowed_03.jld"
$env:HNODECB_STEP2B_RESULT_DIR = "res_afm_03_windowed"
$env:HNODECB_STEP2B_RESULT_NAME = "afm_03_windowed.jld"
$env:HNODECB_STEP2B_LOG_SUBDIR = "step2b_windowed/print"
$env:HNODECB_STEP2B_SHARD_LOG_SUBDIR = "step2b_windowed/print"
$env:HNODECB_STEP2B_LOG_PREFIX = "log2_03_step2b_windowed_local"
$env:HNODECB_STEP2B_USE_L2 = "0"
$env:HNODECB_STAGE2_PLATEAU_EARLY_STOP = "0"
$env:HNODECB_STEP2B_GOOD_ENOUGH_LOSS = "0"
$env:HNODECB_STEP2B_MAX_LBFGS_ITERS = "100"
$env:HNODECB_STEP2B_RESUME = "1"
$env:HNODECB_STEP2B_CHECKPOINT_EVERY = "5"
$env:HNODECB_STEP2B_DYNAMIC_GNN = "1"
$env:HNODECB_STEP2B_FIXED_GNN = "0"
$env:HNODECB_STEP2B_GNN_Q = "0.95"
$env:HNODECB_STEP2B_GNN_MIN = "1e-12"
$env:HNODECB_STEP2B_GNN_MAX = "1e12"
$env:HNODECB_STEP2B_GNN_LOG10_UPDATE_MIN = "1.0"

if (-not (Test-Path -Path (Join-Path $repoRoot "logs"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs") | Out-Null
}
if (-not (Test-Path -Path (Join-Path $repoRoot "logs\\step2b_windowed\\print"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs\\step2b_windowed\\print") -Force | Out-Null
}
if (-not (Test-Path -Path (Join-Path $repoRoot "logs\\step2b_windowed\\visualization"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs\\step2b_windowed\\visualization") -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$driverLog = Join-Path $repoRoot ("logs\\step2b_windowed\\print\\log2_03_step2b_windowed_local_driver_" + $timestamp + ".txt")
$runner = Join-Path $repoRoot "runner\\stagewise\\afm_train_step2b_windowed_03.jl"

Write-Host "Running Step2b WINDOWED (03) with JULIA_NUM_THREADS=$env:JULIA_NUM_THREADS, BLAS_THREADS=$env:OPENBLAS_NUM_THREADS"
Write-Host "Repo root  -> $repoRoot"
Write-Host "Project    -> $projectPath"
Write-Host "Driver log -> $driverLog"
Write-Host "Shard logs -> logs\\step2b_windowed\\print\\log2_03_step2b_windowed_local[_resumeNN]_p#.txt"
Write-Host "LBFGS      -> maxiters=$env:HNODECB_STEP2B_MAX_LBFGS_ITERS"
Write-Host "EarlyStop  -> plateau=$env:HNODECB_STAGE2_PLATEAU_EARLY_STOP, good_enough=$env:HNODECB_STEP2B_GOOD_ENOUGH_LOSS"
Write-Host "Resume     -> checkpoint every $env:HNODECB_STEP2B_CHECKPOINT_EVERY epoch(s), resume=$env:HNODECB_STEP2B_RESUME"
Write-Host "Figures    -> logs\\step2b_windowed\\visualization"

$prevErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& julia --project=$projectPath $runner *>&1 | Tee-Object -FilePath $driverLog
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $prevErrorAction

if ($exitCode -ne 0) {
  throw "Step2b windowed local run failed with exit code $exitCode. See log: $driverLog"
}
