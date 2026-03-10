param(
  [int]$Threads = 16,
  [int]$SelfTest = 0,
  [int]$Shards = 1,
  [int]$BlasThreads = 1,
  [int]$Candidate = 6,
  [int]$InitRetries = 200,
  [int]$NNRank = 1
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

$env:HNODECB_SELFTEST = "$SelfTest"
$env:HNODECB_LR_ADAPT = "1"
$env:HNODECB_LR_MIN = "1e-30"
$env:HNODECB_LR_MAX = "2e-3"
$env:HNODECB_LR_ETA = "0.35"
$env:HNODECB_LR_EMA = "0.90"
$env:HNODECB_LR_TARGET = "0.4"
$env:HNODECB_LR_AGGRESSIVE_KEEP = "1"
$env:HNODECB_LR_AGGRESSIVE_UP = "1.08"
$env:HNODECB_STAGE2_LR_FORCE = "3e-4"
$env:HNODECB_STAGE2_OPTIMIZER = "amsgrad"
$env:HNODECB_STAGE2_GRAD_SCALE_P_NET = "0.2"
$env:HNODECB_STAGE2_GRAD_SCALE_MECH = "1.0"
$env:HNODECB_STAGE2_GROUP_ADAPT = "1"
$env:HNODECB_STAGE2_GROUP_ETA = "0.15"

$env:HNODECB_ODE_MAXITERS = "5000000"
$env:HNODECB_STAGE2_EPOCH_RETRIES = "12"
$env:HNODECB_STAGE2_RETRY_LR_FACTOR = "0.2"
$env:HNODECB_STAGE2_RETRY_LR_FLOOR = "1e-30"
$env:HNODECB_STAGE2_STEP_GUARD = "1"
$env:HNODECB_STAGE2_STEP_RETRIES = "6"
$env:HNODECB_STAGE2_STEP_RETRY_LR_FACTOR = "0.1"
$env:HNODECB_STAGE2_STEP_MAX_LOSS_INCREASE = "1e-6"
$env:HNODECB_STAGE2_PLATEAU_EARLY_STOP = "0"
$env:HNODECB_STAGE2_VERIFY_ADAM_DIR = "1"

$env:HNODECB_STAGE2_LOG_EVERY = "1"
$env:HNODECB_STAGE2_AUTOSPAWN = "1"
$env:HNODECB_STAGE2_SHARD_COUNT = "$Shards"
$env:HNODECB_STAGE2_INPUT_TOPK = "9"
$env:HNODECB_STAGE2LIGHT_CANDIDATE = "$Candidate"
$env:HNODECB_STAGE2_CANDIDATE_INDICES = "$Candidate"
$env:HNODECB_STAGE2_FINAL_TOPK = "1"
$env:HNODECB_STAGE2_RESULT_BASENAME = "afm_param_stage2light_03.jld"
$env:HNODECB_STAGE2_X3R_WEIGHT = "0.0"
$env:HNODECB_STAGE2_SCRIPT = "afm_param_stage2light_03.jl"
$env:HNODECB_STAGE2_INIT_RETRIES = "$InitRetries"
$env:HNODECB_STAGE2_NN_WARM_ENABLED = "1"
$env:HNODECB_STAGE2_NN_WARM_INPUT_BASENAME = "afm_param_stage1pluslight_03.jld"
$env:HNODECB_STAGE2_NN_WARM_RANK = "$NNRank"

$env:HNODECB_JULIA_PROJECT = "$projectPath"
$env:JULIA_PROJECT = "$projectPath"

$env:HNODECB_STAGE2_LOG_SUBDIR = "stage2_step2a/local"
$env:HNODECB_STAGE2_LOG_PREFIX = "log2_03_step2a_stage2light_local"

Remove-Item Env:HNODECB_STAGE2_SHARD_INDEX -ErrorAction SilentlyContinue

if (-not (Test-Path -Path (Join-Path $repoRoot "logs"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs") | Out-Null
}
if (-not (Test-Path -Path (Join-Path $repoRoot "logs\\stage2_step2a\\local"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs\\stage2_step2a\\local") -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stage2Log = Join-Path $repoRoot ("logs\\stage2_step2a\\local\\log2_03_step2a_stage2light_local_driver_" + $timestamp + ".txt")
$stage2Runner = Join-Path $repoRoot "runner\\stagewise\\afm_param_search_stage2light_03.jl"

Write-Host "Running Stage2light (03) with JULIA_NUM_THREADS=$env:JULIA_NUM_THREADS, BLAS_THREADS=$env:OPENBLAS_NUM_THREADS, SELFTEST=$env:HNODECB_SELFTEST, SHARDS=$env:HNODECB_STAGE2_SHARD_COUNT, CANDIDATE=$env:HNODECB_STAGE2LIGHT_CANDIDATE, NN_RANK=$env:HNODECB_STAGE2_NN_WARM_RANK, INIT_RETRIES=$env:HNODECB_STAGE2_INIT_RETRIES, OPT=$env:HNODECB_STAGE2_OPTIMIZER, G_SCALE=$env:HNODECB_STAGE2_GRAD_SCALE_P_NET/$env:HNODECB_STAGE2_GRAD_SCALE_MECH, GROUP_ADAPT=$env:HNODECB_STAGE2_GROUP_ADAPT, GROUP_ETA=$env:HNODECB_STAGE2_GROUP_ETA, LR_AGGR_KEEP=$env:HNODECB_LR_AGGRESSIVE_KEEP, LR_AGGR_UP=$env:HNODECB_LR_AGGRESSIVE_UP"
Write-Host "Repo root  -> $repoRoot"
Write-Host "Project    -> $projectPath"
Write-Host "Driver log -> $stage2Log"
Write-Host "Shard logs -> logs\\stage2_step2a\\local\\log2_03_step2a_stage2light_local_p#.txt"

$prevErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& julia --project=$projectPath $stage2Runner *>&1 | Tee-Object -FilePath $stage2Log
$stage2ExitCode = $LASTEXITCODE
$ErrorActionPreference = $prevErrorAction

if ($stage2ExitCode -ne 0) {
  throw "Stage2light local run failed with exit code $stage2ExitCode. See log: $stage2Log"
}
