param(
  [int]$Threads = 6,
  [int]$Processes = 3,
  [int]$Preflight = 0
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
$env:HNODECB_STAGE1_SHARD_COUNT = "$Processes"
$env:HNODECB_STAGE1_PREFLIGHT = "$Preflight"
$env:HNODECB_STAGE1_LOG_EVERY = "1"
$env:HNODECB_LR_ADAPT = "1"
$env:HNODECB_INF_LOG = "1"
$env:HNODECB_LOG_NN_ERR = "1"
$env:HNODECB_STAGE1_AUTOSPAWN = "1"
$env:HNODECB_STAGE1_RUN_LABEL = "Stage1"
$env:HNODECB_STAGE1_RESULT_STEM = "afm_param_stage1_03"
$env:HNODECB_STAGE1_LOG_SUBDIR = "stage1_step2a\\local"
$env:HNODECB_STAGE1PLUSLIGHT_LOG_SUBDIR = "stage1_step2a\\local"
$env:HNODECB_STAGE1PLUSLIGHT_LOG_PREFIX = "log2_03_step2a_stage1_archscreen"
$env:HNODECB_STAGE1_ARCH_SHARD_LOG_SUBDIR = "as_per_shard"
$env:HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_COUNT = "$Processes"
$env:HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH = "10"
$env:HNODECB_STAGE1PLUS_ARCH_EPOCHS = "10"
$env:HNODECB_STAGE1PLUS_ARCH_LR = "1e-2"
$env:HNODECB_STAGE1PLUS_ARCH_LR_ADAPT = "1"
$env:HNODECB_STAGE1PLUS_ARCH_LR_UP_ONLY = "1"
$env:HNODECB_STAGE1PLUS_ARCH_LR_UP_ETA = "5.0"
$env:HNODECB_STAGE1PLUS_ARCH_LR_UP_TRIGGER_RATIO = "1.10"
$env:HNODECB_STAGE1PLUS_ARCH_LR_UP_MIN_FACTOR = "1.10"
$env:HNODECB_STAGE1PLUS_ARCH_LR_UP_MAX_FACTOR = "10.0"
$env:HNODECB_STAGE1PLUS_ARCH_LR_TARGET_MULT = "1.0"
$env:HNODECB_STAGE1PLUS_ARCH_DYNAMIC_GNN = "1"
$env:HNODECB_STAGE1PLUS_ARCH_FIXED_GNN = "0"
$env:HNODECB_STAGE1PLUS_ARCH_FIXED_GNN_Q = "0.95"
$env:HNODECB_STAGE1PLUS_ARCH_FIXED_GNN_MIN = "1e-12"
$env:HNODECB_STAGE1PLUS_ARCH_FIXED_GNN_MAX = "1e12"
$env:HNODECB_STAGE1PLUS_ARCH_GNN_LOG10_UPDATE_MIN = "1.0"
$env:HNODECB_STAGE1PLUS_ARCH_STEP_MAX_LOSS_FRAC = "0.01"
$env:HNODECB_STAGE1PLUS_ARCH_PARETO_PLOT = "1"
$env:HNODECB_STAGE1_ARCH_RANKING_SUBDIR = "ranking"
$env:HNODECB_JULIA_PROJECT = "$projectPath"
$env:JULIA_PROJECT = "$projectPath"
Remove-Item Env:HNODECB_STAGE1_SHARD_INDEX -ErrorAction SilentlyContinue

if (-not (Test-Path -Path (Join-Path $repoRoot "logs"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs") | Out-Null
}

$stage1LogDir = Join-Path $repoRoot "logs\\stage1_step2a\\local"
if (-not (Test-Path -Path $stage1LogDir)) {
  New-Item -ItemType Directory -Path $stage1LogDir -Force | Out-Null
}

$stage1DriverDir = Join-Path $stage1LogDir "driver"
if (-not (Test-Path -Path $stage1DriverDir)) {
  New-Item -ItemType Directory -Path $stage1DriverDir -Force | Out-Null
}

$stage1Log = Join-Path $stage1DriverDir "log2_03_step2a_stage1_archscreen_driver.txt"
$stage1Runner = Join-Path $repoRoot "runner\\stagewise\\afm_param_search_stage1_03.jl"

Write-Host "Running Stage1 (03) archi screening with $Processes shard(s), JULIA_NUM_THREADS=$env:JULIA_NUM_THREADS, PREFLIGHT=$env:HNODECB_STAGE1_PREFLIGHT, ARCH_TRIALS_PER_ARCH=$env:HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH, ARCH_EPOCHS=$env:HNODECB_STAGE1PLUS_ARCH_EPOCHS"
Write-Host "Repo root    -> $repoRoot"
Write-Host "Project     -> $projectPath"
Write-Host "Driver log  -> $stage1Log"
Write-Host "Shard logs  -> logs/stage1_step2a/local/as_per_shard/log2_03_step2a_stage1_archscreen_p#.txt"
Write-Host "Ranking TXT -> logs/stage1_step2a/local/ranking/"
Write-Host "Viz output  -> logs/stage1_step2a/local/visualization/"

$cmdLine = 'julia --project="' + $projectPath + '" "' + $stage1Runner + '" 2>&1'
cmd.exe /d /c $cmdLine | Tee-Object -FilePath $stage1Log
if ($LASTEXITCODE -ne 0) {
  throw "Stage1 Julia process failed with exit code $LASTEXITCODE. See $stage1Log"
}
