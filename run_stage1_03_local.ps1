param(
  [int]$Threads = 6,
  [int]$Processes = 1,
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
$env:HNODECB_LR_ADAPT = "0"
$env:HNODECB_INF_LOG = "1"
$env:HNODECB_LOG_NN_ERR = "1"
$env:HNODECB_STAGE1_AUTOSPAWN = "1"
$env:HNODECB_STAGE1_LOG_SUBDIR = "stage1_step2a\\local"
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

$stage1Log = Join-Path $stage1LogDir "log2_03_step2a_stage1_driver.txt"
$stage1Runner = Join-Path $repoRoot "runner\\stagewise\\afm_param_search_stage1_03.jl"

Write-Host "Running Stage1 (03) with $Processes shard(s), JULIA_NUM_THREADS=$env:JULIA_NUM_THREADS, PREFLIGHT=$env:HNODECB_STAGE1_PREFLIGHT"
Write-Host "Repo root    -> $repoRoot"
Write-Host "Project     -> $projectPath"
Write-Host "Driver log  -> $stage1Log"
Write-Host "Shard logs  -> logs/stage1_step2a/local/log2_03_step2a_stage1_p#.txt"

& julia --project=$projectPath $stage1Runner 2>&1 | Tee-Object -FilePath $stage1Log
