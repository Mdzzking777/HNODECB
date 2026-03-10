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
$env:HNODECB_INF_LOG = "1"
$env:HNODECB_LOG_NN_ERR = "1"
$env:HNODECB_JULIA_PROJECT = "$projectPath"
$env:JULIA_PROJECT = "$projectPath"
# Ensure autosplit spawns shards (no fixed index in parent process)
Remove-Item Env:HNODECB_STAGE1_SHARD_INDEX -ErrorAction SilentlyContinue

if (-not (Test-Path -Path "logs")) {
  New-Item -ItemType Directory -Path "logs" | Out-Null
}

Write-Host "Running pipeline03 with $Processes shards, JULIA_NUM_THREADS=$env:JULIA_NUM_THREADS, PREFLIGHT=$env:HNODECB_STAGE1_PREFLIGHT"
Write-Host "Project      -> $projectPath"
Write-Host "Stage1 logs -> logs/log2_03_step2a_stage1_p#.txt"

& julia --project=$projectPath .\runner\afm_param_search_pipeline_03.jl
