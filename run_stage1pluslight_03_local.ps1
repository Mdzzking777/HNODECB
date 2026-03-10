param(
  [int]$Threads = 8,
  [int]$Candidate = 6,
  [int]$Shards = 3,
  [int]$TrialsPerShard = 333,
  [int]$FinalTopK = 10
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
$env:HNODECB_STAGE1_VARIANT = "stage1pluslight"
$env:HNODECB_STAGE1_RESULT_STEM = "afm_param_stage1pluslight_03"
$env:HNODECB_STAGE1PLUS_MODE = "1"
$env:HNODECB_STAGE1PLUS_INPUT_BASENAME = "afm_param_stage1_03.jld"
$env:HNODECB_STAGE1PLUS_INPUT_TOPK = "9"
$env:HNODECB_STAGE1PLUSLIGHT_CANDIDATE = "$Candidate"
$env:HNODECB_STAGE1PLUS_BASE_INDICES = "$Candidate"
$totalTrials = $Shards * $TrialsPerShard
$env:HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE = "$totalTrials"
$env:HNODECB_STAGE1PLUS_FINAL_TOPK = "$FinalTopK"
$env:HNODECB_STAGE1PLUS_ZERO_NN = "0"
$env:HNODECB_STAGE1_SHARD_COUNT = "$Shards"
$env:HNODECB_STAGE1_AUTOSPAWN = "1"
$env:HNODECB_STAGE1PLUSLIGHT_LOG_SUBDIR = "stage1_step2a/local"
$env:HNODECB_STAGE1PLUSLIGHT_LOG_PREFIX = "log2_03_step2a_stage1pluslight_local"
$env:HNODECB_STAGE1_LOG_EVERY = "1"
$env:HNODECB_INF_LOG = "1"
$env:HNODECB_LOG_NN_ERR = "1"
$env:HNODECB_JULIA_PROJECT = "$projectPath"
$env:JULIA_PROJECT = "$projectPath"

Remove-Item Env:HNODECB_STAGE1_SHARD_INDEX -ErrorAction SilentlyContinue

if (-not (Test-Path -Path (Join-Path $repoRoot "logs"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs") | Out-Null
}
if (-not (Test-Path -Path (Join-Path $repoRoot "logs\\stage1_step2a\\local"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs\\stage1_step2a\\local") -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$driverLog = Join-Path $repoRoot ("logs\\stage1_step2a\\local\\log2_03_step2a_stage1pluslight_local_driver_" + $timestamp + ".txt")
$runner = Join-Path $repoRoot "runner\\stagewise\\afm_param_search_stage1pluslight_03.jl"

Write-Host "Running Stage1pluslight (03) with JULIA_NUM_THREADS=$env:JULIA_NUM_THREADS, CANDIDATE=$env:HNODECB_STAGE1PLUSLIGHT_CANDIDATE, SHARDS=$env:HNODECB_STAGE1_SHARD_COUNT, TRIALS_PER_SHARD=$TrialsPerShard, TOTAL_TRIALS=$env:HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE, FINAL_TOPK=$env:HNODECB_STAGE1PLUS_FINAL_TOPK"
Write-Host "Repo root  -> $repoRoot"
Write-Host "Project    -> $projectPath"
Write-Host "Driver log -> $driverLog"
Write-Host "Shard logs -> logs\\stage1_step2a\\local\\log2_03_step2a_stage1pluslight_local_p#.txt"

$prevErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& julia --project=$projectPath $runner *>&1 | Tee-Object -FilePath $driverLog
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $prevErrorAction

if ($exitCode -ne 0) {
  throw "Stage1pluslight local run failed with exit code $exitCode. See log: $driverLog"
}
