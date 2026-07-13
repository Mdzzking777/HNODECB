param(
  [int]$Threads = 5,
  [int]$Candidate = 1,
  [int]$Shards = 4,
  [int]$ArchScreenShards = 3,
  [int]$TrialsPerShard = 500,
  [int]$FinalTopK = 0,
  [int]$KsNodes = 50,
  [int]$CsNodes = 50,
  [int]$NnSeedsPerNode = 50
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
$env:HNODECB_STAGE1_INPUT_BASENAME = "afm_param_stage1_03.jld"
$env:HNODECB_STAGE1PLUS_MODE = "1"
$env:HNODECB_STAGE1PLUSLIGHT_MODE = "1"
$env:HNODECB_STAGE1PLUS_WINDOW_MODE = "stage2_w123"
$env:HNODECB_STAGE1PLUS_RESUME = "1"
$env:HNODECB_STAGE1PLUS_CHECKPOINT_EVERY = "100"
$env:HNODECB_STAGE1PLUS_INHERIT_STAGE1_ARCH = "0"
$env:HNODECB_STAGE1PLUS_MANUAL_LAYERS = "1"
$env:HNODECB_STAGE1PLUS_MANUAL_NODES = "4"
$totalTrials = $KsNodes * $CsNodes * $NnSeedsPerNode
$approxTrialsPerShard = [int][Math]::Ceiling($totalTrials / [double][Math]::Max($Shards, 1))
$env:HNODECB_STAGE1PLUS_GRID_KS_NODES = "$KsNodes"
$env:HNODECB_STAGE1PLUS_GRID_CS_NODES = "$CsNodes"
$env:HNODECB_STAGE1PLUS_GRID_NN_SEEDS_PER_NODE = "$NnSeedsPerNode"
$env:HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE = "$totalTrials"
$env:HNODECB_STAGE1PLUS_FINAL_TOPK = "$FinalTopK"
$env:HNODECB_STAGE1PLUS_ZERO_NN = "0"
$env:HNODECB_STAGE1_SHARD_COUNT = "$Shards"
$env:HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_COUNT = "$ArchScreenShards"
$env:HNODECB_STAGE1_AUTOSPAWN = "1"
$env:HNODECB_STAGE1PLUSLIGHT_LOG_SUBDIR = "stage1_step2a/1pluslight/local"
$env:HNODECB_STAGE1PLUSLIGHT_LOG_PREFIX = "log2_03_step2a_stage1pluslight_local"
$env:HNODECB_STAGE1_LOG_EVERY = "1"
$env:HNODECB_INF_LOG = "1"
$env:HNODECB_LOG_NN_ERR = "1"
$env:HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH = "10"
$env:HNODECB_STAGE1PLUS_ARCH_EPOCHS = "10"
$env:HNODECB_STAGE1PLUS_ARCH_DYNAMIC_GNN = "0"
$env:HNODECB_STAGE1PLUS_ARCH_FIXED_GNN = "0"
$env:HNODECB_STAGE1PLUS_ARCH_LR_ADAPT = "1"
$env:HNODECB_STAGE1PLUS_ARCH_LR_MIN = "1e-30"
$env:HNODECB_STAGE1PLUS_ARCH_LR_MAX = "5e1"
$env:HNODECB_STAGE1PLUS_ARCH_LR_ETA = "0.5"
$env:HNODECB_STAGE1PLUS_ARCH_LR_EMA = "0.85"
$env:HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_P_NET = "1.0"
$env:HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_MECH = "1.0"
$env:HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_P_NET_MIN = "0.1"
$env:HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_P_NET_MAX = "1.0"
$env:HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_MECH_MIN = "0.5"
$env:HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_MECH_MAX = "1.5"
$env:HNODECB_STAGE1PLUS_ARCH_GROUP_ADAPT = "0"
$env:HNODECB_STAGE1PLUS_ARCH_GROUP_ETA = "0.05"
$env:HNODECB_STAGE1PLUS_ARCH_STEP_GUARD = "1"
$env:HNODECB_STAGE1PLUS_ARCH_STEP_RETRIES = "6"
$env:HNODECB_STAGE1PLUS_ARCH_STEP_RETRY_LR_FACTOR = "0.1"
$env:HNODECB_STAGE1PLUS_ARCH_STEP_MAX_LOSS_FRAC = "0.01"
$env:HNODECB_STAGE1PLUS_TRIAL_OPT_ENABLE = "0"
$env:HNODECB_STAGE1PLUS_TRIAL_EPOCHS = "0"
$env:HNODECB_STAGE1PLUS_TRIAL_DYNAMIC_GNN = "1"
$env:HNODECB_STAGE1PLUS_TRIAL_FIXED_GNN = "0"
$env:HNODECB_STAGE1PLUS_TRIAL_GNN_LOG10_UPDATE_MIN = "1.0"
$env:HNODECB_STAGE1PLUS_TRIAL_LR = "1e-3"
$env:HNODECB_STAGE1PLUS_TRIAL_LR_ADAPT = "1"
$env:HNODECB_STAGE1PLUS_TRIAL_LR_UP_ONLY = "1"
$env:HNODECB_STAGE1PLUS_TRIAL_LR_MIN = "1e-30"
$env:HNODECB_STAGE1PLUS_TRIAL_LR_MAX = "5e1"
$env:HNODECB_STAGE1PLUS_TRIAL_LR_ETA = "0.5"
$env:HNODECB_STAGE1PLUS_TRIAL_LR_UP_ETA = "1.0"
$env:HNODECB_STAGE1PLUS_TRIAL_LR_UP_TRIGGER_RATIO = "1.10"
$env:HNODECB_STAGE1PLUS_TRIAL_LR_UP_MIN_FACTOR = "1.10"
$env:HNODECB_STAGE1PLUS_TRIAL_LR_UP_MAX_FACTOR = "2.0"
$env:HNODECB_STAGE1PLUS_TRIAL_LR_EMA = "0.85"
$env:HNODECB_STAGE1PLUS_TRIAL_LR_TARGET_MULT = "3.0"
$env:HNODECB_STAGE1PLUS_TRIAL_GRAD_SCALE_P_NET = "1.0"
$env:HNODECB_STAGE1PLUS_TRIAL_GRAD_SCALE_MECH = "1.0"
$env:HNODECB_STAGE1PLUS_TRIAL_STEP_GUARD = "1"
$env:HNODECB_STAGE1PLUS_TRIAL_STEP_RETRIES = "6"
$env:HNODECB_STAGE1PLUS_TRIAL_STEP_RETRY_LR_FACTOR = "0.1"
$env:HNODECB_STAGE1PLUS_TRIAL_STEP_MAX_LOSS_FRAC = "0.01"
$env:HNODECB_STAGE1PLUS_TRIAL_EARLY_STOP_EPOCHS = "5"
$env:HNODECB_STAGE1PLUS_TRIAL_EARLY_STOP_MIN_DROP_FRAC = "0.01"
if (-not $env:HNODECB_STAGE1_RUN_SEED -or [string]::IsNullOrWhiteSpace($env:HNODECB_STAGE1_RUN_SEED)) {
  $env:HNODECB_STAGE1_RUN_SEED = "314159265"
}
$env:HNODECB_JULIA_PROJECT = "$projectPath"
$env:JULIA_PROJECT = "$projectPath"

Remove-Item Env:HNODECB_STAGE1_SHARD_INDEX -ErrorAction SilentlyContinue

if (-not (Test-Path -Path (Join-Path $repoRoot "logs"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs") | Out-Null
}
if (-not (Test-Path -Path (Join-Path $repoRoot "logs\\stage1_step2a\\1pluslight\\local"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs\\stage1_step2a\\1pluslight\\local") -Force | Out-Null
}
if (-not (Test-Path -Path (Join-Path $repoRoot "logs\\stage1_step2a\\1pluslight\\local\\archi_screen_shard_p"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs\\stage1_step2a\\1pluslight\\local\\archi_screen_shard_p") -Force | Out-Null
}
if (-not (Test-Path -Path (Join-Path $repoRoot "logs\\stage1_step2a\\1pluslight\\local\\weight_search_shard_p"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs\\stage1_step2a\\1pluslight\\local\\weight_search_shard_p") -Force | Out-Null
}
if (-not (Test-Path -Path (Join-Path $repoRoot "logs\\stage1_step2a\\1pluslight\\local\\visualization"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs\\stage1_step2a\\1pluslight\\local\\visualization") -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$driverLog = Join-Path $repoRoot ("logs\\stage1_step2a\\1pluslight\\local\\log2_03_step2a_stage1pluslight_local_driver_" + $timestamp + ".txt")
$runner = Join-Path $repoRoot "runner\\stagewise\\afm_param_search_stage1pluslight_03.jl"

Write-Host "Running Stage1pluslight (03) with JULIA_NUM_THREADS=$env:JULIA_NUM_THREADS, STANDALONE=ON, SHARDS=$env:HNODECB_STAGE1_SHARD_COUNT, GRID=${KsNodes}x${CsNodes}, NN_SEEDS_PER_NODE=$NnSeedsPerNode, APPROX_TRIALS_PER_SHARD=$approxTrialsPerShard, TOTAL_TRIALS=$env:HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE, FINAL_TOPK=$env:HNODECB_STAGE1PLUS_FINAL_TOPK, TRIAL_OPT=$env:HNODECB_STAGE1PLUS_TRIAL_OPT_ENABLE"
Write-Host "NN arch mode-> inherit_st1=$env:HNODECB_STAGE1PLUS_INHERIT_STAGE1_ARCH, manual_layers=$env:HNODECB_STAGE1PLUS_MANUAL_LAYERS, manual_nodes=$env:HNODECB_STAGE1PLUS_MANUAL_NODES"
Write-Host "Window mode -> $env:HNODECB_STAGE1PLUS_WINDOW_MODE"
Write-Host "Resume     -> $env:HNODECB_STAGE1PLUS_RESUME (checkpoint_every=$env:HNODECB_STAGE1PLUS_CHECKPOINT_EVERY)"
Write-Host "Note       -> -Candidate is ignored in standalone mode (received $Candidate)"
Write-Host "Note       -> -TrialsPerShard is ignored in node-grid mode (received $TrialsPerShard)"
Write-Host "Repo root  -> $repoRoot"
Write-Host "Project    -> $projectPath"
Write-Host "Run seed   -> $env:HNODECB_STAGE1_RUN_SEED"
Write-Host "Stage1 in  -> $env:HNODECB_STAGE1_INPUT_BASENAME"
Write-Host "Driver log -> $driverLog"
Write-Host "Weight logs-> logs\\stage1_step2a\\1pluslight\\local\\weight_search_shard_p\\log2_03_step2a_stage1pluslight_local[_resumeNN]_p#.txt"
Write-Host "Viz output -> logs\\stage1_step2a\\1pluslight\\local\\visualization\\"

$prevErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& julia --project=$projectPath $runner *>&1 | Tee-Object -FilePath $driverLog
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $prevErrorAction

if ($exitCode -ne 0) {
  throw "Stage1pluslight local run failed with exit code $exitCode. See log: $driverLog"
}
