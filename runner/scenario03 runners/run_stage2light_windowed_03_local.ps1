param(
  [int]$Threads = 20,
  [int]$SelfTest = 0,
  [int]$Shards = 3,
  [int]$BlasThreads = 1,
  [int]$Candidate = 1,
  [int]$InitRetries = 200,
  [int]$NNRank = 1,
  [int]$AutoWindows = 1,
  [int]$WindowStart = 1,
  [int]$WindowLen = 399,
  [string]$RunTag = ""
)

$ErrorActionPreference = "Stop"

function Get-SafeRunTag {
  param(
    [string]$Value
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return ""
  }

  $safe = ($Value -replace '[^A-Za-z0-9._-]+', '_').Trim('_')
  if ([string]::IsNullOrWhiteSpace($safe)) {
    throw "RunTag '$Value' does not contain any usable filename characters."
  }
  return $safe
}

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

$resolvedCandidate = $Candidate
if ($resolvedCandidate -lt 1) {
  $resolvedCandidate = 1
}
$resolvedInputTopK = [Math]::Max(10, $resolvedCandidate)
$safeRunTag = Get-SafeRunTag -Value $RunTag

$env:JULIA_NUM_THREADS = "$Threads"
$env:OPENBLAS_NUM_THREADS = "$BlasThreads"
$env:MKL_NUM_THREADS = "$BlasThreads"
$env:OMP_NUM_THREADS = "$BlasThreads"
$env:VECLIB_MAXIMUM_THREADS = "$BlasThreads"

$env:HNODECB_SELFTEST = "$SelfTest"
$env:HNODECB_LR_ADAPT = "1"
$env:HNODECB_LR_MIN = "1e-30"
$env:HNODECB_LR_MAX = "2e2"
$env:HNODECB_LR_ETA = "0.5"
$env:HNODECB_LR_EMA = "0.85"
$env:HNODECB_LR_TARGET = "0.4"
$env:HNODECB_LR_AGGRESSIVE_KEEP = "1"
$env:HNODECB_LR_AGGRESSIVE_UP = "1.08"
$env:HNODECB_STAGE2_LR_FORCE = "3e-4"
$env:HNODECB_STAGE2_OPTIMIZER = "amsgrad"
$env:HNODECB_STAGE2_GRAD_SCALE_P_NET = "1.0"
$env:HNODECB_STAGE2_GRAD_SCALE_MECH = "1.0"
$env:HNODECB_STAGE2_GRAD_SCALE_P_NET_MIN = "0.1"
$env:HNODECB_STAGE2_GRAD_SCALE_P_NET_MAX = "1.0"
$env:HNODECB_STAGE2_GRAD_SCALE_MECH_MIN = "0.5"
$env:HNODECB_STAGE2_GRAD_SCALE_MECH_MAX = "1.5"
$env:HNODECB_STAGE2_GROUP_ADAPT = "0"
$env:HNODECB_STAGE2_GROUP_ETA = "0.05"

$env:HNODECB_ODE_MAXITERS = "5000000"
$env:HNODECB_STAGE2_EPOCH_RETRIES = "12"
$env:HNODECB_STAGE2_RETRY_LR_FACTOR = "0.2"
$env:HNODECB_STAGE2_RETRY_LR_FLOOR = "1e-30"
$env:HNODECB_STAGE2_STEP_GUARD = "1"
$env:HNODECB_STAGE2_STEP_RETRIES = "6"
$env:HNODECB_STAGE2_STEP_RETRY_LR_FACTOR = "0.1"
$env:HNODECB_STAGE2_STEP_MAX_LOSS_INCREASE_FRAC = "0.01"
$env:HNODECB_STAGE2_PLATEAU_EARLY_STOP = "0"
$env:HNODECB_STAGE2_VERIFY_ADAM_DIR = "1"
$env:HNODECB_STAGE2_VAL_LOG_EVERY = "5"
$env:HNODECB_STAGE2_VAL_LOG_OFFSET = "1"
$env:HNODECB_STAGE2_LOG_EVERY = "1"
$env:HNODECB_STAGE2_MAXITERS = "200"
$env:HNODECB_STAGE2_RESUME = "1"
$env:HNODECB_STAGE2_CHECKPOINT_EVERY = "5"
$env:HNODECB_STAGE2_AUTOSPAWN = "1"
$env:HNODECB_STAGE2_SHARD_COUNT = "$Shards"
$env:HNODECB_STAGE2_WINDOW_AUTOMANIFEST = "$AutoWindows"
$env:HNODECB_STAGE2_INPUT_BASENAME = "afm_param_stage1pluslight_03.jld"
$env:HNODECB_STAGE2_INPUT_TOPK = "$resolvedInputTopK"
$env:HNODECB_STAGE2LIGHT_CANDIDATE = "$resolvedCandidate"
$env:HNODECB_STAGE2_CANDIDATE_INDICES = "$resolvedCandidate"
$env:HNODECB_STAGE2_FINAL_TOPK = "1"
$resultBasename = if ($safeRunTag -ne "") { "afm_param_stage2light_windowed_03_$safeRunTag.jld" } else { "afm_param_stage2light_windowed_03.jld" }
$env:HNODECB_STAGE2_RESULT_BASENAME = $resultBasename
$env:HNODECB_STAGE2_X3R_WEIGHT = "0.0"
$env:HNODECB_STAGE2_DYNAMIC_GNN = "1"
$env:HNODECB_STAGE2_FIXED_GNN = "0"
$env:HNODECB_STAGE2_GNN_Q = "0.95"
$env:HNODECB_STAGE2_GNN_MIN = "1e-12"
$env:HNODECB_STAGE2_GNN_MAX = "1e12"
$env:HNODECB_STAGE2_GNN_LOG10_UPDATE_MIN = "1.0"
$env:HNODECB_STAGE2_SCRIPT = "stage2light_windowed_run.jl"
$env:HNODECB_STAGE2_INIT_RETRIES = "$InitRetries"
$env:HNODECB_STAGE2_NN_WARM_ENABLED = "0"
$env:HNODECB_STAGE2_NN_WARM_INPUT_BASENAME = "afm_param_stage1pluslight_03.jld"
$env:HNODECB_STAGE2_NN_WARM_RANK = "$NNRank"
if ($AutoWindows -eq 0) {
  $env:HNODECB_STAGE2_WINDOW_START = "$WindowStart"
  $env:HNODECB_STAGE2_WINDOW_LEN = "$WindowLen"
} else {
  Remove-Item Env:HNODECB_STAGE2_WINDOW_START -ErrorAction SilentlyContinue
  Remove-Item Env:HNODECB_STAGE2_WINDOW_LEN -ErrorAction SilentlyContinue
}

$env:HNODECB_JULIA_PROJECT = "$projectPath"
$env:JULIA_PROJECT = "$projectPath"

$logSubdir = if ($safeRunTag -ne "") { "stage2_step2a/local/windowed/$safeRunTag" } else { "stage2_step2a/local/windowed" }
$shardLogSubdir = if ($safeRunTag -ne "") { "stage2_step2a/local/windowed/$safeRunTag/window_per_shard" } else { "stage2_step2a/local/windowed/window_per_shard" }
$logPrefix = if ($safeRunTag -ne "") { "log2_03_step2a_stage2light_windowed_local_$safeRunTag" } else { "log2_03_step2a_stage2light_windowed_local" }
$env:HNODECB_STAGE2_LOG_SUBDIR = $logSubdir
$env:HNODECB_STAGE2_SHARD_LOG_SUBDIR = $shardLogSubdir
$env:HNODECB_STAGE2_LOG_PREFIX = $logPrefix

Remove-Item Env:HNODECB_STAGE2_SHARD_INDEX -ErrorAction SilentlyContinue

if (-not (Test-Path -Path (Join-Path $repoRoot "logs"))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot "logs") | Out-Null
}
if (-not (Test-Path -Path (Join-Path $repoRoot ("logs\\" + ($logSubdir -replace '/', '\\'))))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot ("logs\\" + ($logSubdir -replace '/', '\\'))) -Force | Out-Null
}
if (-not (Test-Path -Path (Join-Path $repoRoot ("logs\\" + ($shardLogSubdir -replace '/', '\\'))))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot ("logs\\" + ($shardLogSubdir -replace '/', '\\'))) -Force | Out-Null
}
$visualizationSubdir = $logSubdir + "/Visualization"
$resultMirrorSubdir = $logSubdir + "/Results"
if (-not (Test-Path -Path (Join-Path $repoRoot ("logs\\" + ($visualizationSubdir -replace '/', '\\'))))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot ("logs\\" + ($visualizationSubdir -replace '/', '\\'))) -Force | Out-Null
}
if (-not (Test-Path -Path (Join-Path $repoRoot ("logs\\" + ($resultMirrorSubdir -replace '/', '\\'))))) {
  New-Item -ItemType Directory -Path (Join-Path $repoRoot ("logs\\" + ($resultMirrorSubdir -replace '/', '\\'))) -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$driverLogRel = if ($safeRunTag -ne "") {
  "logs\" + ($logSubdir -replace '/', '\') + "\" + $logPrefix + "_driver_" + $timestamp + ".txt"
} else {
  "logs\\stage2_step2a\\local\\windowed\\log2_03_step2a_stage2light_windowed_local_driver_" + $timestamp + ".txt"
}
$stage2Log = Join-Path $repoRoot $driverLogRel
$stage2Runner = Join-Path $repoRoot "runner\\stagewise\\afm_param_search_stage2light_03.jl"

if ($AutoWindows -eq 0) {
  Write-Host "Running Stage2light WINDOWED (03) with JULIA_NUM_THREADS=$env:JULIA_NUM_THREADS, BLAS_THREADS=$env:OPENBLAS_NUM_THREADS, SELFTEST=$env:HNODECB_SELFTEST, SHARDS=$env:HNODECB_STAGE2_SHARD_COUNT, CANDIDATE_RANK=$env:HNODECB_STAGE2LIGHT_CANDIDATE, INIT_RETRIES=$env:HNODECB_STAGE2_INIT_RETRIES, MAXITERS=$env:HNODECB_STAGE2_MAXITERS, AUTO_WINDOWS=OFF, WINDOW_START=$env:HNODECB_STAGE2_WINDOW_START, WINDOW_LEN=$env:HNODECB_STAGE2_WINDOW_LEN"
} else {
  Write-Host "Running Stage2light WINDOWED (03) with JULIA_NUM_THREADS=$env:JULIA_NUM_THREADS, BLAS_THREADS=$env:OPENBLAS_NUM_THREADS, SELFTEST=$env:HNODECB_SELFTEST, REQUESTED_SHARDS=$env:HNODECB_STAGE2_SHARD_COUNT, CANDIDATE_RANK=$env:HNODECB_STAGE2LIGHT_CANDIDATE, INIT_RETRIES=$env:HNODECB_STAGE2_INIT_RETRIES, MAXITERS=$env:HNODECB_STAGE2_MAXITERS, AUTO_WINDOWS=ON (first-contact / max-x1-pp-change / tail-stable)"
}
if ($Candidate -lt 1) {
  Write-Host "Candidate source -> auto from standalone stage1pluslight best rank = $resolvedCandidate"
} else {
  Write-Host "Candidate source -> explicit argument = $resolvedCandidate"
}
Write-Host "Repo root  -> $repoRoot"
Write-Host "Project    -> $projectPath"
if ($safeRunTag -ne "") {
  Write-Host "Run tag    -> $safeRunTag"
}
Write-Host "Driver log -> $stage2Log"
Write-Host "Shard logs -> logs\\$($shardLogSubdir -replace '/', '\\')\\$logPrefix[_resumeNN]_p#.txt"
Write-Host "Result     -> step2a_hyperparameter_tuning\\hyperparameter_tuning_second_stage\\results_afm\\$resultBasename"
Write-Host "Resume     -> checkpoint every $env:HNODECB_STAGE2_CHECKPOINT_EVERY epoch(s), resume=$env:HNODECB_STAGE2_RESUME"
Write-Host "Mirror     -> logs\\$($resultMirrorSubdir -replace '/', '\\')"
Write-Host "Visualize  -> logs\\$($visualizationSubdir -replace '/', '\\')"
Write-Host "Stage1TopK -> $resolvedInputTopK"
Write-Host "Note       -> -NNRank is ignored; Stage2 uses the selected Stage1pluslight candidate's own p_net"

if ($safeRunTag -ne "") {
  $runRootDir = Join-Path $repoRoot ("logs\\" + ($logSubdir -replace '/', '\\'))
  $runInfoPath = Join-Path $runRootDir "run_info.txt"
  @(
    "RunTag=$safeRunTag"
    "CandidateRank=$resolvedCandidate"
    "Stage1TopK=$resolvedInputTopK"
    "ResultBasename=$resultBasename"
    "DriverLog=$stage2Log"
    "ShardLogSubdir=$shardLogSubdir"
    "VisualizationSubdir=$visualizationSubdir"
    "ResultMirrorSubdir=$resultMirrorSubdir"
    "AutoWindows=$AutoWindows"
    "Threads=$Threads"
    "WrittenAt=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
  ) | Set-Content -Path $runInfoPath -Encoding UTF8
}

$prevErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& julia --project=$projectPath $stage2Runner *>&1 | Tee-Object -FilePath $stage2Log
$stage2ExitCode = $LASTEXITCODE
$ErrorActionPreference = $prevErrorAction

if ($stage2ExitCode -ne 0) {
  throw "Stage2light windowed local run failed with exit code $stage2ExitCode. See log: $stage2Log"
}

$resultDir = Join-Path $repoRoot "step2a_hyperparameter_tuning\\hyperparameter_tuning_second_stage\\results_afm"
$resultRoot = [System.IO.Path]::GetFileNameWithoutExtension($resultBasename)
$resultExt = [System.IO.Path]::GetExtension($resultBasename)
$resultMirrorDir = Join-Path $repoRoot ("logs\\" + ($resultMirrorSubdir -replace '/', '\\'))
$copied = @()

$mergedResult = Join-Path $resultDir $resultBasename
if (Test-Path -Path $mergedResult) {
  Copy-Item -Path $mergedResult -Destination (Join-Path $resultMirrorDir $resultBasename) -Force
  $copied += $resultBasename
}

$shardPattern = $resultRoot + "_p*" + $resultExt
Get-ChildItem -Path $resultDir -Filter $shardPattern -File -ErrorAction SilentlyContinue | ForEach-Object {
  Copy-Item -Path $_.FullName -Destination (Join-Path $resultMirrorDir $_.Name) -Force
  $copied += $_.Name
}

if ($copied.Count -gt 0) {
  Write-Host "Mirrored result files -> logs\\$($resultMirrorSubdir -replace '/', '\\')"
}
