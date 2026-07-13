param(
  [string]$RankDir = "",
  [ValidateSet("result")]
  [string]$EntryPayloadKind = "result",
  [string]$ArchiveRoot = "AFM05\Archive\st2l\valley",
  [ValidateSet("trained", "mech")]
  [string]$ParameterSet = "trained",
  [double]$FiniteDiffEps = 1.0e-4,
  [double]$BaselineReproductionTolerance = 1.0e-6,
  [string]$Components = "x1,x2,x2dot",
  [int]$ParameterLimit = 0,
  [int]$ShardCount = 6
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
while ($true) {
  if (Test-Path -Path (Join-Path $repoRoot "AFM05")) {
    break
  }

  $parent = Split-Path -Path $repoRoot -Parent
  if ([string]::IsNullOrEmpty($parent) -or $parent -eq $repoRoot) {
    throw "Could not locate repository root from $PSScriptRoot"
  }
  $repoRoot = $parent
}

Set-Location -Path $repoRoot

$step3Root = Join-Path $repoRoot "AFM05\step3_parameters_identifiability"
$runnerDir = Join-Path $step3Root "runner"
$vizRunnerDir = Join-Path $runnerDir "visualization runner"
$resultsDir = Join-Path $step3Root "results"
$currentEntry = Join-Path $resultsDir "current_step3_entry.json"

$identifiabilityRunner = Join-Path $runnerDir "run_step3_identifiability_05_local.ps1"
$postAnalysisRunner = Join-Path $runnerDir "run_step3_post_analysis_05_local.ps1"
$thresholdRunner = Join-Path $vizRunnerDir "run_step3_threshold_sweep_05_local.ps1"
$eigenRunner = Join-Path $vizRunnerDir "run_step3_eigen_spectrum_05_local.ps1"
$nullProjectionRunner = Join-Path $vizRunnerDir "run_step3_null_projection_groups_05_local.ps1"

foreach ($script in @($identifiabilityRunner, $postAnalysisRunner, $thresholdRunner, $eigenRunner, $nullProjectionRunner)) {
  if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing required step3 runner: $script"
  }
}

if ([string]::IsNullOrWhiteSpace($RankDir)) {
  if (-not (Test-Path -LiteralPath $currentEntry)) {
    throw "RankDir was not provided and current step3 entry is missing: $currentEntry"
  }
  $entry = Get-Content -LiteralPath $currentEntry -Raw | ConvertFrom-Json
  $RankDir = [string]$entry.rank_dir
  if ([string]::IsNullOrWhiteSpace($RankDir)) {
    throw "Current step3 entry does not contain rank_dir: $currentEntry"
  }
}

Write-Host "Running AFM05 step3 full pipeline"
Write-Host "Repo root     -> $repoRoot"
Write-Host "Rank dir      -> $RankDir"
Write-Host "Archive root  -> $ArchiveRoot"
Write-Host "Payload       -> $EntryPayloadKind"
Write-Host "Parameter set -> $ParameterSet"
Write-Host "Shard count   -> $ShardCount"

& $identifiabilityRunner `
  -ArchiveRoot $ArchiveRoot `
  -RankDir $RankDir `
  -EntryPayloadKind $EntryPayloadKind `
  -ParameterSet $ParameterSet `
  -FiniteDiffEps $FiniteDiffEps `
  -BaselineReproductionTolerance $BaselineReproductionTolerance `
  -Components $Components `
  -ParameterLimit $ParameterLimit `
  -ShardCount $ShardCount
if ($LASTEXITCODE -ne 0) {
  throw "Step3 identifiability runner failed with exit code $LASTEXITCODE"
}

& $postAnalysisRunner
if ($LASTEXITCODE -ne 0) {
  throw "Step3 post-analysis runner failed with exit code $LASTEXITCODE"
}

& $thresholdRunner
if ($LASTEXITCODE -ne 0) {
  throw "Step3 threshold-sweep visualization runner failed with exit code $LASTEXITCODE"
}

& $eigenRunner
if ($LASTEXITCODE -ne 0) {
  throw "Step3 eigen-spectrum visualization runner failed with exit code $LASTEXITCODE"
}

& $nullProjectionRunner
if ($LASTEXITCODE -ne 0) {
  throw "Step3 null-projection visualization runner failed with exit code $LASTEXITCODE"
}

Write-Host "AFM05 step3 full pipeline completed"
