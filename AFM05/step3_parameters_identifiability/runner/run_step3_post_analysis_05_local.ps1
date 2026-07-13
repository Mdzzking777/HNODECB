param(
  [string]$InputResult = "AFM05\step3_parameters_identifiability\results\afm05_step3_identifiability_trained_latest.json",
  [double]$AbsFloor = 1.0e-8,
  [double]$RelFactor = 0.0,
  [string]$Thresholds = "1e-12,1e-10,1e-8",
  [double]$ThresholdLogMin = 1.0e-12,
  [double]$ThresholdLogMax = 1.0e-4,
  [int]$ThresholdPoints = 161,
  [int]$TopK = 12,
  [string]$OutputTag = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
while ($true) {
  $hasAFM05 = Test-Path -Path (Join-Path $repoRoot "AFM05")
  if ($hasAFM05) {
    break
  }

  $parent = Split-Path -Path $repoRoot -Parent
  if ([string]::IsNullOrEmpty($parent) -or $parent -eq $repoRoot) {
    throw "Could not locate repository root from $PSScriptRoot"
  }
  $repoRoot = $parent
}

Set-Location -Path $repoRoot

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -Path $pythonExe)) {
  $pythonExe = "python"
}

$step3Root = Join-Path $repoRoot "AFM05\step3_parameters_identifiability"
$logsDir = Join-Path $step3Root "logs"
$resultsDir = Join-Path $step3Root "results\post_analysis"
$visualizationDir = Join-Path $step3Root "visualization"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
New-Item -ItemType Directory -Path $visualizationDir -Force | Out-Null

$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$analysisScript = Join-Path $step3Root "post_analyze_afm05_step3_identifiability.py"
$logPath = Join-Path $logsDir ("afm05_step3_post_analysis_{0}.txt" -f $runStamp)

if (-not (Test-Path -LiteralPath $analysisScript)) {
  throw "Missing AFM05 step3 post-analysis script: $analysisScript"
}

$inputResultPath = $InputResult
if (-not ([System.IO.Path]::IsPathRooted($inputResultPath))) {
  $inputResultPath = Join-Path $repoRoot $inputResultPath
}
if (-not (Test-Path -LiteralPath $inputResultPath)) {
  throw "Input step3 result not found: $inputResultPath"
}

$transcriptStarted = $false
try {
  Start-Transcript -Path $logPath -Force | Out-Null
  $transcriptStarted = $true

  Write-Host "Running AFM05 step3 post-analysis"
  Write-Host "Repo root     -> $repoRoot"
  Write-Host "Python        -> $pythonExe"
  Write-Host "Input result  -> $inputResultPath"
  Write-Host "Results dir   -> $resultsDir"
  Write-Host "Figures       -> disabled; use runner\visualization runner scripts"
  if ($ThresholdPoints -gt 0) {
    Write-Host "Thresholds    -> primary=max($AbsFloor, $RelFactor * max_abs_eval), sweep=logspace($ThresholdLogMin, $ThresholdLogMax, $ThresholdPoints)"
  } else {
    Write-Host "Thresholds    -> primary=max($AbsFloor, $RelFactor * max_abs_eval), sweep=$Thresholds"
  }
  Write-Host "Run log       -> $logPath"

  $invariant = [System.Globalization.CultureInfo]::InvariantCulture
  $absFloorText = $AbsFloor.ToString("G17", $invariant)
  $relFactorText = $RelFactor.ToString("G17", $invariant)
  $thresholdLogMinText = $ThresholdLogMin.ToString("G17", $invariant)
  $thresholdLogMaxText = $ThresholdLogMax.ToString("G17", $invariant)

  $argsList = @(
    $analysisScript,
    "--input-result", $inputResultPath,
    "--results-dir", $resultsDir,
    "--visualization-dir", $visualizationDir,
    "--abs-floor", $absFloorText,
    "--rel-factor", $relFactorText,
    "--plots", "none",
    "--top-k", "$TopK"
  )

  if ($ThresholdPoints -gt 0) {
    $argsList += @(
      "--threshold-log-min", $thresholdLogMinText,
      "--threshold-log-max", $thresholdLogMaxText,
      "--threshold-points", "$ThresholdPoints"
    )
  } else {
    $argsList += @("--thresholds", $Thresholds)
  }

  if (-not [string]::IsNullOrWhiteSpace($OutputTag)) {
    $argsList += @("--output-tag", $OutputTag)
  }

  & $pythonExe @argsList 2>&1 | ForEach-Object { Write-Host $_ }
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    throw "AFM05 step3 post-analysis failed with exit code $exitCode."
  }

  Write-Host "Step3 post-analysis status -> completed"
} finally {
  if ($transcriptStarted) {
    Stop-Transcript | Out-Null
  }
}

Write-Host "Run log -> $logPath"
