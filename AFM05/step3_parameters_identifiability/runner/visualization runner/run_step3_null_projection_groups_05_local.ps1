param(
  [string]$InputResult = "AFM05\step3_parameters_identifiability\results\afm05_step3_identifiability_trained_latest.json",
  [double]$AbsFloor = 1.0e-5,
  [double]$RelFactor = 0.0,
  [string]$Thresholds = "1e-5,1e-6,1e-7,1e-8",
  [int]$TopK = 12,
  [string]$OutputTag = ""
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

$visualizationScript = Join-Path $step3Root "visualize_afm05_step3_post_analysis.py"
if (-not (Test-Path -LiteralPath $visualizationScript)) {
  throw "Missing AFM05 step3 visualization script: $visualizationScript"
}

$inputResultPath = $InputResult
if (-not ([System.IO.Path]::IsPathRooted($inputResultPath))) {
  $inputResultPath = Join-Path $repoRoot $inputResultPath
}
if (-not (Test-Path -LiteralPath $inputResultPath)) {
  throw "Input step3 result not found: $inputResultPath"
}

$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logsDir ("afm05_step3_visualization_null_projection_groups_{0}.txt" -f $runStamp)
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$absFloorText = $AbsFloor.ToString("G17", $invariant)
$relFactorText = $RelFactor.ToString("G17", $invariant)

$transcriptStarted = $false
try {
  Start-Transcript -Path $logPath -Force | Out-Null
  $transcriptStarted = $true

  Write-Host "Running AFM05 step3 null-projection groups visualization"
  Write-Host "Input result -> $inputResultPath"
  Write-Host "Run log      -> $logPath"

  $argsList = @(
    $visualizationScript,
    "--input-result", $inputResultPath,
    "--results-dir", $resultsDir,
    "--visualization-dir", $visualizationDir,
    "--abs-floor", $absFloorText,
    "--rel-factor", $relFactorText,
    "--thresholds", $Thresholds,
    "--top-k", "$TopK",
    "--plots", "groups",
    "--stable-output-tag",
    "--no-latest"
  )
  if (-not [string]::IsNullOrWhiteSpace($OutputTag)) {
    $argsList += @("--output-tag", $OutputTag)
  }

  & $pythonExe @argsList 2>&1 | ForEach-Object { Write-Host $_ }
  if ($LASTEXITCODE -ne 0) {
    throw "AFM05 step3 null-projection groups visualization failed with exit code $LASTEXITCODE."
  }
} finally {
  if ($transcriptStarted) {
    Stop-Transcript | Out-Null
  }
}

Write-Host "Run log -> $logPath"
