param(
  [int]$WindowIndex = 1,
  [int]$ShardCount = 8,
  [int]$TopK = -1,
  [string]$OutputPrefix = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..\..\..")
$repoRoot = $repoRoot.Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$analysisScript = Join-Path $repoRoot "AFM04\KAN_full_test\runner\analysis\analyze_kft_rs_refit_rerank_04.py"
$analysisOutDir = Join-Path $repoRoot "AFM04\KAN_full_test\logs\analysis"
$logDir = Join-Path $repoRoot "AFM04\KAN_full_test\logs\analysis\refit_rerank_shards"

if (-not (Test-Path -LiteralPath $pythonExe)) {
  throw "Missing venv python: $pythonExe"
}
if (-not (Test-Path -LiteralPath $analysisScript)) {
  throw "Missing analysis script: $analysisScript"
}
if ($ShardCount -lt 1) {
  throw "ShardCount must be >= 1"
}

if ([string]::IsNullOrWhiteSpace($OutputPrefix)) {
  $scope = "top10pct"
  if ($TopK -eq 0) {
    $scope = "all"
  }
  if ($TopK -gt 0) {
    $scope = "top$TopK"
  }
  $OutputPrefix = "kft_rs_refit_rerank_w0_$scope"
}

New-Item -ItemType Directory -Force -Path $analysisOutDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Always start from a clean rerank analysis area. These are postprocess
# artifacts only; this does not touch RS/GBO results, checkpoints, or archive.
Write-Host "Cleaning previous KFT RS refit rerank artifacts..."
Get-ChildItem -LiteralPath $logDir -Force -ErrorAction SilentlyContinue |
  Remove-Item -Force -Recurse
Get-ChildItem -LiteralPath $analysisOutDir -Force -ErrorAction SilentlyContinue |
  Where-Object {
    $_.Name -like "kft_rs_refit_rerank_*.json" -or
    $_.Name -like "kft_rs_refit_rerank_*.csv" -or
    $_.Name -like "$OutputPrefix*.json" -or
    $_.Name -like "$OutputPrefix*.csv"
  } |
  Remove-Item -Force

Write-Host "Running KFT RS refit rerank analysis"
Write-Host "Repo root     -> $repoRoot"
Write-Host "Python        -> $pythonExe"
Write-Host "Window index  -> $WindowIndex"
Write-Host "Shard count   -> $ShardCount"
Write-Host "TopK          -> $TopK (-1 means top 10% of RS trial budget; 0 means all valid trials)"
Write-Host "Output prefix -> $OutputPrefix"
Write-Host "Shard logs    -> $logDir"

$jobs = @()
foreach ($shard in 1..$ShardCount) {
  $stdout = Join-Path $logDir ("refit_rerank_s{0}of{1}.out.txt" -f $shard, $ShardCount)
  $stderr = Join-Path $logDir ("refit_rerank_s{0}of{1}.err.txt" -f $shard, $ShardCount)
  $expectedJson = Join-Path (Join-Path $repoRoot "AFM04\KAN_full_test\logs\analysis") ("{0}_s{1}of{2}.json" -f $OutputPrefix, $shard, $ShardCount)
  $expectedCsv = Join-Path (Join-Path $repoRoot "AFM04\KAN_full_test\logs\analysis") ("{0}_s{1}of{2}.csv" -f $OutputPrefix, $shard, $ShardCount)
  Remove-Item -LiteralPath $expectedJson, $expectedCsv -Force -ErrorAction SilentlyContinue
  $argList = @(
    $analysisScript,
    "--window-index", "$WindowIndex",
    "--shard-count", "$ShardCount",
    "--shard-index", "$shard",
    "--output-prefix", $OutputPrefix
  )
  if ($TopK -ne -1) {
    $argList += @("--top-k", "$TopK")
  }
  $proc = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $argList `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden
  $jobs += [pscustomobject]@{
    Shard = $shard
    Process = $proc
    Stdout = $stdout
    Stderr = $stderr
    ExpectedJson = $expectedJson
    ExpectedCsv = $expectedCsv
  }
  Write-Host ("launched shard {0}/{1} | pid={2}" -f $shard, $ShardCount, $proc.Id)
}

$failed = @()
foreach ($job in $jobs) {
  $job.Process.WaitForExit()
  $job.Process.Refresh()
  $exitCode = $job.Process.ExitCode
  $exitCodeText = $exitCode
  if ($null -eq $exitCode) {
    $exitCodeText = "pending-json-check"
  }
  $stderrLen = 0
  if (Test-Path -LiteralPath $job.Stderr) {
    $stderrLen = (Get-Item -LiteralPath $job.Stderr).Length
  }
  $jsonExists = Test-Path -LiteralPath $job.ExpectedJson
  $ok = ($exitCode -eq 0) -or (($null -eq $exitCode) -and $jsonExists -and ($stderrLen -eq 0))
  if (($null -eq $exitCode) -and $ok) {
    $exitCodeText = "ok-json"
  }
  Write-Host ("shard {0}/{1} exited | code={2} | json={3} | log={4}" -f $job.Shard, $ShardCount, $exitCodeText, $jsonExists, $job.Stdout)
  if (-not $ok) {
    $failed += $job
  }
}

if ($failed.Count -gt 0) {
  $failedSummary = ($failed | ForEach-Object { "s$($_.Shard): stderr=$($_.Stderr)" }) -join "; "
  throw "KFT RS refit rerank shard failure(s): $failedSummary"
}

Write-Host "Merging shard outputs..."
$mergeArgs = @(
  $analysisScript,
  "--window-index", "$WindowIndex",
  "--shard-count", "$ShardCount",
  "--merge-shards",
  "--output-prefix", $OutputPrefix
)
if ($TopK -ne -1) {
  $mergeArgs += @("--top-k", "$TopK")
}
& $pythonExe @mergeArgs
if ($LASTEXITCODE -ne 0) {
  throw "KFT RS refit rerank merge failed with exit code $LASTEXITCODE"
}

Write-Host "KFT RS refit rerank analysis complete."
