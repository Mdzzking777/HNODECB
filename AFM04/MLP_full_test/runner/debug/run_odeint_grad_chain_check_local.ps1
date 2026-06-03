param(
  [int]$ShardIndex = 1,
  [int]$MaxPoints = 256
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
while ($true) {
  $hasAFM04 = Test-Path -Path (Join-Path $repoRoot "AFM04")
  if ($hasAFM04) {
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

Write-Host "Running MFT odeint gradient-chain check"
Write-Host "Repo root   -> $repoRoot"
Write-Host "Python      -> $pythonExe"
Write-Host "Shard index -> $ShardIndex"
Write-Host "Max points  -> $MaxPoints"

& $pythonExe -m AFM04.MLP_full_test.runner.debug.check_odeint_grad_chain --shard-index $ShardIndex --max-points $MaxPoints
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
  throw "MFT odeint gradient-chain check failed with exit code $exitCode"
}
