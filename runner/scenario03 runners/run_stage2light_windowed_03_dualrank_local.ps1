param(
  [int]$RankX = 1,
  [int]$RankY = 8,
  [int]$ThreadsPerRun = 3,
  [int]$SelfTest = 0,
  [int]$BlasThreads = 1,
  [int]$InitRetries = 200,
  [int]$AutoWindows = 1,
  [int]$WindowStart = 1,
  [int]$WindowLen = 399,
  [string]$SessionTag = "",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($RankX -lt 1 -or $RankY -lt 1) {
  throw "RankX and RankY must both be positive integers."
}
if ($RankX -eq $RankY) {
  throw "RankX and RankY must be different."
}
if ($ThreadsPerRun -lt 1) {
  throw "ThreadsPerRun must be >= 1."
}

function Get-SafeSessionTag {
  param([string]$Value)
  if ([string]::IsNullOrWhiteSpace($Value)) {
    return (Get-Date -Format "yyyyMMdd_HHmmss")
  }
  $safe = ($Value -replace '[^A-Za-z0-9._-]+', '_').Trim('_')
  if ([string]::IsNullOrWhiteSpace($safe)) {
    throw "SessionTag '$Value' does not contain any usable filename characters."
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

$childScript = Join-Path $repoRoot "run_stage2light_windowed_03_local.ps1"
if (-not (Test-Path -Path $childScript)) {
  throw "Missing child script: $childScript"
}

$session = Get-SafeSessionTag -Value $SessionTag
$tagX = "RankX"
$tagY = "RankY"

$hostExe = (Get-Process -Id $PID).Path
if (-not $hostExe -or -not (Test-Path -Path $hostExe)) {
  $pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
  if ($pwsh) {
    $hostExe = $pwsh.Source
  } else {
    $powershell = Get-Command powershell -ErrorAction SilentlyContinue
    if (-not $powershell) {
      throw "Could not locate a PowerShell executable for child runs."
    }
    $hostExe = $powershell.Source
  }
}

function Start-Stage2RankProcess {
  param(
    [int]$Rank,
    [string]$RunTag
  )

  $argList = @(
    '-NoLogo',
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $childScript,
    '-Threads', $ThreadsPerRun,
    '-SelfTest', $SelfTest,
    '-BlasThreads', $BlasThreads,
    '-Candidate', $Rank,
    '-InitRetries', $InitRetries,
    '-AutoWindows', $AutoWindows,
    '-WindowStart', $WindowStart,
    '-WindowLen', $WindowLen,
    '-RunTag', $RunTag
  )

  Write-Host "Launch rank $Rank -> RunTag=$RunTag"
  Write-Host ("  " + $hostExe + " " + ($argList -join " "))

  if ($DryRun) {
    return $null
  }

  return Start-Process -FilePath $hostExe -ArgumentList $argList -PassThru
}

Write-Host "Launching two independent Stage2light WINDOWED local runs"
Write-Host "Repo root       -> $repoRoot"
Write-Host "Child script    -> $childScript"
Write-Host "PowerShell exe  -> $hostExe"
Write-Host "Threads/run     -> $ThreadsPerRun"
Write-Host "RankX           -> $RankX"
Write-Host "RankY           -> $RankY"
Write-Host "Ranks pair      -> $RankX / $RankY"
Write-Host "RankX dir       -> logs\\stage2_step2a\\local\\windowed_dualrank\\RankX"
Write-Host "RankY dir       -> logs\\stage2_step2a\\local\\windowed_dualrank\\RankY"
Write-Host "Session tag     -> $session"
Write-Host "Auto windows    -> $AutoWindows"
if ($AutoWindows -eq 0) {
  Write-Host "Manual window   -> start=$WindowStart len=$WindowLen"
}
if ($DryRun) {
  Write-Host "Mode            -> DRY RUN (no processes will be launched)"
}

$procX = Start-Stage2RankProcess -Rank $RankX -RunTag $tagX
$procY = Start-Stage2RankProcess -Rank $RankY -RunTag $tagY

if (-not $DryRun) {
  Write-Host "Started rank $RankX process PID -> $($procX.Id)"
  Write-Host "Started rank $RankY process PID -> $($procY.Id)"
  Write-Host "Each run writes to its own result/log namespace via RunTag."
}
