param(
  [string]$CandidateA = "B01",
  [string]$CandidateB = "B02",
  [string]$CandidateCatalog = "",
  [int]$St2lShardsPerCandidate = 1,
  [int]$Step3ShardCount = 6,
  [ValidateSet("trained", "mech")]
  [string]$Step3ParameterSet = "trained",
  [double]$FiniteDiffEps = 1.0e-4,
  [double]$BaselineReproductionTolerance = 1.0e-6,
  [string]$Components = "x1,x2,x2dot",
  [int]$ParameterLimit = 0,
  [switch]$SkipStage2light,
  [switch]$SkipStep3,
  [switch]$ForceStep3,
  [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
while ($true) {
  if (Test-Path -LiteralPath (Join-Path $repoRoot "AFM05\stage2light")) {
    break
  }
  $parent = Split-Path -Path $repoRoot -Parent
  if ([string]::IsNullOrEmpty($parent) -or $parent -eq $repoRoot) {
    throw "Could not locate repository root from $PSScriptRoot"
  }
  $repoRoot = $parent
}
Set-Location -LiteralPath $repoRoot

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
  $pythonExe = "python"
}
$singleRunner = Join-Path $repoRoot "AFM05\stage2light\runner\run_stage2light_05_local.ps1"
$step3Runner = Join-Path $repoRoot "AFM05\step3_parameters_identifiability\runner\run_step3_full_pipeline_05_local.ps1"
$step3Archiver = Join-Path $repoRoot "AFM05\step3_parameters_identifiability\archive_step3_compact_05.py"
foreach ($requiredScript in @($singleRunner, $step3Runner, $step3Archiver)) {
  if (-not (Test-Path -LiteralPath $requiredScript)) {
    throw "Missing pipeline dependency: $requiredScript"
  }
}
if ($St2lShardsPerCandidate -lt 1) {
  throw "St2lShardsPerCandidate must be at least 1."
}
if ($Step3ShardCount -lt 1) {
  throw "Step3ShardCount must be at least 1."
}

if ([string]::IsNullOrWhiteSpace($CandidateCatalog)) {
  $liveCatalog = Join-Path $repoRoot "AFM05\prestage2\results\afm_prest2_05_candidates_b.pkl.topk.json"
  $archivedCatalog = Join-Path $repoRoot "AFM05\Archive\prest2\valley\results\afm_prest2_05_candidates_b.pkl.topk.json"
  $CandidateCatalog = if (Test-Path -LiteralPath $liveCatalog) { $liveCatalog } else { $archivedCatalog }
} elseif (-not [System.IO.Path]::IsPathRooted($CandidateCatalog)) {
  $CandidateCatalog = Join-Path $repoRoot $CandidateCatalog
}
if (-not (Test-Path -LiteralPath $CandidateCatalog)) {
  throw "Candidate catalog not found: $CandidateCatalog"
}
$prest2InputPath = $CandidateCatalog -replace '\.topk\.json$', ''
if (-not (Test-Path -LiteralPath $prest2InputPath)) {
  throw "Prest2 candidate payload not found beside catalog: $prest2InputPath"
}
$catalogRaw = Get-Content -LiteralPath $CandidateCatalog -Raw | ConvertFrom-Json
$catalog = @()
foreach ($record in $catalogRaw) {
  $catalog += $record
}

function Resolve-Candidate {
  param([Parameter(Mandatory = $true)][string]$Label)
  $match = [regex]::Match($Label.Trim(), '^[Bb](\d+)$')
  if (-not $match.Success) {
    throw "Candidate label must use BXX notation: $Label"
  }
  $number = [int]$match.Groups[1].Value
  if ($number -lt 1 -or $number -gt $catalog.Count) {
    throw "Candidate $Label is outside B01-B$($catalog.Count.ToString('00'))."
  }
  $record = $catalog[$number - 1]
  foreach ($key in @("source_stage1_rank", "source_stage1_trial_id", "source_mech_winner")) {
    if ($null -eq $record.$key) {
      throw "Candidate $Label is missing $key in $CandidateCatalog"
    }
  }
  [pscustomobject]@{
    Label = "B{0:D2}" -f $number
    Number = $number
    Rank = [int]$record.source_stage1_rank
    TrialId = [int]$record.source_stage1_trial_id
    MechWinner = [int]$record.source_mech_winner
  }
}

$candidates = @(
  Resolve-Candidate -Label $CandidateA
  Resolve-Candidate -Label $CandidateB
)
if ($candidates[0].Label -eq $candidates[1].Label) {
  throw "CandidateA and CandidateB must be different."
}

$st2ArchiveRoot = Join-Path $repoRoot "AFM05\Archive\st2l\valley"
$step3ArchiveRoot = Join-Path $repoRoot "AFM05\Archive\step3\valley"
$pipelineLogDir = Join-Path $repoRoot "AFM05\stage2light\logs\pipeline"
New-Item -ItemType Directory -Force -Path $pipelineLogDir | Out-Null

Write-Host "AFM05 two-candidate st2l -> step3 pipeline"
Write-Host "Candidate catalog -> $CandidateCatalog"
foreach ($candidate in $candidates) {
  Write-Host ("  {0} -> rank {1}, trial {2}, mech winner {3}" -f `
    $candidate.Label, $candidate.Rank, $candidate.TrialId, $candidate.MechWinner)
}
Write-Host "st2l execution    -> two isolated candidate processes in parallel"
Write-Host "step3 execution   -> candidates sequential; $Step3ShardCount parameter shards per candidate"
Write-Host "st2l archive      -> $st2ArchiveRoot\BXX_rankXXX"
Write-Host "step3 archive     -> $step3ArchiveRoot\BXX_rankXXX"

if ($PreflightOnly.IsPresent) {
  Write-Host "Prest2 payload    -> $prest2InputPath"
  Write-Host "Pipeline dependencies and candidate identities are valid."
  Write-Host "Pipeline preflight complete; no training or step3 process was launched."
  return
}

if (-not $SkipStage2light.IsPresent) {
  $powershellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
  $liveRoot = Join-Path $repoRoot "AFM05\stage2light\parallel_candidates"
  $launches = @()

  foreach ($candidate in $candidates) {
    $runName = "{0}_rank{1}" -f $candidate.Label, $candidate.Rank
    $outputRoot = Join-Path $liveRoot $runName
    $archiveDir = Join-Path $st2ArchiveRoot $runName
    $archiveManifest = Join-Path $archiveDir "archive_manifest.json"
    $archivedResult = Join-Path $archiveDir "result\stage2light_result_p1.pt"
    $archiveComplete = $false
    if ((Test-Path -LiteralPath $archiveManifest) -and (Test-Path -LiteralPath $archivedResult)) {
      try {
        $meta = Get-Content -LiteralPath $archiveManifest -Raw | ConvertFrom-Json
        $archiveComplete = (($meta.complete -eq $true) -and ($meta.compact -eq $true))
      } catch {
        $archiveComplete = $false
      }
    }
    if ($archiveComplete) {
      Write-Host "Skipping already archived st2l $runName -> $archiveDir"
      continue
    }

    $launcherLogDir = Join-Path $outputRoot "logs\launcher"
    New-Item -ItemType Directory -Force -Path $launcherLogDir | Out-Null
    $stdoutPath = Join-Path $launcherLogDir "stdout.txt"
    $stderrPath = Join-Path $launcherLogDir "stderr.txt"
    $arguments = @(
      "-NoProfile",
      "-ExecutionPolicy Bypass",
      "-File `"$singleRunner`"",
      "-Shards $St2lShardsPerCandidate",
      "-CandidateLabel $($candidate.Label)",
      "-WarmstartSource prest2",
      "-InputCandidate $($candidate.Number)",
      "-Prest2InputPath `"$prest2InputPath`"",
      "-InputRank $($candidate.Rank)",
      "-InputTrialId $($candidate.TrialId)",
      "-InputMechWinner $($candidate.MechWinner)",
      "-OutputRoot `"$outputRoot`"",
      "-ArchiveDir `"$archiveDir`"",
      "-CompactArchive"
    )
    Write-Host "Launching st2l $runName"
    Write-Host "  output  -> $outputRoot"
    Write-Host "  archive -> $archiveDir"
    $process = Start-Process `
      -FilePath $powershellExe `
      -ArgumentList ($arguments -join " ") `
      -WorkingDirectory $repoRoot `
      -WindowStyle Hidden `
      -RedirectStandardOutput $stdoutPath `
      -RedirectStandardError $stderrPath `
      -PassThru
    $launches += [pscustomobject]@{
      RunName = $runName
      Process = $process
      ArchiveDir = $archiveDir
      Stdout = $stdoutPath
      Stderr = $stderrPath
    }
  }

  if ($launches.Count -gt 0) {
    Write-Host "$($launches.Count) isolated st2l candidate process(es) are running in parallel."
  }
  foreach ($launch in $launches) {
    $launch.Process.WaitForExit()
  }
  foreach ($launch in $launches) {
    $launch.Process.Refresh()
    $resultPath = Join-Path $launch.ArchiveDir "result\stage2light_result_p1.pt"
    $manifestPath = Join-Path $launch.ArchiveDir "archive_manifest.json"
    $artifactsComplete = (Test-Path -LiteralPath $resultPath) -and (Test-Path -LiteralPath $manifestPath)
    $rawExitCode = $launch.Process.ExitCode
    $exitCode = if (($null -eq $rawExitCode) -and $artifactsComplete) { 0 } else { $rawExitCode }
    if (($exitCode -ne 0) -or (-not $artifactsComplete)) {
      throw "st2l $($launch.RunName) failed; exit=$exitCode; stdout=$($launch.Stdout); stderr=$($launch.Stderr)"
    }
    Write-Host "Completed and archived st2l $($launch.RunName)"
  }
}

$summaryRecords = @()
foreach ($candidate in $candidates) {
  $runName = "{0}_rank{1}" -f $candidate.Label, $candidate.Rank
  $st2ArchiveDir = Join-Path $st2ArchiveRoot $runName
  $st2Result = Join-Path $st2ArchiveDir "result\stage2light_result_p1.pt"
  $st2Manifest = Join-Path $st2ArchiveDir "archive_manifest.json"
  if (-not (Test-Path -LiteralPath $st2Result)) {
    throw "Completed st2l result is missing for ${runName}: $st2Result"
  }
  if (-not (Test-Path -LiteralPath $st2Manifest)) {
    throw "Compact st2l archive manifest is missing for ${runName}: $st2Manifest"
  }
  $st2ArchiveMeta = Get-Content -LiteralPath $st2Manifest -Raw | ConvertFrom-Json
  if (($st2ArchiveMeta.complete -ne $true) -or ($st2ArchiveMeta.compact -ne $true)) {
    throw "st2l archive is not marked complete+compact for ${runName}: $st2Manifest"
  }

  $step3ArchiveDir = Join-Path $step3ArchiveRoot $runName
  $step3ArchiveManifest = Join-Path $step3ArchiveDir "archive_manifest.json"
  $step3Status = "skipped_by_switch"
  if (-not $SkipStep3.IsPresent) {
    $archiveComplete = $false
    if ((Test-Path -LiteralPath $step3ArchiveManifest) -and -not $ForceStep3.IsPresent) {
      try {
        $existingManifest = Get-Content -LiteralPath $step3ArchiveManifest -Raw | ConvertFrom-Json
        $archiveComplete = ($existingManifest.complete -eq $true)
      } catch {
        $archiveComplete = $false
      }
    }
    if ($archiveComplete) {
      Write-Host "Skipping already archived step3 $runName -> $step3ArchiveDir"
      $step3Status = "already_archived"
    } else {
      $step3StartedAt = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
      Write-Host "Running step3 for $runName"
      & $step3Runner `
        -RankDir $st2ArchiveDir `
        -ArchiveRoot $st2ArchiveRoot `
        -EntryPayloadKind result `
        -ParameterSet $Step3ParameterSet `
        -FiniteDiffEps $FiniteDiffEps `
        -BaselineReproductionTolerance $BaselineReproductionTolerance `
        -Components $Components `
        -ParameterLimit $ParameterLimit `
        -ShardCount $Step3ShardCount

      & $pythonExe $step3Archiver `
        --rank-dir $st2ArchiveDir `
        --archive-dir $step3ArchiveDir `
        --run-name $runName `
        --parameter-set $Step3ParameterSet `
        --parameter-limit $ParameterLimit `
        --started-at $step3StartedAt
      if ($LASTEXITCODE -ne 0) {
        throw "Compact step3 archive failed for $runName with exit code $LASTEXITCODE"
      }
      $step3Status = "completed_now"
    }
  }

  $summaryRecords += [pscustomobject]@{
    candidate = $candidate.Label
    rank = $candidate.Rank
    trial_id = $candidate.TrialId
    mech_winner = $candidate.MechWinner
    st2l_archive = $st2ArchiveDir
    step3_status = $step3Status
    step3_archive = $step3ArchiveDir
  }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$pairName = ($candidates.Label -join "_")
$summaryPath = Join-Path $pipelineLogDir ("st2l_step3_pipeline_{0}_{1}.json" -f $pairName, $stamp)
$summaryRecords | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Host "Pipeline completed -> $summaryPath"
