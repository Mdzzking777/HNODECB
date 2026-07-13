param(
  [int]$Rank = 0,
  [string]$Candidate = "",
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
  [int]$ShardCount = 6,
  [switch]$SkipIdentifiability,
  [switch]$List
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
if ($ShardCount -lt 1) {
  throw "ShardCount must be >= 1"
}

$step3Root = Join-Path $repoRoot "AFM05\step3_parameters_identifiability"
$logsDir = Join-Path $step3Root "logs"
$resultsDir = Join-Path $step3Root "results"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null

$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$selectorScript = Join-Path $step3Root "set_afm05_step3_entry.py"
$computeScript = Join-Path $step3Root "compute_afm05_step3_identifiability.py"
$currentEntry = Join-Path $resultsDir "current_step3_entry.json"
$currentEntrySnapshot = Join-Path $resultsDir ("current_step3_entry_{0}.json" -f $runStamp)
$logPath = Join-Path $logsDir ("afm05_step3_identifiability_{0}.txt" -f $runStamp)

if (-not (Test-Path -LiteralPath $selectorScript)) {
  throw "Missing AFM05 step3 selector script: $selectorScript"
}
if (-not (Test-Path -LiteralPath $computeScript)) {
  throw "Missing AFM05 step3 compute script: $computeScript"
}

$transcriptStarted = $false
try {
  Start-Transcript -Path $logPath -Force | Out-Null
  $transcriptStarted = $true

  Write-Host "Running AFM05 step3 entry selector"
  Write-Host "Repo root -> $repoRoot"
  Write-Host "Python    -> $pythonExe"
  Write-Host "Archive   -> $ArchiveRoot"
  Write-Host "Payload   -> $EntryPayloadKind"
  Write-Host "Step3     -> parameter_set=$ParameterSet finite_diff_eps=$FiniteDiffEps baseline_tol=$BaselineReproductionTolerance components=$Components parameter_limit=$ParameterLimit shard_count=$ShardCount"
  Write-Host "Logs dir  -> $logsDir"
  Write-Host "Results   -> $resultsDir"
  Write-Host "Run log   -> $logPath"

  $argsList = @(
    $selectorScript,
    "--archive-root", $ArchiveRoot,
    "--entry-payload-kind", $EntryPayloadKind,
    "--output", $currentEntry
  )

  $runSelector = $true
  if ($List.IsPresent) {
    $argsList += "--list"
  } elseif (-not [string]::IsNullOrWhiteSpace($RankDir)) {
    $argsList += @("--rank-dir", $RankDir)
    Write-Host "Selector  -> rank dir $RankDir"
  } elseif (-not [string]::IsNullOrWhiteSpace($Candidate)) {
    $argsList += @("--candidate", $Candidate)
    Write-Host "Selector  -> candidate $Candidate"
  } elseif ($Rank -gt 0) {
    $argsList += @("--rank", "$Rank")
    Write-Host "Selector  -> rank $Rank"
  } else {
    $runSelector = $false
    if (-not (Test-Path -LiteralPath $currentEntry)) {
      throw "No rank/candidate was provided and the current step3 entry is missing: $currentEntry"
    }
    Write-Host "Selector  -> preserve current entry $currentEntry"
  }

  if ($runSelector) {
    & $pythonExe @argsList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
      throw "AFM05 step3 entry selector failed with exit code $exitCode."
    }
  }

  if (-not $List.IsPresent) {
    if (-not (Test-Path -LiteralPath $currentEntry)) {
      throw "Step3 selector completed but current entry file was not written: $currentEntry"
    }
    Copy-Item -LiteralPath $currentEntry -Destination $currentEntrySnapshot -Force
    Write-Host "Current step3 entry -> $currentEntry"
    Write-Host "Entry snapshot      -> $currentEntrySnapshot"

    if ($SkipIdentifiability.IsPresent) {
      Write-Host "Step3 status -> entry selected and validated; identifiability computation skipped by -SkipIdentifiability"
    } else {
      $invariant = [System.Globalization.CultureInfo]::InvariantCulture
      $epsText = $FiniteDiffEps.ToString("G17", $invariant)
      $baselineTolText = $BaselineReproductionTolerance.ToString("G17", $invariant)

      if (($ParameterSet -eq "trained") -and ($ShardCount -gt 1)) {
        $shardRunDir = Join-Path $resultsDir ("shards\step3_{0}_trained_shards{1}" -f $runStamp, $ShardCount)
        New-Item -ItemType Directory -Path $shardRunDir -Force | Out-Null
        Write-Host "Running AFM05 step3 trained parameter-column shards"
        Write-Host "Shard run -> $shardRunDir"

        $jobs = @()
        for ($shardIndex = 1; $shardIndex -le $ShardCount; $shardIndex++) {
          $shardLog = Join-Path $logsDir ("afm05_step3_identifiability_{0}_shard_{1:D3}_of_{2:D3}.txt" -f $runStamp, $shardIndex, $ShardCount)
          $shardArgs = @(
            $computeScript,
            "--entry", $currentEntry,
            "--results-dir", $resultsDir,
            "--parameter-set", $ParameterSet,
            "--finite-diff-eps", $epsText,
            "--baseline-reproduction-tolerance", $baselineTolText,
            "--components", $Components,
            "--parameter-limit", "$ParameterLimit",
            "--shard-count", "$ShardCount",
            "--shard-index", "$shardIndex",
            "--shard-run-dir", $shardRunDir
          )
          Write-Host ("Starting shard {0}/{1} -> {2}" -f $shardIndex, $ShardCount, $shardLog)
          $jobs += Start-Job -ScriptBlock {
            param($PythonExe, $ShardArgs, $ShardLog)
            $ErrorActionPreference = "Stop"
            ("[{0}] shard process start" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) | Tee-Object -FilePath $ShardLog -Append
            & $PythonExe @ShardArgs 2>&1 | Tee-Object -FilePath $ShardLog -Append
            $exitCode = $LASTEXITCODE
            ("[{0}] shard process exit_code={1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $exitCode) | Tee-Object -FilePath $ShardLog -Append
            if ($exitCode -ne 0) {
              exit $exitCode
            }
          } -ArgumentList $pythonExe, $shardArgs, $shardLog
        }

        $jobs | Wait-Job | Out-Null
        $failedJobs = @()
        foreach ($job in $jobs) {
          $jobOutput = Receive-Job -Job $job
          if ($jobOutput) {
            $jobOutput | ForEach-Object { Write-Host $_ }
          }
          if ($job.State -ne "Completed") {
            $failedJobs += $job
          }
        }
        $jobs | Remove-Job -Force
        if ($failedJobs.Count -gt 0) {
          throw "One or more AFM05 step3 shard jobs failed: $($failedJobs.Count)"
        }

        Write-Host "All shard jobs completed; merging sensitivity columns and computing H/eigen"
        $mergeArgs = @(
          $computeScript,
          "--results-dir", $resultsDir,
          "--parameter-set", $ParameterSet,
          "--shard-count", "$ShardCount",
          "--shard-run-dir", $shardRunDir,
          "--merge-shards"
        )
        & $pythonExe @mergeArgs 2>&1 | ForEach-Object { Write-Host $_ }
        $computeExitCode = $LASTEXITCODE
        if ($computeExitCode -ne 0) {
          throw "AFM05 step3 shard merge/eigen computation failed with exit code $computeExitCode."
        }
      } else {
        if (($ParameterSet -ne "trained") -and ($ShardCount -gt 1)) {
          Write-Host "ShardCount is ignored for parameter_set=$ParameterSet"
        }
        $computeArgs = @(
          $computeScript,
          "--entry", $currentEntry,
          "--results-dir", $resultsDir,
          "--parameter-set", $ParameterSet,
          "--finite-diff-eps", $epsText,
          "--baseline-reproduction-tolerance", $baselineTolText,
          "--components", $Components,
          "--parameter-limit", "$ParameterLimit"
        )

        Write-Host "Running AFM05 step3 identifiability eigen computation"
        & $pythonExe @computeArgs 2>&1 | ForEach-Object { Write-Host $_ }
        $computeExitCode = $LASTEXITCODE
        if ($computeExitCode -ne 0) {
          throw "AFM05 step3 identifiability computation failed with exit code $computeExitCode."
        }
      }
      Write-Host "Step3 status -> entry selected, validated, and identifiability eigen computation completed"
    }
  }
} finally {
  if ($transcriptStarted) {
    Stop-Transcript | Out-Null
  }
}

Write-Host "Run log -> $logPath"
