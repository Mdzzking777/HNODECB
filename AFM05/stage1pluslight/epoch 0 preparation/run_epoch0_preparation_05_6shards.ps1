param(
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"

function Quote-Arg {
    param([string]$Value)
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

$MainOutDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $MainOutDir "..\..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script = Join-Path $MainOutDir "run_epoch0_preparation_05.py"
$Missing = @($Python, $Script) | Where-Object { -not (Test-Path -LiteralPath $_) }
if ($Missing.Count -gt 0) {
    throw "Required epoch-0 preparation path is missing: $($Missing -join ', ')"
}
$LogDir = Join-Path $MainOutDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $LogDir "parallel_6shards_$Stamp"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$RunLog = Join-Path $LogDir "afm05_st1pl_epoch0_preparation_6shards_parallel_$Stamp.log"
Start-Transcript -Path $RunLog -Force | Out-Null

Write-Host "AFM05 st1pl epoch-0 preparation: launching 6 shards in parallel"
Write-Host "Run root: $RunRoot"
Write-Host "Main output dir: $MainOutDir"
Write-Host "Fresh final overwrite: $Fresh"

$Processes = @()
for ($ShardIndex = 1; $ShardIndex -le 6; $ShardIndex++) {
    $ShardOut = Join-Path $RunRoot ("shard_{0:D2}" -f $ShardIndex)
    New-Item -ItemType Directory -Force -Path $ShardOut | Out-Null
    $ShardStdout = Join-Path $RunRoot ("shard_{0:D2}.stdout.txt" -f $ShardIndex)
    $ShardStderr = Join-Path $RunRoot ("shard_{0:D2}.stderr.txt" -f $ShardIndex)
    $Args = @(
        $Script,
        "--shard-count", "6",
        "--shard-index", "$ShardIndex",
        "--outdir", $ShardOut,
        "--overwrite",
        "--progress-every", "25"
    )
    $ArgLine = ($Args | ForEach-Object { Quote-Arg $_ }) -join " "
    Write-Host "Starting shard $ShardIndex / 6 -> $ShardOut"
    $Proc = Start-Process `
        -FilePath $Python `
        -ArgumentList $ArgLine `
        -RedirectStandardOutput $ShardStdout `
        -RedirectStandardError $ShardStderr `
        -PassThru `
        -WindowStyle Hidden
    $Processes += [PSCustomObject]@{
        Shard = $ShardIndex
        Process = $Proc
        Stdout = $ShardStdout
        Stderr = $ShardStderr
        OutDir = $ShardOut
    }
}

foreach ($Item in $Processes) {
    $Item.Process.WaitForExit()
    $Item.Process.Refresh()
    $DoneFile = Join-Path $Item.OutDir "afm05_st1pl_epoch0_preparation_done.json"
    $CsvFile = Join-Path $Item.OutDir "afm05_st1pl_epoch0_preparation_best_mech_winners.csv"
    $ExitCode = $Item.Process.ExitCode
    Write-Host ("Shard {0} process returned exit code {1}" -f $Item.Shard, $ExitCode)
    if ($null -ne $ExitCode -and $ExitCode -ne 0) {
        Write-Host "stdout: $($Item.Stdout)"
        Write-Host "stderr: $($Item.Stderr)"
        throw "Shard $($Item.Shard) failed"
    }
    if (-not (Test-Path -LiteralPath $DoneFile)) {
        if (-not (Test-Path -LiteralPath $CsvFile)) {
            Write-Host "stdout: $($Item.Stdout)"
            Write-Host "stderr: $($Item.Stderr)"
            throw "Shard $($Item.Shard) did not produce CSV or done marker"
        }
        Write-Host "Shard $($Item.Shard) has no done marker but produced CSV; accepting this legacy run."
    }
}

Write-Host "All 6 shards finished. Merging shard CSV files..."
$MergeStdout = Join-Path $RunRoot "merge.stdout.txt"
$MergeArgs = @(
    $Script,
    "--merge-only",
    "--merge-shard-root", $RunRoot,
    "--outdir", $MainOutDir,
    "--overwrite"
)
if (-not $Fresh) {
    # Merge output is deterministic from this completed run; overwriting stale
    # final reports is intended even in non-Fresh mode.
}
& $Python @MergeArgs 2>&1 | Tee-Object -FilePath $MergeStdout
if ($LASTEXITCODE -ne 0) {
    throw "Shard merge failed"
}

Write-Host "Parallel epoch-0 preparation complete."
Write-Host "Logs written to $LogDir"
Write-Host "Shard run root: $RunRoot"
Write-Host "Final reranked CSV: $(Join-Path $LogDir 'afm05_st1pl_epoch0_preparation_reranked_1000_trials.csv')"

Stop-Transcript | Out-Null
