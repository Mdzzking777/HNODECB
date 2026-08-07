param(
    [int]$Nsteps = 750000,
    [int]$NumQ = 21,
    [int]$NumFdOverM = 21,
    [double]$MaxFactor = 1.5,
    [string]$OutputRoot = "",
    [string]$ReferenceManifest = "",
    [switch]$Overwrite,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script = Join-Path $PSScriptRoot "generate_q_fd_random_sweep_datasets_06a.py"
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $PSScriptRoot "data"
}
if ([string]::IsNullOrWhiteSpace($ReferenceManifest)) {
    $ReferenceManifest = Join-Path $RepoRoot "AFM06a\datasets\e0.0_real\data\afm06a_generation_manifest.json"
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $ReferenceManifest)) {
    throw "AFM06a reference manifest not found: $ReferenceManifest"
}

$Arguments = @(
    $Script,
    "--output-root", $OutputRoot,
    "--reference-manifest", $ReferenceManifest,
    "--nsteps", $Nsteps,
    "--num-q", $NumQ,
    "--num-fd-over-m", $NumFdOverM,
    "--max-factor", $MaxFactor
)
if ($Overwrite) { $Arguments += "--overwrite" }
if ($SmokeTest) { $Arguments += "--smoke-test" }

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "AFM06a random Q x Fd/m data generation failed with exit code $LASTEXITCODE"
}
