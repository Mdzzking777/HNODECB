param(
[string]$DataRoot = "",
[string]$ArchiveDir = "",
[string]$ReferenceManifest = "",
[int]$MaxDatasets = 0,
[ValidateSet("dataset_specific_full_span_x1_mean_std", "trained_archive_x1_mean_std")]
[string]$NormalizerPolicy = "trained_archive_x1_mean_std",
[switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script = Join-Path $PSScriptRoot "predict_q_fd_random_with_trained_kan_06a.py"
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path $PSScriptRoot "data"
}
if ([string]::IsNullOrWhiteSpace($ArchiveDir)) {
    $ArchiveDir = Join-Path $RepoRoot "AFM06a\archive\st2l\rank1_20260801_131123"
}
if ([string]::IsNullOrWhiteSpace($ReferenceManifest)) {
    $ReferenceManifest = Join-Path $RepoRoot "AFM06a\datasets\e0.0_real\data\afm06a_generation_manifest.json"
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $ArchiveDir)) {
    throw "AFM06a trained-KAN archive not found: $ArchiveDir"
}
if (-not (Test-Path -LiteralPath $ReferenceManifest)) {
    throw "AFM06a e0.0_real reference manifest not found: $ReferenceManifest"
}

$Arguments = @(
    $Script,
    "--data-root", $DataRoot,
    "--archive-dir", $ArchiveDir,
    "--reference-manifest", $ReferenceManifest,
    "--normalizer-policy", $NormalizerPolicy
)
if ($MaxDatasets -gt 0) {
    $Arguments += @("--max-datasets", $MaxDatasets)
}
if ($Resume) {
    $Arguments += @("--resume")
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "AFM06a Q x Fd/m trained-KAN prediction failed with exit code $LASTEXITCODE"
}
