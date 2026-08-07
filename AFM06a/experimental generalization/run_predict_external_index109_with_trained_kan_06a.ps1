$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\Public\Documents\GitHub\HNODECB"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script = Join-Path $RepoRoot "AFM06a\experimental generalization\predict_external_index109_with_trained_kan_06a.py"
$Archive = Join-Path $RepoRoot "AFM06a\archive\st2l\rank1_20260801_131123"

& $Python $Script --archive-dir $Archive
if ($LASTEXITCODE -ne 0) {
    throw "AFM06a experimental generalization prediction failed with exit code $LASTEXITCODE"
}
