$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script = Join-Path $RepoRoot "AFM06a\experimental generalization\rollout_external_index109_x1_with_trained_kan_06a.py"
$Archive = Join-Path $RepoRoot "AFM06a\archive\st2l\rank1_20260803_142741"
$Output = Join-Path $RepoRoot "AFM06a\experimental generalization\x1_real_super_real"

& $Python $Script --archive-dir $Archive --output-dir $Output
if ($LASTEXITCODE -ne 0) {
    throw "AFM06a external x1 rollout failed with exit code $LASTEXITCODE"
}
