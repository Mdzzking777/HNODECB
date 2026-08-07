[CmdletBinding()]
param(
    [int]$Seed = 1
)

$ErrorActionPreference = "Stop"
$AfmRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $AfmRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Generator = Join-Path $PSScriptRoot "generate_measurement_noise_datasets_06a.py"
$Plotter = Join-Path $PSScriptRoot "plot_measurement_noise_full_timespan_06a.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Repository Python executable was not found: $Python"
}
if (-not (Test-Path -LiteralPath $Generator -PathType Leaf)) {
    throw "Noise generator was not found: $Generator"
}
if (-not (Test-Path -LiteralPath $Plotter -PathType Leaf)) {
    throw "Noise visualization script was not found: $Plotter"
}

Push-Location $RepoRoot
try {
    & $Python $Generator --seed $Seed --levels 0.01 0.05
    if ($LASTEXITCODE -ne 0) {
        throw "AFM06a noisy-data generation failed with exit code $LASTEXITCODE"
    }
    & $Python $Plotter
    if ($LASTEXITCODE -ne 0) {
        throw "AFM06a noisy-data visualization failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
