# Full reproduction pipeline (PowerShell)
# Usage:  pwsh run_all.ps1            (skip data generation, use existing data)
#         pwsh run_all.ps1 -Regenerate

param([switch]$Regenerate)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($name, $script, $argsList) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host ("==> " + $name)
    Write-Host ("=" * 70)
    & python $script @argsList
    if ($LASTEXITCODE -ne 0) { throw ("failed: " + $script) }
}

if ($Regenerate) { Step "generate dataset" "scripts\01_generate_data.py" @() }
Step "tabular baselines"      "scripts\02_tabular_baselines.py" @()
Step "train sequence models"  "scripts\03_train_sequence.py" @()
Step "evaluate all splits"    "scripts\04_evaluate.py" @()
Step "uncertainty"            "scripts\05_uncertainty.py" @()
Step "transfer"               "scripts\06_transfer.py" @()
Step "interpretability"       "scripts\07_interpretability.py" @()
Step "protocol optimisation"  "scripts\08_protocol_optimization.py" @()
Step "figures"                "scripts\09_figures.py" @()
Step "statistics"             "scripts\10_statistics.py" @()
Step "paper numbers"          "scripts\11_paper_numbers.py" @()

Push-Location paper
try {
    $tex = Get-Command tectonic -ErrorAction SilentlyContinue
    if ($tex) { & tectonic --reruns 2 main.tex }
    else { & pdflatex main.tex; & bibtex main; & pdflatex main.tex; & pdflatex main.tex }
} finally { Pop-Location }

Write-Host ""
Write-Host "done - see results\tables, results\figures and paper\main.pdf"
