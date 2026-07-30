# Run the catheter-removal pipeline with the local virtual environment
$ErrorActionPreference = "Stop"

$Python = ".venv\Scripts\python.exe"

$Scripts = @(
    "create_data_panel.py",
    "fit_nuisance_models.py",
    "build_policy_intervention_panels.py",
    "evaluate_gformula_policies.py",
    "evaluate_ipw_policies.py",
    "evaluate_aipw_policies.py",
    "audit_policy_evaluation_baseline.py",
    "prepare_results_plots.py"
)

foreach ($Script in $Scripts) {
    Write-Host ""
    Write-Host "Running $Script" -ForegroundColor Cyan
    $ScriptPath = Join-Path $PSScriptRoot $Script
    & $Python $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Script failed with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Pipeline complete." -ForegroundColor Green
