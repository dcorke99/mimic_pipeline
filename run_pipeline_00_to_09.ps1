param(
    [string]$PythonCmd = "python"
)

$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\DavidUni\Repos\mimic_pipeline"

$Steps = @(
    # "00_identify_required_catheter_episodes.py",
    # "01_extract_raw_chart_covariates.py",
    "02_preprocess_raw_chart_covariates.py",
    "03_validate_raw_chart_covariates.py",
    "04_clean_raw_chart_covariates.py",
    "05_validate_cleaned_chart_covariates.py",
    "06_build_master_panel.py",
    "07_select_retained_covariates.py",
    "08_filter_panel.py",
    "09_build_feature_panel.py"
)

Set-Location -Path $RepoRoot

foreach ($Step in $Steps) {
    $StepPath = Join-Path $RepoRoot $Step

    if (-not (Test-Path -LiteralPath $StepPath)) {
        throw "Step not found: $StepPath"
    }

    Write-Host ""
    Write-Host "=== Running $Step ==="
    Write-Host ("Start: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

    & $PythonCmd -u $StepPath
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: $Step"
    }

    Write-Host ("Done:  {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
}

Write-Host ""
Write-Host "Pipeline completed: 00 to 09"
