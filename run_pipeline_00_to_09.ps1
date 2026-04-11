param(
    [string]$PythonCmd = "python"
)

$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\DavidUni\Repos\mimic_pipeline"

# Run the core panel-building steps in order.
$Steps = @(
    "00_build_base_panel.py",
    "01_extract_raw_chart_covariates.py",
    "02_preprocess_raw_chart_covariates.py",
    "03_validate_raw_chart_covariates.py",
    "04_clean_raw_chart_covariates.py",
    "05_validate_cleaned_chart_covariates.py",
    "06_build_master_panel.py",
    "07_select_retained_covariates.py",
    "08_build_filtered_panel.py",
    "09_build_feature_panel.py"
)

Set-Location -Path $RepoRoot

foreach ($Step in $Steps) {
    # Resolve the full script path once before execution.
    $StepPath = Join-Path $RepoRoot $Step

    if (-not (Test-Path -LiteralPath $StepPath)) {
        throw "Step not found: $StepPath"
    }

    # Print start timing before the Python process begins.
    Write-Host ""
    Write-Host "=== Running $Step ==="
    Write-Host ("Start: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

    # Run the step and stop immediately if Python returns a non-zero exit code.
    & $PythonCmd -u $StepPath
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: $Step"
    }

    # Print end timing after a successful step run.
    Write-Host ("Done:  {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
}

Write-Host ""
Write-Host "Pipeline completed: 00 to 09"
