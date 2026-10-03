$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

# & $Python (Join-Path $PSScriptRoot 'create_data_panel.py')
# & $Python (Join-Path $PSScriptRoot 'create_semi_synthetic_panel.py')
# & $Python (Join-Path $PSScriptRoot 'fit_nuisance_models.py')
# & $Python (Join-Path $PSScriptRoot 'build_policy_intervention_panels.py')
& $Python (Join-Path $PSScriptRoot 'evaluate_gformula_policies.py')
& $Python (Join-Path $PSScriptRoot 'evaluate_ipw_policies.py')
& $Python (Join-Path $PSScriptRoot 'evaluate_aipw_policies.py')
& $Python (Join-Path $PSScriptRoot 'audit_policy_evaluation_baseline.py')
& $Python (Join-Path $PSScriptRoot 'prepare_results_plots.py')
