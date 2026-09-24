.\.venv\Scripts\python.exe .\create_semi_synthetic_panel.py --write-randomised-action

foreach ($panel in @("validation", "validation-omitted", "validation-randomised")) {
    .\.venv\Scripts\python.exe .\fit_nuisance_models.py --panel $panel
    .\.venv\Scripts\python.exe .\build_policy_intervention_panels.py --panel $panel
    .\.venv\Scripts\python.exe .\evaluate_gformula_policies.py --panel $panel
    .\.venv\Scripts\python.exe .\evaluate_ipw_policies.py --panel $panel
    .\.venv\Scripts\python.exe .\evaluate_aipw_policies.py --panel $panel
}
