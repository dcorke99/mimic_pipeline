.\.venv\Scripts\python.exe .\create_data_panel.py
.\.venv\Scripts\python.exe .\fit_nuisance_models.py --panel real --model-type xgboost
.\.venv\Scripts\python.exe .\build_policy_intervention_panels.py --panel real
.\.venv\Scripts\python.exe .\evaluate_gformula_policies.py --panel real
.\.venv\Scripts\python.exe .\evaluate_ipw_policies.py --panel real
.\.venv\Scripts\python.exe .\evaluate_aipw_policies.py --panel real
.\.venv\Scripts\python.exe .\audit_policy_evaluation_baseline.py
.\.venv\Scripts\python.exe .\prepare_results_plots.py
