.\.venv\Scripts\python.exe .\create_semi_synthetic_panel.py --write-randomised-action --overwrite-validation-outputs

.\.venv\Scripts\python.exe .\fit_nuisance_models.py --panel validation --model-type xgboost
.\.venv\Scripts\python.exe .\build_policy_intervention_panels.py --panel validation
.\.venv\Scripts\python.exe .\evaluate_gformula_policies.py --panel validation
.\.venv\Scripts\python.exe .\evaluate_ipw_policies.py --panel validation
.\.venv\Scripts\python.exe .\evaluate_aipw_policies.py --panel validation

.\.venv\Scripts\python.exe .\fit_nuisance_models.py --panel validation-omitted --model-type xgboost
.\.venv\Scripts\python.exe .\build_policy_intervention_panels.py --panel validation-omitted
.\.venv\Scripts\python.exe .\evaluate_gformula_policies.py --panel validation-omitted
.\.venv\Scripts\python.exe .\evaluate_ipw_policies.py --panel validation-omitted
.\.venv\Scripts\python.exe .\evaluate_aipw_policies.py --panel validation-omitted

.\.venv\Scripts\python.exe .\fit_nuisance_models.py --panel validation-randomised --model-type xgboost
.\.venv\Scripts\python.exe .\build_policy_intervention_panels.py --panel validation-randomised
.\.venv\Scripts\python.exe .\evaluate_gformula_policies.py --panel validation-randomised
.\.venv\Scripts\python.exe .\evaluate_ipw_policies.py --panel validation-randomised
.\.venv\Scripts\python.exe .\evaluate_aipw_policies.py --panel validation-randomised
