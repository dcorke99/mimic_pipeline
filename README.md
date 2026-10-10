# MIMIC catheter policy pipeline

`requirements.txt` defines the python environment

Run each script with `python <script.py>` or VS Code

Raw MIMIC-IV data defaults to `../Data/MIMIC-IV/mimic-iv-3.1`, overridable with `MIMIC_DIR`.

## Main run order

The pipeline is run on separate data panels representing real data, semi-synthetic measured confounding, confounder omitted, and randomised actions. It supports 3 estimators (g-formula, IPW and AIPW) each with 2 confidence interval bootstrap modes (fixed-model and refit-model). For each selected learner this workflow generates up to **4 panels × 3 estimators × 2 bootstrap modes = 24 combinations**.

Each script has its own configuration section.

Select panels independently by editing `PANEL_RUNS` in `fit_nuisance_models.py`, `build_policy_panels.py` and each evaluator. Comment out entries to skip panels.

Select learners with `MODEL_TYPE="xgboost" or MODEL_TYPE="all"` in `fit_nuisance_models.py` and evaluation learners with `NUISANCE_MODEL_TYPE` in each evaluator script. Bootstrap modes are selected in each evaluator by setting `BOOTSTRAP_MODES`.

For fixed-model bootstrap, each evaluator’s `NUISANCE_MODEL_TYPE` must refer to a learner previously fitted for the selected panels by `fit_nuisance_models.py`.

Refitted-model bootstrap fits the selected learner during evaluation and can run without saved models or predictions.

Implemented nuisance models are logistic regression, random forest, XGBoost, LightGBM and Super Learner. Set `MODEL_TYPE="superlearner"` to fit only the ensemble, or `MODEL_TYPE="all"` to include it with the standalone learners. Its outputs use the `superlearner` folder and the same prediction columns, diagnostics and saved-model formats; evaluators can select it with `NUISANCE_MODEL_TYPE="superlearner"`.

Super Learner combines logistic regression, random forest, Extra Trees, XGBoost, LightGBM and CatBoost. Within each outer patient cross-fit training set, five inner patient-grouped folds produce out-of-fold probabilities. Nonnegative weights summing to one minimise Brier loss, then all six learners are refitted on that outer training set. Imputation and scaling are fitted separately within each inner training fold. The inner fold count is reduced for small patient sets; single-class or unusable-feature inner fits use training-only Laplace-smoothed probabilities. CatBoost runs on CPU and XGBoost retains the existing CUDA training configuration. Install the updated `requirements.txt` before selecting Super Learner. Nested fitting also applies to learning curves and refitted-model bootstrap, so these runs take longer than a standalone learner. `superlearner_weights_by_fold.csv` reports learner weights and inner Brier scores for each task and outer fold. Feature importance is reported as unavailable for the ensemble because the six learners' importance scales differ.

“For semi-synthetic validation, to compare an estimated policy outcome with its known true value, that policy must be included in both `build_policy_panels.py` and `create_semi_synthetic_panel.py`.

1. `create_data_panel.py` — Builds catheter episodes, cleans chart covariates, and creates the modelling panel from raw MIMIC-IV data.
2. `create_semi_synthetic_panel.py` — Generates three semi-synthetic validation panels, records known row-level probabilities, and calculates known true CAUTI policy risks.
3. `fit_nuisance_models.py` — Fits nuisance models for the selected MODEL_TYPEs and produces cross-fitted predictions and diagnostics for its selected panels.
4. `panel_analysis.py` — Creates descriptive summaries and statistical comparisons of the real-data modelling panel, using saved nuisance-model features. The launcher runs it immediately after nuisance fitting.
5. `build_policy_panels.py` — Builds catheter-removal policy panels and checks for its selected panels.
6. `evaluate_gformula_policies.py` — Estimates policy outcomes for the selected panels using the g-formula, with the enabled bootstrap modes and diagnostics.
7. `evaluate_ipw_policies.py` — Estimates its selected panels using inverse probability weighting with the enabled bootstrap modes and diagnostics.
8. `evaluate_aipw_policies.py` — Estimates its selected panels using augmented inverse probability weighting with the enabled bootstrap modes and diagnostics.
9. `check_policy_results.py` — Checks the real-data policy panels and consistency of the g-formula, IPW and AIPW results for the selected bootstrap mode.

Nuisance fitting (step 3) and policy-panel construction (step 5) can run in either order once the selected panels exist. Panel analysis (step 4) requires nuisance fitting.
Steps 6–8 can run in any order once their selected data panels and policy panels exist. Fixed-model mode also requires nuisance fitting (step 3); refitted-model mode does not require saved models or predictions.

Each panel’s `counterfactual_policies/panels` folder contains one CSV per policy. These describe the `policy_action` under the hypothetical policy for each row in the observed data panel. `counterfactual_policies/policy_panels.csv` indexes the policies.

`policy_panel_checks.csv` summarises the counterfactual actions versus the observed actions for each policy. `n_applicable_policy_keep_rows` and `n_applicable_policy_remove_rows` count actions on observed decision rows; `n_policy_remove_rows` counts all hypothetical removal rows. `proportion_policy_matches_observed_action_today` is a fraction between 0 and 1.

`known_truth_policy_values.csv` contains known CAUTI policy risks; truth-only fields stay separate from estimator inputs.

Each evaluator configures `N_BOOTSTRAP=1000`, `BOOTSTRAP_SEED` and `REFIT_CROSSFIT_FOLDS` locally. To estimate confidence intervals it uses `N_BOOTSTRAP` per panel and mode (refitting retrains nuisance models for each patient bootstrap sample). Set `BOOTSTRAP_MODES=("fixed",)` for fixed-model bootstrap, `("refit",)` for refitted-model bootstrap, or `("fixed", "refit")` for both in each evaluator.

For each panel and estimator, fixed-model results and diagnostics are saved in `policy_eval/<estimator>/fixed/`; refitted-model results and diagnostics are saved in `policy_eval/<estimator>/refit/`. Each run folder contains its own `confidence_intervals/` subfolder.

`check_policy_results.py` checks real-data results for the mode selected by its `BOOTSTRAP_MODE` setting (default `"fixed"`) and writes `policy_eval/policy_results_checks_<mode>.csv`. Missing or invalid diagnostic counts fail the checks; no particular removal day is required.

`panel_analysis.py` provides a statistical descriptive analysis of `data/modelling_panel.csv`. It reads outcome feature rankings and saved propensity models from `artefacts/nuisance_models/xgboost` (configurable with `NUISANCE_MODEL_TYPE`).

Run `./main_pipeline.ps1` in PowerShell to execute the main run order above using `python` from your active environment. Activate `PhDResearch` before running it. The launcher resolves Python from PATH, prints its path, and stops immediately if Python is unavailable or any script fails.

## Shared import modules

- `policy_eval_common.py` — Provides shared panel validation, prediction, and reporting helpers.
- `policy_bootstrap_common.py` — Computes patient-level bootstrap confidence intervals, optionally refitting nuisance models.

## Tests

- `test_terminal_panel.py` — Checks terminal-event timing, episode boundaries, and CAUTI risk flags.
- `test_nuisance_cauti_risk_set.py` — Checks that CAUTI nuisance models use the correct training risk set.
- `test_nuisance_panel_runs.py` — Checks all-panel fitting, separate output paths, and normal errors when reading missing inputs.
- `test_policy_confidence_intervals.py` — Checks patient-clustered bootstrap estimates and confidence intervals.
- `test_policy_bootstrap_refitting.py` — Checks nuisance refitting, patient clustering, and evaluator entry points.

Run all tests independently of the pipeline with `python -m unittest discover`.
