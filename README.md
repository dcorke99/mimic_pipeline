# MIMIC catheter policy pipeline

`requirements.txt` defines the python environment

Edit [pipeline_config.py](pipeline_config.py) for all run choices. See [Run configuration](RUN_CONFIGURATION.md) for available selections. Run the selected stages with `python run_pipeline.py`, or run an individual script with `python <script.py>` or VS Code.

Raw MIMIC-IV data defaults to `../Data/MIMIC-IV/mimic-iv-3.1`, overridable with `MIMIC_DIR`.

The panel builder uses the original `makic` timing: IN windows start at insertion, end at removal or the terminal event, and OUT windows restart at removal. Final windows may be shorter than 24 hours.

## Main run order

The pipeline is run on separate data panels representing real data, semi-synthetic measured confounding, confounder omitted, and randomised actions. It supports 3 estimators (g-formula, IPW and AIPW) each with 2 confidence interval bootstrap modes (fixed-model and refit-model). For each selected learner this workflow generates up to **4 panels × 3 estimators × 2 bootstrap modes = 24 combinations**.

All run choices live in `pipeline_config.py`. Shared paths are listed in the same file. Model hyperparameters, seeds, fold counts, simulation coefficients and cleaning settings stay in implementation modules.

Select stages with `PIPELINE_STAGES`; comment out scripts to skip them. Select panels independently with `NUISANCE_PANELS`, `POLICY_PANELS` and `EVALUATION_PANELS`. These lists contain panel names only; the `PANEL_PATHS` table lists the baseline data and artefact locations. `ANALYSIS_PANEL` and `RESULT_CHECK_PANEL` select the panel for descriptive analysis and the result audit.

Select learners with `MODEL_TYPES`; comment or uncomment entries to include them. For example, `MODEL_TYPES = ["xgboost", "superlearner"]` runs those learners in order for each nuisance panel. An empty list skips fitting. `RUN_LEARNING_CURVES` enables or disables learning curves. `NUISANCE_MODEL_TYPE` chooses the learner for all evaluators and descriptive analysis; `BOOTSTRAP_MODES` selects fixed, refit, or both for all evaluators. AIPW uses Hájek normalization for its residual correction.

For fixed-model bootstrap, each evaluator’s `NUISANCE_MODEL_TYPE` must refer to a learner previously fitted for the selected panels by `fit_nuisance_models.py`.

Refitted-model bootstrap fits the selected learner during evaluation and can run without saved models or predictions.

Implemented nuisance models are logistic regression, random forest, Extra Trees, XGBoost, LightGBM, CatBoost and Super Learner. Select Extra Trees with `"extra_trees"` and CatBoost with `"catboost"` in `MODEL_TYPES`; each has its own output folder and can also be selected with `NUISANCE_MODEL_TYPE` in the evaluators. Set `MODEL_TYPES = ["superlearner"]` to fit only the ensemble, or include additional names in the list to fit other learners too. Its outputs use the `superlearner` folder and the same prediction columns, diagnostics and saved-model formats; evaluators can select it with `NUISANCE_MODEL_TYPE="superlearner"`. The ensemble's six-member library is configured separately in `SUPERLEARNER_BASE_MODELS`, independently of which standalone models are selected.

Super Learner combines logistic regression, random forest, Extra Trees, XGBoost, LightGBM and CatBoost. Within each outer patient cross-fit training set, five inner patient-grouped folds produce out-of-fold probabilities. Nonnegative weights summing to one minimise Brier loss, then all six learners are refitted on that outer training set. Imputation and scaling are fitted separately within each inner training fold. The inner fold count is reduced for small patient sets; single-class or unusable-feature inner fits use training-only Laplace-smoothed probabilities. Base learners share the settings in `LEARNER_CONFIGURATIONS` with their standalone versions, including the CatBoost and XGBoost training-device settings. Install the updated `requirements.txt` before selecting Super Learner. Nested fitting also applies to learning curves and refitted-model bootstrap, so these runs take longer than a standalone learner. `superlearner_weights_by_fold.csv` reports learner weights and inner Brier scores for each task and outer fold. Feature importance is reported as unavailable for the ensemble because the six learners' importance scales differ.

For semi-synthetic validation, policy construction and known-truth calculations use the same `POLICY_DAYS` selection from `pipeline_config.py`. The semi-synthetic builder continues to generate all three validation datasets when its stage is run.

1. `create_data_panel.py` — Builds catheter episodes, cleans chart covariates, and creates the modelling panel from raw MIMIC-IV data.
2. `create_semi_synthetic_panel.py` — Generates three semi-synthetic validation panels, records known row-level probabilities, and calculates known true CAUTI policy risks.
3. `fit_nuisance_models.py` — Fits the learners listed in `MODEL_TYPES` and produces cross-fitted predictions and diagnostics for its selected panels.
4. `panel_analysis.py` — Creates descriptive summaries and statistical comparisons of `ANALYSIS_PANEL`, using saved nuisance-model features. Run it after nuisance fitting.
5. `build_policy_panels.py` — Builds catheter-removal policy panels and checks for its selected panels.
6. `evaluate_gformula_policies.py` — Estimates policy outcomes for the selected panels using the g-formula, with the enabled bootstrap modes and diagnostics.
7. `evaluate_ipw_policies.py` — Estimates its selected panels using inverse probability weighting with the enabled bootstrap modes and diagnostics.
8. `evaluate_aipw_policies.py` — Estimates its selected panels using augmented inverse probability weighting with the enabled bootstrap modes and diagnostics.
9. `check_policy_results.py` — Checks `RESULT_CHECK_PANEL` and consistency of the g-formula, IPW and AIPW results for `RESULT_CHECK_BOOTSTRAP_MODE`.

Nuisance fitting (step 3) and policy-panel construction (step 5) can run in either order once the selected panels exist. Panel analysis (step 4) requires nuisance fitting.
Steps 6–8 can run in any order once their selected data panels and policy panels exist. Fixed-model mode also requires nuisance fitting (step 3); refitted-model mode does not require saved models or predictions.

Each panel’s `counterfactual_policies/panels` folder contains one CSV per policy. These describe the `policy_action` under the hypothetical policy for each row in the observed data panel. `counterfactual_policies/policy_panels.csv` indexes the policies.

`policy_panel_checks.csv` summarises the counterfactual actions versus the observed actions for each policy. `n_applicable_policy_keep_rows` and `n_applicable_policy_remove_rows` count actions on observed decision rows; `n_policy_remove_rows` counts all hypothetical removal rows. `proportion_policy_matches_observed_action_today` is a fraction between 0 and 1.

`known_truth_policy_values.csv` contains known CAUTI policy risks; truth-only fields stay separate from estimator inputs.

Each evaluator configures `N_BOOTSTRAP=1000`, `BOOTSTRAP_SEED` and `REFIT_CROSSFIT_FOLDS` locally. To estimate confidence intervals it uses `N_BOOTSTRAP` per panel and mode (refitting retrains nuisance models for each patient bootstrap sample). Set `BOOTSTRAP_MODES=["fixed"]` for fixed-model bootstrap, `["refit"]` for refitted-model bootstrap, or `["fixed", "refit"]` for both in `pipeline_config.py`.

For each panel and estimator, fixed-model results and diagnostics are saved in `policy_eval/<estimator>/fixed/`; refitted-model results and diagnostics are saved in `policy_eval/<estimator>/refit/`. Each run folder contains its own `confidence_intervals/` subfolder.

`check_policy_results.py` checks the panel and mode selected by `RESULT_CHECK_PANEL` and `RESULT_CHECK_BOOTSTRAP_MODE` in `pipeline_config.py`, and writes `policy_eval/policy_results_checks_<mode>.csv` under that panel's artefact directory. Missing or invalid diagnostic counts fail the checks; no particular removal day is required.

`panel_analysis.py` describes `ANALYSIS_PANEL` and reads that panel's saved nuisance features and propensity models for `NUISANCE_MODEL_TYPE`. `ANALYSIS_INCLUDE_AGE_SPLIT` controls age-stratified reports.

Run `python run_pipeline.py` to execute `PIPELINE_STAGES` from `pipeline_config.py`. Activate `PhDResearch` first. The runner executes the scripts in order and stops if a script fails.

Run selections are centralized in `pipeline_config.py`. The panel builder retains the original timing and risk-set definitions. Explicit elapsed-time policies and duration-aware evaluation remain a separate methodological update.

## Shared import modules

- `run_pipeline.py` — Executes only the centrally selected stages with the active interpreter.
- `policy_eval_common.py` — Provides shared panel validation, prediction, and reporting helpers.
- `policy_bootstrap_common.py` — Computes patient-level bootstrap confidence intervals, optionally refitting nuisance models.

## Tests

- `test_base_panel.py` — Checks baseline snapshot equality, removal/terminal boundaries, risk sets and covariate timing.
- `test_pipeline_config.py` — Checks config wiring across all stages, isolated paths, empty selections and launcher orchestration.
- `test_terminal_panel.py` — Checks terminal-event timing, episode boundaries, and CAUTI risk flags.
- `test_nuisance_cauti_risk_set.py` — Checks that CAUTI nuisance models use the correct training risk set.
- `test_nuisance_panel_runs.py` — Checks all-panel fitting, separate output paths, and normal errors when reading missing inputs.
- `test_policy_confidence_intervals.py` — Checks patient-clustered bootstrap estimates and confidence intervals.
- `test_policy_bootstrap_refitting.py` — Checks nuisance refitting, patient clustering, and evaluator entry points.

Run all tests independently of the pipeline with `python -m unittest discover`.
