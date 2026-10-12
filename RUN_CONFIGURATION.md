# Run configuration

Edit `pipeline_config.py` in this worktree to choose what runs. Individual Python scripts and the pipeline launcher read this same file at startup. Restart a running script after changing it.

The config contains only run choices. Paths are derived in `pipeline_runtime.py`; model hyperparameters, training-device settings, random seeds, bootstrap replicate counts, fold counts, simulation coefficients, cleaning thresholds and column definitions remain in their implementation modules.

| Choice | Purpose |
| --- | --- |
| `PIPELINE_STAGES` | Ordered script list for the launcher; comment out stages to skip them. An empty list runs nothing. |
| `MODEL_TYPES` | Ordered learners to fit. An empty list skips fitting. |
| `RUN_LEARNING_CURVES` | `True` or `False`. |
| `SUPERLEARNER_BASE_MODELS` | Learners included in the ensemble, independently of standalone fitting. |
| `NUISANCE_PANELS` | Panels to fit nuisance models on. |
| `POLICY_PANELS` | Panels to build policy definitions for. |
| `EVALUATION_PANELS` | Panels for all three evaluators. |
| `NUISANCE_MODEL_TYPE` | Single learner used by evaluators and descriptive analysis. |
| `BOOTSTRAP_MODES` | `fixed`, `refit`, or both. An empty list skips evaluation runs. |
| `POLICY_DAYS` | Removal-day policies shared by policy construction and known-truth calculations. |
| `ANALYSIS_PANEL` | Single panel for descriptive analysis. |
| `ANALYSIS_INCLUDE_AGE_SPLIT` | Enables or disables age-stratified reports. |
| `RESULT_CHECK_PANEL` | Single panel for the cross-estimator audit. |
| `RESULT_CHECK_BOOTSTRAP_MODE` | `fixed` or `refit` reports to audit. |

Supported panel names are `real`, `semi_synthetic_with_confounding`, `confounder_omitted`, and `randomised_action`. Supported learners are `logistic_regression`, `random_forest`, `extra_trees`, `xgboost`, `lightgbm`, `catboost`, and `superlearner`. The ensemble library cannot contain itself.

AIPW uses Hájek normalization for its residual correction and bootstrap confidence intervals.

The existing run choices were preserved: XGBoost, learning curves off, real-only nuisance fitting, all four policy/evaluation panels, all nine launcher stages, fixed bootstrap, and real-data analysis/audit. For fixed bootstrap, saved nuisance models and predictions must already exist for every evaluation panel; include those panels in `NUISANCE_PANELS` when fitting them in this run. The result audit requires outputs from all three estimators. Skipping a producer stage assumes its required outputs already exist. Running the semi-synthetic stage creates all three validation variants; the panel lists choose which variants downstream stages process.

## Running

Activate `PhDResearch`, then run the selected stages:

```powershell
python .\run_pipeline.py
```

The Python runner executes the configured scripts in order using the active Python interpreter and stops if a script fails.

An individual script can still be run with `python .\fit_nuisance_models.py`, for example. `PIPELINE_STAGES` controls the launcher only; direct execution runs the requested script using its centrally configured choices.

## Output routing

The sole panel builder uses the original `makic` timing and the normal `data/` and `artefacts/` folders. Semi-synthetic datasets live in `artefacts/semi-synthetic_validation/semi_synthetic_measured_confounding/`; each variant's downstream runs live in its `pipeline_runs/<panel>/` subfolder. All stages use the same real panel's `data/covariate_dictionary.csv`.

There is no panel-structure selector. Previously generated experiment outputs are left untouched and are no longer selected by the pipeline. Policy assignment and evaluation retain their existing behavior pending the separate work on explicit removal timing and interval duration.

## QA

```powershell
python -m unittest test_pipeline_config test_base_panel test_terminal_panel
python -m unittest discover
```

The config tests change selections in memory and check all nine stage modules, baseline paths, empty selections, invalid choices, stage order and stopping after failure. They also exercise the actual downstream entry points and evaluator input/output folder setup for all four panels and both bootstrap modes. They do not run MIMIC processing or full model fitting.
