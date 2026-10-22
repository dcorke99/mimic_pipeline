# Run configuration

All routine run choices are in `pipeline_config.py`.

| Setting | Purpose |
| --- | --- |
| `PIPELINE_STAGES` | Scripts run by `run_pipeline.py`, in order. |
| `MODEL_TYPES` | Nuisance learners fitted for each selected panel. |
| `SUPERLEARNER_BASE_MODELS` | Members of the Super Learner library. |
| `NUISANCE_PANELS` | Panels passed to nuisance fitting. |
| `POLICY_PANELS` | Datasets used to build policy inputs for both evaluators. |
| `EVALUATION_PANELS` | Panels evaluated by g-formula or Policy-DML. |
| `NUISANCE_MODEL_TYPE` | Learner consumed by evaluators. |
| `BOOTSTRAP_MODES` | `fixed`, `refit`, both, or an empty list. |
| `POLICY_DAYS` | One-based catheter-removal days to evaluate. |
| `POLICY_DML_HORIZON_HOURS` | Policy-DML follow-up by panel: 720 hours for observed data and 336 hours for each semi-synthetic panel. |
| `POLICY_DML_CONTINUATION_LEARNER` | Continuation learner: `ridge`, `extra_trees`, `random_forest`, or `xgboost`. |
| `POLICY_DML_N_BOOTSTRAP` | Number of Policy-DML bootstrap draws. |
| `POLICY_DML_CLIPPING_SENSITIVITY` | Additional propensity-clipping values to evaluate; empty by default. |
| `ANALYSIS_PANEL` | Panel used by `panel_analysis.py`. |
| `ANALYSIS_INCLUDE_AGE_SPLIT` | Enables the optional age split in descriptive analysis. |

`PANEL_PATHS` maps each panel name to its input CSV and nuisance-model root.
`POLICY_DIRS` and `RESULT_DIRS` give the matching panel-specific intermediate
and final-result directories.
The semi-synthetic builder always creates all three validation variants when
that stage runs.

Build policy panels for every dataset selected in `EVALUATION_PANELS` before
running either evaluator. Existing policy panels made before the shared
`policy_dml_action` column was added must be rebuilt.

Policy-DML has no command-line run options. Select its panels, bootstrap modes,
horizons, continuation learner, and bootstrap count in `pipeline_config.py`, then run:

```powershell
python evaluate_policy_dml_policies.py
```

The DML evaluator defines its technical settings near the top of
`evaluate_policy_dml_policies.py`:

- `SETTINGS`: five outer folds, two inner folds, seed, horizon and clipping.
- `LEARNER_CONFIGURATIONS`: separate parameter dictionaries for Ridge, Random
  Forest, Extra Trees and XGBoost.
- `LEARNER_PREPROCESSING`: imputation, standardisation and native missing-value
  handling for each learner.

The tree learners use 100 trees and all processors; the run uses 100 bootstrap
draws. XGBoost uses squared-error loss and histogram trees with native
missing-value handling. Its configured device controls training (currently
CUDA); after fitting, the evaluator switches each model to CPU prediction to
match the pandas inputs. Ridge uses median imputation and standardisation; the
forests use median imputation.

The g-formula comparator still covers complete episodes. Match horizons before
comparing estimates.
