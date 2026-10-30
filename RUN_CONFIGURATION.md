# Run configuration

All routine run choices are in `pipeline_config.py`.
Settings are grouped by the script that consumes them, with shared paths and
evaluation choices defined once at the top. Policy days and panels are in the
`build_policy_panels.py` section. The semi-synthetic builder also uses those
policy days and the horizons in the Policy-DML section.

| Setting | Purpose |
| --- | --- |
| `PIPELINE_STAGES` | Scripts run by `run_pipeline.py`, in order. |
| `MODEL_TYPES` | Nuisance learners fitted for each selected panel. |
| `SUPERLEARNER_BASE_MODELS` | Members of the Super Learner library. |
| `NUISANCE_PANELS` | Panels passed to nuisance fitting. |
| `POLICY_PANELS` | Datasets used to build policy inputs for both evaluators. |
| `EVALUATION_PANELS` | Panels evaluated by g-formula or Policy-DML. |
| `EVALUATION_MODEL_TYPES` | Nuisance learners evaluated separately by g-formula and used as behaviour learners by Policy-DML; empty skips both evaluators. |
| `NUISANCE_MODEL_TYPE` | Single nuisance learner used by descriptive panel analysis. |
| `BOOTSTRAP_MODES` | `fixed`, `refit`, both, or an empty list. |
| `POLICY_DAYS` | One-based catheter-removal days used to build policy panels and synthetic known-truth values. Both evaluators discover existing panel files. |
| `POLICY_DML_HORIZON_HOURS` | Policy-DML follow-up by panel: 720 hours for observed data and 336 hours for each semi-synthetic panel. |
| `POLICY_DML_CONTINUATION_LEARNERS` | List of continuation learners: `ridge`, `extra_trees`, `random_forest`, `xgboost`, or `catboost`; empty skips Policy-DML. |
| `POLICY_DML_N_BOOTSTRAP` | Number of Policy-DML bootstrap draws. |
| `GFORMULA_N_BOOTSTRAP` | Number of g-formula bootstrap draws. |
| `POLICY_DML_CLIPPING_SENSITIVITY` | Additional propensity-clipping values to evaluate; empty by default. |
| `ANALYSIS_PANEL` | Panel used by `panel_analysis.py`. |
| `ANALYSIS_INCLUDE_AGE_SPLIT` | Enables the optional age split in descriptive analysis. |

`PANEL_PATHS` maps each panel name to its input CSV and nuisance-model root.
`POLICY_DIRS` and `RESULT_DIRS` give the matching panel-specific intermediate
and final-result directories.
The semi-synthetic builder always creates all three validation variants when
that stage runs.

Select evaluation learners independently of the `MODEL_TYPES` fitting list:

```python
EVALUATION_MODEL_TYPES = ["catboost", "superlearner"]
POLICY_DML_CONTINUATION_LEARNERS = ["ridge", "xgboost"]
```

G-formula runs once per selected nuisance learner, panel and bootstrap mode.
Policy-DML runs every nuisance/continuation pair: this example produces four
Policy-DML runs per panel and bootstrap mode. Each run has its own models and
estimates; predictions from different selections are not combined. Fixed mode
requires saved nuisance predictions for each selected learner and panel;
g-formula also requires its saved outcome models. Refit mode uses the selected
nuisance and continuation learners inside each bootstrap draw.

Policy-DML reuses identical behaviour predictions across continuation choices
within the same invocation. The first continuation choice fits auxiliary
behaviour models for its nested patient splits; later choices reuse those
probabilities while fitting their own continuation models and recursive targets.
Refit mode shares behaviour predictions only within the corresponding bootstrap
draw with matching inputs and patient multiplicities. The cache holds
probabilities in memory and is cleared between panels/behaviour learners.

Results are separated by learner even if only one is selected:

```text
artefacts/results/<panel>/gformula/<nuisance>/<bootstrap>/
artefacts/results/<panel>/policy_dml/<nuisance>/<continuation>/<bootstrap>/
```

Previous results in estimator-level bootstrap directories remain where they
are. New runs write only to the model-specific directories above.

Build policy panels for every dataset selected in `EVALUATION_PANELS` before
running either evaluator. Existing policy panels made before the shared
`policy_dml_action` column was added must be rebuilt.
Both evaluators discover every existing CSV in
`artefacts/policy_panels/<panel>/panels/` and add current practice. Policy names
and removal days come from filenames such as `remove_on_day_3.csv`.
No `policy_panels.csv` index is created or read. `POLICY_DAYS` controls which
panels the builder writes; evaluation includes any panels left from earlier builds.

Policy-DML has no command-line run options. Select its panels, bootstrap modes,
horizons, learner lists, and bootstrap count in `pipeline_config.py`, then run:

```powershell
python evaluate_policy_dml_policies.py
```

The DML evaluator defines its technical settings near the top of
`evaluate_policy_dml_policies.py`:

- `SETTINGS`: five outer folds, two inner folds, seed, horizon and clipping.
- `LEARNER_CONFIGURATIONS`: separate parameter dictionaries for Ridge, Random
  Forest, Extra Trees, XGBoost and CatBoost.
- `LEARNER_PREPROCESSING`: imputation, standardisation and native missing-value
  handling for each learner.

The tree learners use 100 trees and all processors; bootstrap draw counts are
set independently with `GFORMULA_N_BOOTSTRAP` and `POLICY_DML_N_BOOTSTRAP`.
XGBoost uses squared-error loss and histogram trees with native
missing-value handling. Its configured device controls training (currently
CUDA); after fitting, the evaluator switches each model to CPU prediction to
match the pandas inputs. Ridge uses median imputation and standardisation; the
forests use median imputation.

Add `"catboost"` to `POLICY_DML_CONTINUATION_LEARNERS` to select
`CatBoostRegressor`. It uses RMSE loss, 100 boosting iterations, native missing
values, and GPU training by default, matching the existing GPU nuisance setup.
Set its `task_type` to `"CPU"` in `LEARNER_CONFIGURATIONS` for CPU training.
Constant targets are allowed; a stage with entirely constant predictors uses
the training mean. Predictions remain real-valued and are not clipped to
probabilities. The option is initially commented out to preserve selected runs.

The g-formula comparator still covers complete episodes. Match horizons before
comparing estimates.
