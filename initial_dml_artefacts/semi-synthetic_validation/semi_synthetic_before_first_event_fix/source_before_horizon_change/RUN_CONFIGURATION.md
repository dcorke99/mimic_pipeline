# Run configuration

All routine run choices are in `pipeline_config.py`.

| Setting | Purpose |
| --- | --- |
| `PIPELINE_STAGES` | Scripts run by `run_pipeline.py`, in order. |
| `MODEL_TYPES` | Nuisance learners fitted for each selected panel. |
| `SUPERLEARNER_BASE_MODELS` | Members of the Super Learner library. |
| `NUISANCE_PANELS` | Panels passed to nuisance fitting. |
| `POLICY_PANELS` | Panels used to build g-formula policy inputs. |
| `EVALUATION_PANELS` | Panels evaluated by g-formula or Policy-DML. |
| `NUISANCE_MODEL_TYPE` | Learner consumed by evaluators. |
| `BOOTSTRAP_MODES` | `fixed`, `refit`, both, or an empty list. |
| `POLICY_DAYS` | One-based catheter-removal days to evaluate. |
| `ANALYSIS_PANEL` | Panel used by `panel_analysis.py`. |
| `ANALYSIS_INCLUDE_AGE_SPLIT` | Enables the optional age split in descriptive analysis. |

`PANEL_PATHS` maps each panel name to its input CSV and nuisance-model root.
The semi-synthetic builder always creates all three validation variants when
that stage runs.

The Policy-DML evaluator also accepts command-line overrides:

```powershell
python evaluate_policy_dml_policies.py --panel real --bootstrap-mode fixed --n-bootstrap 5 --max-patients 100 --output-root artefacts/policy_dml_smoke --continuation-learner ridge
```

`--max-patients` requires a separate `--output-root`. Other options include
`--horizon-hours`, `--propensity-clip`, `--inner-folds`,
`--continuation-trees`, and `--continuation-jobs`. Policy-DML defaults to five
outer folds inherited from saved nuisance predictions, two inner folds, 100
Extra Trees per continuation model using all processors, and 100 bootstrap
draws. Its evaluation horizon is 30 days from insertion (720 hours); the existing
g-formula comparator still covers complete episodes. Match horizons before
comparing their estimates. These defaults are set near the top of
`evaluate_policy_dml_policies.py`.
