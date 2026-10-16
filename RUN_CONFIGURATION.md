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
`--horizon-hours` and `--propensity-clip`. The full Policy-DML bootstrap count
is set by `N_BOOTSTRAP` in `evaluate_policy_dml_policies.py`.
