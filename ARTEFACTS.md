# Artefact guide

The artefacts directory is organised by purpose and then by panel. The same
panel names are used throughout: `observed`, `semi_synthetic_with_confounding`,
`confounder_omitted`, and `randomised_action`.

## Results to inspect

Final policy estimates are here:

```text
artefacts/results/<panel>/<estimator>/<bootstrap>/
```

Each g-formula or Policy-DML result directory contains:

| File | Use |
| --- | --- |
| `policy_values.csv` | Main policy estimates and confidence intervals. |
| `policy_differences.csv` | Main contrasts with current practice and confidence intervals. |
| `diagnostics.csv` | Checks needed to interpret the estimates. |

Policy-DML also writes `run_settings.json`. It writes
`clipping_sensitivity.csv` only when clipping sensitivity values are selected
in `pipeline_config.py`.

The two `policy_*.csv` files are the important results. Point estimates appear
in their `point_estimate` column, so a separate summary file is unnecessary.

## Nuisance models

```text
artefacts/nuisance_models/<panel>/<learner>/
```

The fitted model files and `nuisance_predictions.csv` are computational inputs
to the evaluators. The remaining CSV files report performance, calibration,
fold performance, and feature importance. Super Learner adds its weights file.

## Policy panels

```text
artefacts/policy_panels/<panel>/
```

`policy_panels.csv` is the index read by both evaluators. The `panels`
subdirectory contains one preserved-grid panel for each fixed removal day.
`policy_action` describes the g-formula policy trajectory. `policy_dml_action`
also specifies what to do on observed decision rows after the target day: remove
the catheter if it is still in. Current practice uses the fitted behaviour
probabilities rather than a fixed-day policy panel.

## Semi-synthetic validation data

```text
artefacts/validation/
```

This directory contains the three simulated panels, row-level truth,
simulation coefficients, and separate known-truth tables for the full-episode
g-formula estimand and the horizon-limited Policy-DML estimand.

## Descriptive analysis

```text
artefacts/panel_analysis/
```

These numbered tables and the single event-rate figure describe the panel.
They are reports rather than inputs to later stages.

Older runs may still contain the previous `policy_eval` and deeply nested
`semi-synthetic_validation` directories. Current scripts do not add to those
locations. They can be removed after any results that need to be retained have
been copied out.
