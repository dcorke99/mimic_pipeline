# MIMIC catheter policy pipeline

Use a Python environment with `requirements.txt` installed; run each script with `python <script.py>` or VS Code's Run Python File.
Keep the project layout together; raw MIMIC-IV data defaults to `../Data/MIMIC-IV/mimic-iv-3.1`, overridable with `MIMIC_DIR`.

## Main run order

For a selected ML model, this workflow generates **4 panels × 3 estimators × 2 nuisance modes = 24 combinations** without changing dataset or bootstrap-mode settings.
The panels are real data, semi-synthetic measured confounding, confounder omitted, and randomised actions; both modes produce bootstrap confidence intervals.
Match fitting's `MODEL_TYPE` to each evaluator's `NUISANCE_MODEL_TYPE`; fitting with `MODEL_TYPE="all"` trains every learner, but each evaluator still uses only its selected learner.

1. `create_data_panel.py` — Builds catheter episodes, cleans chart covariates, and creates the modelling panel from MIMIC-IV.
2. `create_semi_synthetic_panel.py` — Generates the three validation panels and oracle truth from the observed modelling panel.
3. `fit_nuisance_models.py` — Fits separate nuisance models, cross-fitted predictions, and diagnostics for all four panels (`MODEL_TYPE="all"` fits every learner).
4. `build_policy_intervention_panels.py` — Builds catheter-removal policy panels and quality checks for all four datasets.
5. `evaluate_gformula_policies.py` — Estimates all four panels using the g-formula with fixed-model and refitted-model bootstrap inference.
6. `evaluate_ipw_policies.py` — Estimates all four panels using inverse probability weighting with both bootstrap modes and diagnostics.
7. `evaluate_aipw_policies.py` — Estimates all four panels using augmented inverse probability weighting with both bootstrap modes and diagnostics.
8. `audit_policy_evaluation_baseline.py` — Checks consistency of real-data saved-model policy panels and outputs from all three estimators.
9. `prepare_results_plots.py` — Produces comparison plots and a summary table for the real-data saved-model evaluation and audit outputs.

Steps 3–4 can run in either order once all four panels exist; steps 5–7 can run in any order after both are complete.
Each evaluator uses `N_BOOTSTRAP=1000` per panel and mode; refitting retrains nuisance models for each patient bootstrap sample.
Each evaluator's `main()` contains its own panel and bootstrap-mode loops. To run only the fixed bootstrap, comment out the `True,` line in that evaluator's `for refit in (...)` loop.
The scripts read inputs directly: missing files or required columns raise their normal Python or pandas errors at the point of use.
All-panel loops leave their module settings configured for the final panel and mode; they do not save and restore global settings.
Saved-model results use each panel's `policy_eval/<estimator>` folder, with intervals in `confidence_intervals/fixed`.
Refitted results use `policy_eval/<estimator>/refit_nuisance`, with intervals in `confidence_intervals/refit` beneath it.
The final audit and plots cover only real-data saved-model results, rather than all 24 combinations.

## PowerShell launcher

- `main_pipeline.ps1` — Creates all four panels, fits nuisance models, builds policies, runs both estimation modes, and creates the real-data saved-model audit and plots.

Run `./main_pipeline.ps1` in PowerShell to execute the main run order above using the project's `.venv` interpreter.

## Optional scripts

- `audit_duplicate_episode_days.py` — Audits duplicate episode-day rows; run after creating the modelling panel.
- `panel_analysis.py` — Creates descriptive summaries, plots, and hypothesis tests; requires the panel and existing `artefacts/step1` feature tables.
- `panel_analysis+.py` — Runs the same panel analysis with additional median-split age tests; use instead of `panel_analysis.py` when wanted.

## Shared modules (imported automatically)

- `policy_eval_common.py` — Provides shared panel validation, prediction, and reporting helpers.
- `policy_bootstrap.py` — Computes patient-level bootstrap confidence intervals, optionally refitting nuisance models.

## Tests

- `test_terminal_panel.py` — Checks terminal-event timing, episode boundaries, and CAUTI risk flags.
- `test_nuisance_cauti_risk_set.py` — Checks that CAUTI nuisance models use the correct training risk set.
- `test_nuisance_panel_runs.py` — Checks all-panel fitting, separate output paths, and normal errors when reading missing inputs.
- `test_policy_confidence_intervals.py` — Checks patient-clustered bootstrap estimates and confidence intervals.
- `test_policy_bootstrap_refitting.py` — Checks nuisance refitting, patient clustering, and evaluator entry points.

Run all tests independently of the pipeline with `python -m unittest discover`.
