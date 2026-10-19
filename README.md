# MIMIC catheter policy pipeline

This repository builds a longitudinal catheter panel, fits cross-fitted nuisance
models, and evaluates catheter-removal policies. Recursive Policy-DML is the
primary causal evaluator. G-formula remains as a comparator.

Install the packages in `requirements.txt`, edit `pipeline_config.py`, and run:

```powershell
python run_pipeline.py
```

Individual stages can also be run with `python <script.py>`. Raw MIMIC-IV data
defaults to `../Data/MIMIC-IV/mimic-iv-3.1`; set `MIMIC_DIR` to override it.

## Panel structure

The observed panel uses a fixed 24-hour grid anchored at catheter insertion.
The grid continues through removal and ends at death, reinsertion, or ICU exit.
`catheter_state` is the state at period start. IN periods are decision rows, and
the period containing removal has `observed_action="remove"`. Events on a
period boundary belong to the preceding period.

A removal period can contain both catheter-in and catheter-out time. The
`catheter_in_hours` and `catheter_out_hours` columns retain that split. CAUTI
follow-up continues for 48 elapsed hours after removal and ends after the first
CAUTI. Reinsertion fitting uses `reinsertion_remove_period` and
`reinsertion_periods_out`, which are also available to counterfactual policies.

Chart features use the interval
`[max(ICU admission, period start - 24 hours), period start)`, so measurements
at or after a decision time cannot enter that decision's predictors.

Semi-synthetic CAUTI generation retains only the first event in each episode
and ends its risk set afterwards. A source reinsertion endpoint becomes a
simulated reinsertion only if simulated removal precedes it. Otherwise the
episode ends at its explicitly recorded `simulation_end_time`, with reason
`simulation_horizon` and no invented clinical event. All source episodes remain
in the validation cohort.

## Pipeline stages

1. `create_observed_panel.py` builds `data/modelling_panel.csv`.
2. `create_semi_synthetic_panel.py` creates the measured-confounding,
   omitted-confounder, and randomised-action panels and their known truth.
3. `fit_nuisance_models.py` fits patient-grouped cross-fitted behaviour and
   outcome models for the selected panels and learners.
4. `panel_analysis.py` creates optional descriptive summaries.
5. `build_policy_panels.py` builds preserved-grid policy inputs for both
   evaluators.
6. `evaluate_gformula_policies.py` evaluates the selected fixed-day policies.
7. `evaluate_policy_dml_policies.py` evaluates current practice and fixed-day
   policies with the recursive Policy-DML estimator.

Select stages, panels, learners, policies, and bootstrap modes in
`pipeline_config.py`. See [RUN_CONFIGURATION.md](RUN_CONFIGURATION.md) for the
available settings.

Policy-DML reads the modelling panel and the saved fixed-day policy panels.
The policy panels must be built for the selected evaluation datasets before
running either evaluator. In fixed bootstrap mode Policy-DML reuses the saved out-of-fold
behaviour probabilities. In refit mode it refits the behaviour and continuation
models within each patient bootstrap sample. Both evaluators print bootstrap
progress every five replicates. Policy-DML also reports its point-estimate work
size and progress through the outer folds, inner folds, and policies.
Its configured horizons are 30 days for observed data and 14 days for the
semi-synthetic validation panels. To resume only the validation panels, select
those three entries in `EVALUATION_PANELS`, select the required mode in
`BOOTSTRAP_MODES`, and run:

```powershell
python evaluate_policy_dml_policies.py
```

The simulator exports matching CAUTI reference values in
`artefacts/validation/known_truth_policy_dml.csv`, with panel and horizon columns.

The shared modelling panels, nuisance-model functions, and
`policy_inference.py` bootstrap reporting helpers form the interface for the
future LTMLE evaluator.

## Outputs

Final estimates are under
`artefacts/results/<panel>/<estimator>/<bootstrap>/`. Start with
`policy_values.csv` and `policy_differences.csv`. Diagnostics and run settings
sit beside them.

Fitted nuisance models are under
`artefacts/nuisance_models/<panel>/<learner>/`, shared policy panels are
under `artefacts/policy_panels/<panel>/`, and simulated panels and known truth
are under `artefacts/validation/`. See [ARTEFACTS.md](ARTEFACTS.md) for the full
map.

## Tests

Run the maintained test suite with:

```powershell
python -m unittest discover -p "test_*.py" -v
```

The tests cover panel timing and terminal events, nuisance risk sets and fold
isolation, Super Learner fitting, configuration wiring, recursive Policy-DML
arithmetic, leakage protection, patient bootstrapping, and end-to-end outputs.
