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

## Pipeline stages

1. `create_data_panel.py` builds `data/modelling_panel.csv`.
2. `create_semi_synthetic_panel.py` creates the measured-confounding,
   omitted-confounder, and randomised-action panels and their known truth.
3. `fit_nuisance_models.py` fits patient-grouped cross-fitted behaviour and
   outcome models for the selected panels and learners.
4. `panel_analysis.py` creates optional descriptive summaries.
5. `build_policy_panels.py` builds preserved-grid inputs for the g-formula
   comparator.
6. `evaluate_gformula_policies.py` evaluates the selected fixed-day policies.
7. `evaluate_policy_dml_policies.py` evaluates current practice and fixed-day
   policies with the recursive Policy-DML estimator.

Select stages, panels, learners, policies, and bootstrap modes in
`pipeline_config.py`. See [RUN_CONFIGURATION.md](RUN_CONFIGURATION.md) for the
available settings.

Policy-DML reads the modelling panel directly and creates fixed-day policies
from `POLICY_DAYS`. In fixed bootstrap mode it reuses the saved out-of-fold
behaviour probabilities. In refit mode it refits the behaviour and continuation
models within each patient bootstrap sample. Both evaluators print bootstrap
progress every five replicates.

The shared modelling panels, nuisance-model functions, and
`policy_inference.py` bootstrap reporting helpers form the interface for the
future LTMLE evaluator.

## Outputs

Nuisance outputs are written below each panel's `nuisance_models/<learner>/`
directory. They include out-of-fold predictions, fold assignments, fitted
models, performance metrics, and constant-feature records.

Policy-DML outputs are written below
`policy_eval/policy_dml/{fixed,refit}/`. They include policy summaries,
episode and decision scores, compact model ownership metadata, support
diagnostics, run metadata, bootstrap draws, and percentile confidence
intervals. See [POLICY_DML_IMPLEMENTATION.md](POLICY_DML_IMPLEMENTATION.md) for
the estimator-to-code mapping.

## Tests

Run the maintained test suite with:

```powershell
python -m unittest discover -p "test_*.py" -v
```

The tests cover panel timing and terminal events, nuisance risk sets and fold
isolation, Super Learner fitting, configuration wiring, recursive Policy-DML
arithmetic, leakage protection, patient bootstrapping, and end-to-end outputs.
