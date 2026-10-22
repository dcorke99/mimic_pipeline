# Recursive longitudinal Policy-DML

`evaluate_policy_dml_policies.py` implements the recursive estimator described
in CoR Section 3.5 and Appendix B.2. It reads the modelling panel, shared
fixed-day policy panels, and nuisance predictions. Decision preparation,
continuation fitting, recursive scoring, diagnostics and bootstrap reporting
are defined in this evaluator. Settings and policy definitions use dictionaries.

## Estimator mapping

| Quantity | Code |
| --- | --- |
| `Q_t(S,a) = E[R_t + psi_(t+1) | S,A=a]` | `fit_continuation` fits real-valued action-specific continuation regressions. |
| `Qbar_t = sum_a pi(a|S) Q_t(S,a)` | `recursive_step` forms the deterministic or stochastic policy mixture. |
| `w_t = pi(A|S) / g(A|S)` | `ratio_quantities` uses the cross-fitted behaviour probability. |
| `psi_t = Qbar_t + w_t(R_t + psi_(t+1) - Q_t(S,A))` | `recursive_step`, applied backwards by `crossfit_policy_scores`. |
| `psi_(T+1) = 0` | The final linked decision has a terminal continuation value of zero. |
| `V(pi) = mean_e psi_(e,1)` | `episode_scores` selects the first score and `policy_summary` averages episodes. |
| `Delta = V(pi) - V(current)` | Summary and bootstrap contrasts use current practice as the paired comparator. |

The continuation learners are Ridge, Random Forest, Extra Trees and XGBoost.
Tree learners use 100 trees and all available processors. Select `"xgboost"`
in `POLICY_DML_CONTINUATION_LEARNER` to use squared-error regression with
histogram trees. XGBoost trains on its configured device (currently CUDA), then
switches the fitted model to CPU prediction for the pandas inputs, as in the
nuisance fitting script. Each learner's parameters are in the evaluator's
`LEARNER_CONFIGURATIONS` dictionary, following the nuisance fitting script's
layout. `LEARNER_PREPROCESSING` controls imputation and standardisation. XGBoost
uses native missing-value handling; Ridge uses median imputation and
standardisation; the forests use median imputation. Scores are not clipped
to `[0, 1]`.

## Cross-fitting and nuisance models

Patients, including all their episodes, stay in one outer fold. The saved
nuisance predictions provide five outer folds. For each outer fold, two inner
patient folds produce out-of-fold recursive scores for the outer
continuation targets. The final outer models then score only the held-out outer
patients. The model audit records model IDs, training folds, held-out fold,
features, policy, outcome, and period without exporting patient lists.

Fixed evaluation loads `p_remove_obs` and its fold from the selected learner's
`nuisance_predictions.csv`. Refit evaluation fits behaviour and continuation
models inside each bootstrap draw. Saved prediction loading checks the required
schema, probability bounds, and one-fold-per-patient rule.

`prepare_decisions` links each decision to its next decision and attaches later
OUT-state outcome contributions to the last decision. It rejects broken episode
ordering, decisions after removal, invalid terminal events, recurrent CAUTI,
and unexplained censoring. CAUTI uses the 48-hour post-removal window.

Current practice is the cross-fitted stochastic behaviour policy
`pi_current(a|S)=g_hat(a|S)`. Fixed-day policies remove at the first available
IN decision on or after the requested day. The shared policy panels provide
these instructions in `policy_dml_action`, including on observed IN rows after
the requested day. The g-formula `policy_action` instead describes the policy
trajectory, which is OUT after removal. Current practice uses the fitted
behaviour probabilities.

The configured evaluation horizon is 30 days (720 hours) for the observed panel and
14 days (336 hours) for each semi-synthetic panel. Decisions and linked outcomes
after the corresponding boundary do not contribute. Complete-episode g-formula
estimates need the same cutoff before comparison. The simulator exports matching
semi-synthetic CAUTI reference values by panel in
`known_truth_policy_dml.csv`, including the horizon. Before fitting, the
evaluator checks that every required period has training trajectories in each
outer and inner split, so unsupported tails fail before expensive model fits.
Run choices (panels, bootstrap modes, horizons, continuation learner and
bootstrap count) are read from `pipeline_config.py`. Technical fitting settings
are in the evaluator's `SETTINGS`, `LEARNER_CONFIGURATIONS` and
`LEARNER_PREPROCESSING` dictionaries.

Semi-synthetic episodes can also end at an explicitly recorded fixed
`simulation_end_time`. A `simulation_horizon` endpoint must match that schedule
and have no clinical terminal event. It represents the preserved simulation
follow-up grid, rather than unmodelled loss to follow-up. Synthetic CAUTI and
its risk flags stop after the first event; synthetic reinsertion requires a
preceding synthetic removal.

## Bootstrap inference

Fixed mode resamples patient-level sums of fitted episode scores. Refit mode
resamples patients, preserves each patient's multiplicity and fold ownership,
and refits all Policy-DML nuisance models. Policy values and differences use the
same patient draw. `policy_inference.py` contains the estimator-independent
paired-difference and percentile-interval reporting used here and available to
the future LTMLE evaluator.

Before fitting, the evaluator reports the patient, decision, stage, fold,
policy, learner, processor, and approximate model-fit counts. Point-estimate
progress is reported by outer fold, inner fold, and policy. Bootstrap progress
is printed every five replicates and at completion.

## Outputs

Files below `artefacts/results/<panel>/policy_dml/<fixed or refit>/` are:

| File | Content |
| --- | --- |
| `policy_values.csv` | Point estimates and percentile intervals for each policy value. |
| `policy_differences.csv` | Point estimates and percentile intervals for contrasts with current practice. |
| `diagnostics.csv` | Support, weights, clipping, agreement, and effective sample sizes. |
| `run_settings.json` | Estimator settings, selected continuation parameters and preprocessing, seeds, horizon, and saved-behaviour source. |
| `clipping_sensitivity.csv` | Optional results for configured clipping alternatives. |

Episode scores, decision scores, individual bootstrap draws, and per-model
audit records are used during estimation but are not saved. They duplicated
the result tables or produced very large files without being pipeline inputs.

## Running and testing

```powershell
python evaluate_policy_dml_policies.py
python -m unittest test_policy_dml -v
```

`test_policy_dml.py` contains hand-calculated recursion checks, trajectory and
risk-set checks, patient leakage tests, fixed and refit bootstrap checks, and an
end-to-end output test.

After correcting the synthetic outcomes, saved removal probabilities can be
reused when actions, predictors, and fold ownership are unchanged. Policy-DML
fits fresh continuation models from the corrected panels. Existing synthetic
outcome models and g-formula results require refitting before reuse.
