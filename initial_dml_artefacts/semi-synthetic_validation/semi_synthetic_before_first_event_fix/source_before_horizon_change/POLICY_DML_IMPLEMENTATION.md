# Recursive longitudinal Policy-DML

`evaluate_policy_dml_policies.py` implements the recursive estimator described
in CoR Section 3.5 and Appendix B.2. It operates directly on the same modelling
panels and nuisance-model conventions used elsewhere in the pipeline.

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

The default continuation learner is Extra Trees with 100 trees using all
available processors; Ridge and Random Forest are available. Scores are not
clipped to `[0, 1]`.

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
IN decision on or after the requested day. `configured_policies` builds them
directly from `pipeline_config.POLICY_DAYS`.

The default evaluation horizon is 30 days (720 hours) from insertion. Decisions
and linked outcomes after that boundary do not contribute. This is a 30-day
policy value; complete-episode g-formula estimates and known-truth files need
the same cutoff before they can be compared directly. Before fitting, the
evaluator checks that every required period has training trajectories in each
outer and inner split, so unsupported tails fail before expensive model fits.

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

Files below `policy_eval/policy_dml/{fixed,refit}/` are:

| File | Content |
| --- | --- |
| `policy_dml_policy_outcomes_summary.csv` | Policy values, contrasts, and cohort counts. |
| `policy_dml_episode_scores.csv` | One score row per policy and episode. |
| `policy_dml_decision_scores.csv` | Rewards, Q values, recursive scores, probabilities, ratios, and model IDs. |
| `policy_dml_model_audit.json` | Compact fold and model ownership metadata. |
| `policy_dml_run_metadata.json` | Settings, policy interpretation, seeds, and saved-behaviour source. |
| `policy_dml_support_diagnostics.csv` | Support, weights, clipping, agreement, and effective sample sizes. |
| `policy_dml_clipping_sensitivity.csv` | Optional results for configured clipping alternatives. |
| `confidence_intervals/policy_dml_bootstrap_replicate_estimates.csv` | Policy values, paired differences, and draw counts. |
| `confidence_intervals/policy_dml_policy_value_confidence_intervals.csv` | Percentile intervals for policy values. |
| `confidence_intervals/policy_dml_policy_difference_confidence_intervals.csv` | Percentile intervals for paired contrasts. |

## Running and testing

```powershell
python evaluate_policy_dml_policies.py
python -m unittest test_policy_dml -v
```

`test_policy_dml.py` contains hand-calculated recursion checks, trajectory and
risk-set checks, patient leakage tests, fixed and refit bootstrap checks, and an
end-to-end output test.
