# G-formula current-practice comparator

`evaluate_gformula_policies.py` estimates episode-level expected first-CAUTI
risk as `1 - product(1 - hazard)` along the factual treatment trajectory on
the preserved episode grid. This comparator uses the observed actions, including
simulated actions in semi-synthetic panels; it does not average over new action
trajectories.

The current-practice prediction window includes periods whose start is strictly
before `removed + 48 hours`, matching `build_known_truth_policy_values()`.
Periods starting exactly at that cutoff contribute zero. The episode grid ends
at death, reinsertion, ICU exit, or the recorded simulation endpoint, so no
follow-up is added beyond an episode boundary. Realised CAUTI does not shorten
this expected-risk prediction window.

`policy_eval_common.add_current_practice_cauti_predictions()` scores the IN or
OUT CAUTI model belonging to each patient's held-out fold. Inputs retain factual
pre-decision covariates, period timing, and the observed removal action. The
separate `p_cauti_current_practice_unmasked` column holds these hazards. Eligible
rows with a missing fold model or invalid probability fail explicitly.

Factual `p_cauti_if_*` scores can contain structural zeros after infection.
Fixed-nuisance evaluation therefore rescores current-practice hazards from the
saved `outcome_models.pkl` even when the saved probability columns are complete.
Existing saved nuisance models can be reused if they were fitted on the corrected
first-event training risk sets and their saved features and fold assignments
still match the panel. Models trained before that correction require refitting.
The evaluator cannot recover underlying hazards from masked probabilities alone.

New nuisance fits export the separate unmasked column. Refit point estimates and
every refitted bootstrap draw generate it from their newly fitted held-out
models. CAUTI training still uses only factual `at_risk_cauti == 1` rows: the
first observed event is retained, and post-infection rows remain excluded.

Fixed bootstrap mode resamples patient-level sums of the corrected episode
predictions. Refit bootstrap mode keeps the existing patient multiplicities and
grouped folds, then repeats model fitting, hazard scoring, and episode aggregation.

The structural window and separate hazard selection are explicitly enabled by
the g-formula evaluator. Shared helper defaults retain their factual risk-mask
semantics. Policy-DML, fixed-day policy windows and predictions, and the
semi-synthetic generator and known-truth values are unchanged. Fixed-day nuisance
underprediction and model recalibration are separate issues.

Regression checks are in `test_gformula_current_practice.py`:

```powershell
python -m unittest test_gformula_current_practice -v
```
