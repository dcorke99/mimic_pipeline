"""Regression tests for patient-clustered policy-value inference.

The fixture has unequal numbers of episodes per patient, nonuniform support
weights, and a patient with no fixed-policy support.  Physical row expansion
provides an independent check of the sufficient-statistic bootstrap.
"""

from io import StringIO
import unittest

import numpy as np
import pandas as pd

import policy_bootstrap as ci
import evaluate_aipw_policies as aipw
import evaluate_gformula_policies as gformula
import evaluate_ipw_policies as ipw


ESTIMATORS = {"gformula": gformula, "ipw": ipw, "aipw": aipw}


OUTCOMES = (
    "cauti", "recatheterisation", "death", "icu_exit_alive",
    "catheter_exposure_days",
)


def example_episodes():
    subjects = pd.Index([10, 20, 30], name="subject_id")
    policies = pd.DataFrame({
        "policy_name": ["current_practice", "remove_on_day_1", "remove_on_day_3"],
        "policy_remove_day": [np.nan, 1.0, 3.0],
    })
    plugins = {
        "cauti": [0.1, 0.2, 0.3, 0.4],
        "recatheterisation": [0.1, 0.1, 0.1, 1.0],
        "death": [0.9, 0.9, 0.9, 0.0],
        "icu_exit_alive": [0.5, 0.4, 0.3, 0.2],
        "catheter_exposure_days": [1.0, 2.0, 3.0, 4.0],
    }
    observed = {
        "cauti": [1.0, 0.0, 1.0, 1.0],
        "recatheterisation": [0.0, 0.0, 0.0, 0.0],
        "death": [1.0, 1.0, 1.0, 1.0],
        "icu_exit_alive": [0.0, 1.0, 1.0, 1.0],
        "catheter_exposure_days": [4.0, 1.0, 7.0, 8.0],
    }
    gf_columns = {
        "cauti": "predicted_any_cauti",
        "recatheterisation": "predicted_any_recatheterisation",
        "death": "predicted_any_death",
        "icu_exit_alive": "predicted_icu_exit_alive",
        "catheter_exposure_days": "expected_catheter_exposure_days",
    }
    ipw_columns = {
        "cauti": "any_cauti",
        "recatheterisation": "any_recatheterisation",
        "death": "any_death",
        "icu_exit_alive": "observed_icu_exit_alive",
        "catheter_exposure_days": "observed_catheter_exposure_days",
    }
    frames = {name: [] for name in ("gformula", "ipw", "aipw")}
    for policy in policies.itertuples(index=False):
        current = policy.policy_name == "current_practice"
        base = pd.DataFrame({
            "subject_id": [10, 10, 20, 30],
            "catheter_episode_id": [101, 102, 201, 301],
            "policy_name": policy.policy_name,
            "policy_remove_day": policy.policy_remove_day,
            "policy_type": "observed" if current else "fixed_day_removal",
            "prediction_complete": True,
            "episode_adherent_to_policy": [1, 1, 1 if current else 0, 1],
        })
        weights = np.ones(4) if current else np.array([1.0, 3.0, 0.0, 2.0])
        gformula, ipw, aipw = (base.copy() for _ in range(3))
        # These interval-count columns are required by the existing summary
        # helpers even though inference uses duration in days.
        gformula["expected_catheter_in_intervals"] = [1, 2, 3, 4]
        gformula["expected_catheter_in_interval_rows"] = [1, 2, 3, 4]
        ipw["observed_catheter_in_intervals"] = [4, 1, 7, 8]
        ipw["episode_ipw_weight"] = weights
        aipw["residual_correction_weight"] = weights
        aipw["plugin_expected_catheter_in_interval_rows"] = [1, 2, 3, 4]
        for outcome in OUTCOMES:
            plugin = np.asarray(plugins[outcome])
            obs = np.asarray(observed[outcome])
            gformula[gf_columns[outcome]] = plugin
            ipw[ipw_columns[outcome]] = obs
            aipw["plugin_" + gf_columns[outcome]] = plugin
            aipw["observed_" + (
                "catheter_exposure_days" if outcome == "catheter_exposure_days"
                else "icu_exit_alive" if outcome == "icu_exit_alive"
                else "any_" + outcome
            )] = obs
            aipw["residual_" + outcome] = obs - plugin
            aipw["aipw_ht_score_" + outcome] = plugin + weights * (obs - plugin)
        frames["gformula"].append(gformula)
        # Match persisted IPW outputs: only adherent fixed-policy episodes,
        # plus every episode in the separately loaded observed baseline.
        frames["ipw"].append(ipw.loc[ipw["episode_adherent_to_policy"].eq(1)])
        frames["aipw"].append(aipw)
    episodes = {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}
    return episodes, policies, subjects


def expand_patients(episodes, subjects, counts):
    multiplicity = pd.Series(counts, index=subjects)
    return {
        estimator: frame.loc[
            frame.index.repeat(frame["subject_id"].map(multiplicity).to_numpy())
        ].reset_index(drop=True)
        for estimator, frame in episodes.items()
    }


def example_summaries(episodes, policies, rounded=True):
    """Build realistic saved summaries, including comparisons before rounding."""
    summaries = {
        "gformula": gformula.build_policy_summary(
            episodes["gformula"], gformula.PREDICTION_MODE,
        ),
        "aipw": aipw.build_policy_summary(episodes["aipw"], "hajek"),
    }
    rows = []
    for policy in policies.itertuples(index=False):
        group = episodes["ipw"].loc[episodes["ipw"].policy_name.eq(policy.policy_name)]
        rows.append(ipw.summarise_episode_estimates(
            group, policy.policy_name, policy.policy_remove_day, 4, len(group),
        ))
    summaries["ipw"] = ipw.add_current_practice_comparisons(pd.DataFrame(rows))
    return {name: frame.round(3) if rounded else frame for name, frame in summaries.items()}


def reference_values(episodes, policies):
    summaries = example_summaries(episodes, policies, rounded=False)
    return np.array([
        summaries[name].set_index("policy_name").loc[policy, SUMMARY_COLUMNS[name][outcome]]
        for name in ESTIMATORS for policy in policies.policy_name for outcome in OUTCOMES
    ], dtype=float)


SUMMARY_COLUMNS = {
    "gformula": dict(zip(OUTCOMES, [
        "predicted_cauti_risk", "predicted_recatheterisation_risk",
        "predicted_death_risk", "predicted_icu_exit_alive_risk",
        "expected_mean_catheter_exposure_days",
    ])),
    "ipw": {
        name: f"ipw_weighted_{name}_risk" for name in ipw.OUTCOME_SPECS
    } | {"catheter_exposure_days": "ipw_weighted_mean_catheter_exposure_days"},
    "aipw": {
        name: f"aipw_{spec['summary_stub']}"
        for name, spec in aipw.OUTCOME_SPECS.items()
    } | {"catheter_exposure_days": "aipw_mean_catheter_exposure_days"},
}


def bootstrap_values(design, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    n_patients = len(design.statistics)
    values = np.empty((n_bootstrap, len(design.entries)))
    diagnostics = []
    for replicate in range(n_bootstrap):
        # One shared draw for all estimators, policies and outcomes is essential
        # for paired differences from current practice.
        counts = np.bincount(rng.integers(0, n_patients, size=n_patients), minlength=n_patients)
        values[replicate] = design.estimate(counts)
        diagnostics.append({
            "bootstrap_replicate": replicate + 1,
            "n_patient_draws": int(counts.sum()),
            "n_unique_patients_resampled": int(np.count_nonzero(counts)),
            "n_episodes_resampled": int(counts @ design.episode_counts),
        })
    return values, pd.DataFrame(diagnostics)


class PatientBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.episodes, self.policies, self.subjects = example_episodes()
        self.design = ci.build_patient_statistics(
            self.episodes, self.policies, self.subjects,
        )

    def entry_index(self, estimator, policy, outcome):
        entries = self.design.entries
        matches = np.flatnonzero(
            entries["estimator"].eq(estimator)
            & entries["policy_name"].eq(policy)
            & entries["outcome"].eq(outcome)
        )
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_single_estimator_statistics_match_joint_statistics(self):
        for name in ESTIMATORS:
            with self.subTest(estimator=name):
                design = ci.build_patient_statistics({name: self.episodes[name]}, self.policies, self.subjects)
                mask = self.design.entries.estimator.eq(name)
                for counts in (np.ones(3, dtype=int), np.array([2, 0, 1])):
                    np.testing.assert_allclose(
                        design.estimate(counts), self.design.estimate(counts)[mask],
                        rtol=1e-12, atol=1e-12, equal_nan=True,
                    )

    def test_all_estimators_policies_outcomes_and_original_estimates(self):
        self.assertEqual(len(self.design.entries), 3 * 3 * 5)
        for estimator in self.episodes:
            for policy in self.policies["policy_name"]:
                for outcome in OUTCOMES:
                    self.entry_index(estimator, policy, outcome)
        np.testing.assert_allclose(
            self.design.estimate(np.ones(3, dtype=int)),
            reference_values(self.episodes, self.policies),
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )

    def test_whole_patients_match_physical_episode_expansion(self):
        counts = np.array([2, 0, 1])
        expanded = expand_patients(self.episodes, self.subjects, counts)
        baseline = expanded["gformula"].query("policy_name == 'current_practice'")
        self.assertEqual(baseline["catheter_episode_id"].value_counts().to_dict(),
                         {101: 2, 102: 2, 301: 1})
        np.testing.assert_array_equal(self.design.episode_counts, [2, 1, 1])
        self.assertEqual(int(counts @ self.design.episode_counts), len(baseline))
        np.testing.assert_allclose(
            self.design.estimate(counts), reference_values(expanded, self.policies),
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )

    def test_hajek_residual_correction_is_not_average_ht_score(self):
        values = self.design.estimate(np.ones(3, dtype=int))
        index = self.entry_index("aipw", "remove_on_day_1", "cauti")
        # mean(plugin)=.25; weighted residual sum=1.5; weight sum=6.
        self.assertAlmostEqual(values[index], 0.5)
        self.assertNotAlmostEqual(values[index], 0.625)  # HT score mean
        self.assertAlmostEqual(values[self.entry_index(
            "aipw", "remove_on_day_1", "death")], 1.0)
        self.assertAlmostEqual(values[self.entry_index(
            "aipw", "remove_on_day_1", "recatheterisation")], 0.0)
        self.assertAlmostEqual(values[self.entry_index(
            "aipw", "remove_on_day_1", "catheter_exposure_days")], 23.0 / 6.0)

    def test_no_supported_patient_gives_undefined_weighted_estimates(self):
        values = self.design.estimate(np.array([0, 3, 0]))
        for estimator in ("ipw", "aipw"):
            for outcome in OUTCOMES:
                self.assertTrue(np.isnan(values[self.entry_index(
                    estimator, "remove_on_day_1", outcome)]))
        for estimator in self.episodes:
            for outcome in OUTCOMES:
                self.assertTrue(np.isfinite(values[self.entry_index(
                    estimator, "current_practice", outcome)]))

    def test_draws_are_shared_and_reproducible(self):
        seed, n_bootstrap = 1234, 50
        values, diagnostics = bootstrap_values(self.design, n_bootstrap, seed)
        repeated, repeated_diagnostics = bootstrap_values(self.design, n_bootstrap, seed)
        np.testing.assert_array_equal(values, repeated)
        pd.testing.assert_frame_equal(diagnostics, repeated_diagnostics)
        self.assertEqual(values.shape, (n_bootstrap, len(self.design.entries)))
        rng, raw_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        for replicate in range(n_bootstrap):
            counts = np.bincount(rng.integers(0, 3, size=3), minlength=3)
            expected = np.bincount(raw_rng.integers(0, 3, size=3), minlength=3)
            np.testing.assert_array_equal(counts, expected)
            self.assertEqual(int(counts.sum()), 3)
            # One draw is used jointly for all estimators and policies.
            np.testing.assert_allclose(values[replicate], self.design.estimate(counts),
                                       rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_policy_differences_use_paired_replicate_baselines(self):
        values, _ = bootstrap_values(self.design, 50, 91)
        differences = ci.paired_differences(values, self.design.entries)
        for index, entry in self.design.entries.iterrows():
            baseline = self.entry_index(entry["estimator"], "current_practice", entry["outcome"])
            np.testing.assert_allclose(
                differences[:, index], values[:, index] - values[:, baseline],
                rtol=1e-12, atol=1e-12, equal_nan=True,
            )

    def test_replicate_csv_has_one_row_per_draw_and_preserves_appended_estimates(self):
        values, diagnostics = bootstrap_values(self.design, 5, 91)
        values[2, self.entry_index("ipw", "remove_on_day_1", "cauti")] = np.nan
        diagnostics["n_fallback_folds"] = [0, 1, 0, 0, 0]
        table = ci.bootstrap_replicate_table(self.design.entries, values, diagnostics)
        self.assertEqual(len(table), 5)
        self.assertTrue(table.columns.is_unique)
        self.assertEqual(len(table.columns) - len(diagnostics.columns), 2 * len(self.design.entries))
        pd.testing.assert_frame_equal(table[diagnostics.columns], diagnostics)
        value_columns = [name for name in table if name.endswith("__policy_value")]
        difference_columns = [name for name in table if name.endswith("__difference_vs_current_practice")]
        np.testing.assert_array_equal(table[value_columns].to_numpy(), values)
        np.testing.assert_array_equal(
            table[difference_columns].to_numpy(), ci.paired_differences(values, self.design.entries),
        )

        # The refitting script writes one completed pass at a time. Reading
        # those appended rows must give exactly the same table as batch export.
        output = StringIO()
        for replicate in range(len(values)):
            row = ci.bootstrap_replicate_table(
                self.design.entries, values[replicate:replicate + 1], diagnostics.iloc[[replicate]],
            )
            row.to_csv(output, index=False, header=replicate == 0)
        restored = pd.read_csv(StringIO(output.getvalue()), float_precision="round_trip")
        pd.testing.assert_frame_equal(restored, table)

    def test_missing_values_and_incomplete_predictions_keep_original_masks(self):
        episodes = {name: frame.copy() for name, frame in self.episodes.items()}
        for estimator in ("gformula", "aipw"):
            frame = episodes[estimator]
            target = frame.policy_name.eq("remove_on_day_1")
            frame.loc[target & frame.catheter_episode_id.eq(101), "prediction_complete"] = False
            plugin = ("predicted_any_cauti" if estimator == "gformula"
                      else "plugin_predicted_any_cauti")
            frame.loc[target & frame.catheter_episode_id.eq(102), plugin] = np.nan
        frame = episodes["aipw"]
        target = frame.policy_name.eq("remove_on_day_1")
        frame.loc[target & frame.catheter_episode_id.eq(301), "residual_death"] = np.nan
        # A missing residual at zero weight also follows the original 0*NaN mask.
        frame.loc[target & frame.catheter_episode_id.eq(201), "residual_cauti"] = np.nan
        design = ci.build_patient_statistics(episodes, self.policies, self.subjects)
        for counts in (np.ones(3, dtype=int), np.array([2, 0, 1])):
            with self.subTest(counts=counts.tolist()):
                expanded = expand_patients(episodes, self.subjects, counts)
                np.testing.assert_allclose(
                    design.estimate(counts), reference_values(expanded, self.policies),
                    rtol=1e-12, atol=1e-12, equal_nan=True,
                )
        values = design.estimate(np.ones(3, dtype=int))
        self.assertAlmostEqual(values[self.entry_index(
            "gformula", "remove_on_day_1", "cauti")], 0.35)
        # Plugin excludes missing values independently of the residual mask:
        # mean(.3,.4) + (3*(-.2) + 2*.6)/(3+2) = .47.
        self.assertAlmostEqual(values[self.entry_index(
            "aipw", "remove_on_day_1", "cauti")], 0.47)
        # The missing death residual removes its weight only for death.
        self.assertAlmostEqual(values[self.entry_index(
            "aipw", "remove_on_day_1", "death")], 0.7)

    def test_ipw_excludes_nonpositive_nonfinite_weights_and_nonfinite_outcomes(self):
        cases = [
            ("episode_ipw_weight", invalid)
            for invalid in (0.0, -1.0, np.nan, np.inf, -np.inf)
        ] + [
            ("any_cauti", invalid) for invalid in (np.nan, np.inf, -np.inf)
        ]
        for column, invalid in cases:
            with self.subTest(column=column, invalid=invalid):
                episodes = {name: frame.copy() for name, frame in self.episodes.items()}
                frame = episodes["ipw"]
                mask = frame.policy_name.eq("remove_on_day_1") & frame.catheter_episode_id.eq(102)
                frame.loc[mask, column] = invalid
                design = ci.build_patient_statistics(episodes, self.policies, self.subjects)
                values = design.estimate(np.ones(3, dtype=int))
                np.testing.assert_allclose(
                    values, reference_values(episodes, self.policies),
                    rtol=1e-12, atol=1e-12, equal_nan=True,
                )
                self.assertAlmostEqual(values[self.entry_index(
                    "ipw", "remove_on_day_1", "cauti")], 1.0)

    def test_percentiles_exclude_and_count_undefined_replicates(self):
        entries = self.design.entries.iloc[:3].reset_index(drop=True)
        bootstrap = np.array([
            [0.0, 0.1, np.nan], [0.2, np.nan, np.nan],
            [0.4, 0.5, np.nan], [0.6, np.inf, np.nan],
            [0.8, 0.9, np.nan], [1.0, -np.inf, np.nan],
        ])
        table = ci.interval_table(entries, np.array([0.5, 0.5, np.nan]), bootstrap, 3)
        np.testing.assert_allclose(table.loc[0, ["ci_lower", "ci_upper"]].to_numpy(dtype=float),
                                   [0.025, 0.975])
        np.testing.assert_allclose(table.loc[1, ["ci_lower", "ci_upper"]].to_numpy(dtype=float),
                                   [0.12, 0.88])
        self.assertTrue(table.loc[2, ["ci_lower", "ci_upper"]].isna().all())
        np.testing.assert_array_equal(table.n_bootstrap, [6, 6, 6])
        np.testing.assert_array_equal(table.n_valid_bootstrap, [6, 3, 0])
        np.testing.assert_array_equal(table.n_undefined_bootstrap, [0, 3, 6])
        np.testing.assert_array_equal(table.n_unique_patients, [3, 3, 3])

    def test_difference_interval_uses_percentiles_of_paired_differences(self):
        indices = [self.entry_index("aipw", policy, "cauti")
                   for policy in ("current_practice", "remove_on_day_1")]
        entries = self.design.entries.iloc[indices].reset_index(drop=True)
        values = np.array([[0.1, 0.6], [0.2, 0.5], [0.3, 0.8], [0.4, 0.7]])
        point = np.array([0.25, 0.65])
        table = ci.interval_table(entries, ci.paired_differences(point, entries),
                                  ci.paired_differences(values, entries), 3)
        bounds = table[["ci_lower", "ci_upper"]].to_numpy()
        np.testing.assert_allclose(bounds, [[0.0, 0.0], [0.3, 0.5]])
        self.assertAlmostEqual(table.loc[1, "point_estimate"], 0.4)
        # Subtracting marginal CI endpoints loses the shared patient draw.
        marginal = np.percentile(values, [2.5, 97.5], axis=0)
        incorrect = [marginal[0, 1] - marginal[1, 0], marginal[1, 1] - marginal[0, 0]]
        self.assertFalse(np.allclose(bounds[1], incorrect))




if __name__ == "__main__":
    unittest.main()
