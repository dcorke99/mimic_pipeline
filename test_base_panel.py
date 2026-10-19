"""Insertion-anchored decision grid, linked outcomes and pre-decision features."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import create_observed_panel as cdp


T0 = pd.Timestamp("2024-01-01 06:00:00")


def episode(stay_id, removal=38, terminal=110, reason="icu_exit_alive", cauti=None):
    stamp = lambda hours: T0 + pd.Timedelta(hours=hours)
    return {
        "subject_id": stay_id, "hadm_id": stay_id, "stay_id": stay_id,
        "inserted": T0, "removed": stamp(removal),
        "reinsertion_time": stamp(terminal) if reason == "reinsertion" else pd.NaT,
        "death_time": stamp(terminal) if reason == "death" else pd.NaT,
        "ICU_in": stamp(-6),
        "ICU_out": stamp(terminal if reason == "icu_exit_alive" else terminal + 24),
        "cauti_time": stamp(cauti) if cauti is not None else pd.NaT,
        "gender": "F", "age": 60, "ethnicity": "WHITE",
        "ethnicity_group": "White", "ethnicity_missing": 0,
    }


def example_episodes():
    cases = [
        (1, 38, 110, "icu_exit_alive", None),
        (2, 48, 99, "icu_exit_alive", None),
        (3, 96, 60, "death", 12),
        (4, 60, 60, "death", None),
        (5, 38, 42, "reinsertion", None),
        (6, 38, 110, "icu_exit_alive", 86),
    ]
    return cdp.add_episode_endpoints(pd.DataFrame([episode(*case) for case in cases]))


class BaselinePanelTests(unittest.TestCase):

    def setUp(self):
        self.episodes = example_episodes()

    def test_clinical_removal_is_unique_and_terminal_ties_are_not_actions(self):
        for stay_id, rows in self.build().groupby("stay_id"):
            with self.subTest(stay_id=stay_id):
                self.assertEqual(rows.removed_in_period.sum(), int(stay_id not in (3, 4)))
                for event in rows.loc[rows.removed_in_period.eq(1)].itertuples():
                    self.assertEqual(event.catheter_state, "in")
                    self.assertLess(event.period_start, event.removed)
                    self.assertLessEqual(event.removed, event.period_end)
                if stay_id in (3, 4):
                    self.assertTrue(rows.catheter_state.eq("in").all())

    def test_terminal_events_clip_periods_and_end_all_followup(self):
        terminal_cols = ["death_in_period", "reinsertion_in_period", "icu_exit_alive_in_period"]
        for stay_id, rows in self.build().groupby("stay_id"):
            with self.subTest(stay_id=stay_id):
                self.assertTrue(rows.period_start.lt(rows.period_end).all())
                self.assertTrue(rows.period_end.le(rows.episode_end_time).all())
                self.assertTrue((rows.period_end - rows.period_start).le(cdp.PERIOD_DURATION).all())
                self.assertEqual(rows.iloc[-1].period_end, rows.iloc[-1].episode_end_time)
                self.assertEqual(rows.iloc[:-1][terminal_cols].to_numpy().sum(), 0)
                self.assertEqual(rows.iloc[-1][terminal_cols].sum(), 1)
                self.assertTrue(rows.iloc[:-1].episode_end_reason.isna().all())

    def test_cauti_exact_48h_cutoff_and_first_event_risk(self):
        rows = self.build().query("stay_id == 6")
        event = rows.loc[rows.cauti_in_period.eq(1)]
        with self.subTest():
            self.assertEqual(len(event), 1)
            self.assertEqual(event.iloc[0].at_risk_cauti, 1)
            self.assertTrue(rows.loc[rows.episode_index.gt(event.iloc[0].episode_index),
                                     "at_risk_cauti"].eq(0).all())
            # A later row must never remain eligible merely because a
            # shifted/truncated grid was mistaken for elapsed time.
            cutoff = T0 + pd.Timedelta(hours=86)
            self.assertTrue(rows.loc[rows.period_start.ge(cutoff), "at_risk_cauti"].eq(0).all())
        early = self.episodes.copy()
        early.loc[early.stay_id.eq(1), "cauti_time"] = T0 + pd.Timedelta(hours=37)
        rows = self.build(early).query("stay_id == 1")
        event_index = rows.loc[rows.cauti_in_period.eq(1), "episode_index"].iloc[0]
        self.assertTrue(rows.loc[rows.episode_index.gt(event_index), "at_risk_cauti"].eq(0).all())

    def test_event_on_boundary_is_assigned_once_to_preceding_period(self):
        for hours in (14, 24, 38, 48, 62, 72, 86):
            episodes = cdp.add_episode_endpoints(pd.DataFrame([episode(1, cauti=hours)]))
            rows = self.build(episodes)
            with self.subTest(hours=hours):
                event = rows.loc[rows.cauti_in_period.eq(1)]
                self.assertEqual(len(event), 1)
                self.assertLess(event.iloc[0].period_start, T0 + pd.Timedelta(hours=hours))
                self.assertGreaterEqual(event.iloc[0].period_end, T0 + pd.Timedelta(hours=hours))

    def test_predecision_covariates_exclude_at_and_after_decision_time(self):
        row = self.build().query("stay_id == 1 and removed_in_period == 1").copy()
        decision = row.iloc[0].period_start
        window_start = row.iloc[0].cov_start
        with self.subTest(), TemporaryDirectory() as temporary:
            def aggregate(extra):
                chart = pd.DataFrame([
                    (1, 123, window_start, 2.0),
                    (1, 123, decision - pd.Timedelta(hours=1), 4.0),
                    *extra,
                ], columns=["stay_id", "itemid", "charttime", "valuenum"])
                chart.to_csv(Path(temporary) / "cleaned_chart_covariates.csv", index=False)
                with patch.object(cdp, "DATA_DIR", Path(temporary)), redirect_stdout(StringIO()):
                    return cdp.aggregate_itemid_covariates(row, {123: "synthetic"})[0]
            baseline = aggregate([])
            poisoned = aggregate([
                (1, 123, window_start - pd.Timedelta(seconds=1), 1000.0),
                (1, 123, decision, 2000.0),
                (1, 123, decision + pd.Timedelta(seconds=1), 3000.0),
                (1, 123, T0 + pd.Timedelta(hours=39), 4000.0),
                (1, 123, T0 + pd.Timedelta(hours=111), 5000.0),
            ])
            pd.testing.assert_frame_equal(baseline, poisoned)
            self.assertEqual(poisoned.iloc[0]["itemid_123__count"], 2)
            self.assertEqual(poisoned.iloc[0]["itemid_123__mean"], 3)
            self.assertEqual(poisoned.iloc[0]["itemid_123__last"], 4)

    def build(self, episodes=None):
        return cdp.build_base_panel(self.episodes if episodes is None else episodes)

    def test_cohort_demographics_and_episode_event_counts_match_original_snapshot(self):
        fixture = Path(__file__).parent / "tests/fixtures/current_base_panel.json"
        baseline = json.loads(fixture.read_text(encoding="utf-8"))
        previous = pd.DataFrame(baseline["data"])
        current = self.build()
        events = ["removed_in_period", "cauti_in_period", "reinsertion_in_period",
                  "death_in_period", "icu_exit_alive_in_period"]
        pd.testing.assert_frame_equal(
            previous.groupby("stay_id")[events].sum(), current.groupby("stay_id")[events].sum(),
        )
        demographics = ["subject_id", "hadm_id", "age", "sex_M", "sex_missing", "ethnicity_White"]
        pd.testing.assert_frame_equal(
            previous.groupby("stay_id")[demographics].first(), current.groupby("stay_id")[demographics].first(),
        )

    def test_off_grid_removal_does_not_split_or_restart_the_insertion_clock(self):
        rows = self.build().query("stay_id == 1")
        actual = [((row.period_start - T0) / pd.Timedelta(hours=1),
                   (row.period_end - T0) / pd.Timedelta(hours=1),
                   row.catheter_state, row.removed_in_period)
                  for row in rows.itertuples()]
        self.assertEqual(actual, [(0, 24, "in", 0), (24, 48, "in", 1),
                                  (48, 72, "out", 0), (72, 96, "out", 0), (96, 110, "out", 0)])
        self.assertEqual(rows.episode_index.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(rows.episode_day.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(rows.is_decision_row.tolist(), [True, True, False, False, False])
        self.assertEqual(rows.periods_in_state.tolist(), [1, 2, 1, 2, 3])
        self.assertEqual(rows.catheter_in_hours.tolist(), [24, 14, 0, 0, 0])
        self.assertEqual(rows.catheter_out_hours.tolist(), [0, 10, 24, 24, 14])

    def test_same_period_removal_and_reinsertion_remain_linked_to_final_decision(self):
        rows = self.build().query("stay_id == 5")
        removal = rows.loc[rows.removed_in_period.eq(1)].iloc[0]
        reinsertion = rows.loc[rows.reinsertion_in_period.eq(1)].iloc[0]
        self.assertEqual(removal.catheter_state, "in")
        self.assertEqual(removal.period_end, T0 + pd.Timedelta(hours=42))
        self.assertEqual(reinsertion.name, removal.name)
        self.assertEqual(reinsertion.catheter_state, "in")
        self.assertEqual(reinsertion.period_start, T0 + pd.Timedelta(hours=24))
        self.assertEqual(reinsertion.period_end, T0 + pd.Timedelta(hours=42))
        self.assertEqual(reinsertion.at_risk_reinsertion, 1)
        self.assertEqual(reinsertion.catheter_in_hours, 14)
        self.assertEqual(reinsertion.catheter_out_hours, 4)
        self.assertEqual(len(rows), 2)

    def test_boundary_removal_belongs_to_previous_period_and_stops_decisions(self):
        rows = self.build().query("stay_id == 2")
        removal = rows.loc[rows.removed_in_period.eq(1)].iloc[0]
        self.assertEqual(removal.episode_day, 2)
        self.assertEqual(removal.period_end, T0 + pd.Timedelta(hours=48))
        self.assertEqual(removal.catheter_out_hours, 0)
        self.assertTrue(rows.loc[rows.episode_index.gt(removal.episode_index), "is_decision_row"].eq(False).all())

    def test_same_period_post_removal_cauti_is_retained_once(self):
        episodes = cdp.add_episode_endpoints(pd.DataFrame([episode(1, removal=38, terminal=42,
                                                                   reason="reinsertion", cauti=40)]))
        rows = self.build(episodes)
        final = rows.iloc[-1]
        self.assertEqual(final.removed_in_period, 1)
        self.assertEqual(final.reinsertion_in_period, 1)
        self.assertEqual(final.cauti_in_period, 1)
        self.assertEqual(final.at_risk_cauti, 1)
        self.assertEqual(final.at_risk_reinsertion, 1)
        self.assertEqual(rows.cauti_in_period.sum(), 1)

    def test_out_followup_features_use_grid_start_not_actual_removal(self):
        rows = self.build().query("stay_id == 1 and catheter_state == 'out'")
        first = rows.iloc[0]
        self.assertEqual(first.period_start, T0 + pd.Timedelta(hours=48))
        self.assertEqual(first.cov_start, T0 + pd.Timedelta(hours=24))
        self.assertEqual(first.cov_end, first.period_start)

    def test_out_followup_aggregation_rebuilds_the_new_lookback_without_future_values(self):
        row = self.build().query("stay_id == 1 and episode_index == 2").copy()
        with TemporaryDirectory() as temporary, patch.object(cdp, "DATA_DIR", Path(temporary)), \
             redirect_stdout(StringIO()):
            chart = pd.DataFrame([
                (1, 123, T0 + pd.Timedelta(hours=23), 1000),
                (1, 123, T0 + pd.Timedelta(hours=24), 2),
                (1, 123, T0 + pd.Timedelta(hours=37), 3),
                (1, 123, T0 + pd.Timedelta(hours=47), 4),
                (1, 123, T0 + pd.Timedelta(hours=48), 2000),
                (1, 123, T0 + pd.Timedelta(hours=49), 3000),
            ], columns=["stay_id", "itemid", "charttime", "valuenum"])
            chart.to_csv(Path(temporary) / "cleaned_chart_covariates.csv", index=False)
            aggregated = cdp.aggregate_itemid_covariates(row, {123: "synthetic"})[0].iloc[0]
            self.assertEqual(aggregated["itemid_123__count"], 3)
            self.assertEqual(aggregated["itemid_123__mean"], 3)
            self.assertEqual(aggregated["itemid_123__last"], 4)

    def test_pre_icu_insertion_retains_an_empty_feature_window(self):
        episodes = self.episodes.copy()
        episodes.loc[episodes.stay_id.eq(1), "ICU_in"] = T0 + pd.Timedelta(hours=1)
        base = self.build(episodes)
        initial = base.query("stay_id == 1 and episode_index == 0").iloc[0]
        self.assertGreater(initial.cov_start, initial.cov_end)

    def test_builder_stages_preserve_schema_actions_and_qa(self):
        with TemporaryDirectory() as temporary, patch.object(cdp, "DATA_DIR", Path(temporary)), \
             patch.object(cdp, "build_catheter_episodes", return_value=self.episodes), redirect_stdout(StringIO()):
            cdp.create_episode_cohort_and_base_panel()
            base = self.build()
            chart = pd.DataFrame({
                "stay_id": base["stay_id"], "itemid": 123,
                "charttime": base["period_start"] - pd.Timedelta(hours=1),
                "valuenum": np.arange(len(base), dtype=float),
            })
            chart.to_csv(Path(temporary) / "cleaned_chart_covariates.csv", index=False)
            with patch.object(cdp, "load_item_labels", return_value={123: "synthetic"}):
                cdp.build_modelling_panel()
            modelling = pd.read_csv(Path(temporary) / "modelling_panel.csv")
            self.assertEqual(len(modelling), len(base))
            self.assertTrue((set(base.columns) - {"cov_start", "cov_end", "row_id"}).issubset(modelling.columns))
            self.assertEqual(modelling.loc[modelling.removed_in_period.eq(1), "observed_action"].unique().tolist(), ["remove"])


if __name__ == "__main__":
    unittest.main()
