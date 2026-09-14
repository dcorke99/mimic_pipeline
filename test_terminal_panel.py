import unittest

import pandas as pd

import create_data_panel as cdp


class TerminalPanelTests(unittest.TestCase):
    @staticmethod
    def episode(
        stay_id,
        *,
        inserted,
        removed,
        reinsertion_time,
        death_time,
        icu_out,
        cauti_time=pd.NaT,
    ):
        return {
            "subject_id": stay_id,
            "hadm_id": stay_id,
            "stay_id": stay_id,
            "inserted": inserted,
            "removed": removed,
            "reinsertion_time": reinsertion_time,
            "death_time": death_time,
            "ICU_in": inserted - pd.Timedelta(hours=1),
            "ICU_out": icu_out,
            "cauti_time": cauti_time,
            "gender": "F",
            "age": 60,
            "ethnicity": "WHITE",
            "ethnicity_group": "White",
            "ethnicity_missing": 0,
        }

    def setUp(self):
        t0 = pd.Timestamp("2024-01-01 00:00:00")
        self.t0 = t0
        self.episodes = pd.DataFrame(
            [
                # Death truncates the catheter-in segment before recorded removal.
                self.episode(
                    1,
                    inserted=t0,
                    removed=t0 + pd.Timedelta(hours=96),
                    reinsertion_time=t0 + pd.Timedelta(hours=120),
                    death_time=t0 + pd.Timedelta(hours=60),
                    icu_out=t0 + pd.Timedelta(hours=144),
                ),
                # Reinsertion closes an already-removed catheter episode.
                self.episode(
                    2,
                    inserted=t0,
                    removed=t0 + pd.Timedelta(hours=24),
                    reinsertion_time=t0 + pd.Timedelta(hours=72),
                    death_time=t0 + pd.Timedelta(hours=96),
                    icu_out=t0 + pd.Timedelta(hours=120),
                ),
                # ICU discharge without death is an ICU-exit-alive terminal event.
                self.episode(
                    3,
                    inserted=t0,
                    removed=t0 + pd.Timedelta(hours=24),
                    reinsertion_time=pd.NaT,
                    death_time=pd.NaT,
                    icu_out=t0 + pd.Timedelta(hours=54),
                ),
                # Death takes precedence when death and ICU exit timestamps tie.
                self.episode(
                    4,
                    inserted=t0,
                    removed=t0 + pd.Timedelta(hours=24),
                    reinsertion_time=pd.NaT,
                    death_time=t0 + pd.Timedelta(hours=48),
                    icu_out=t0 + pd.Timedelta(hours=48),
                ),
                # A procedure end tied with death is not a removal decision.
                self.episode(
                    5,
                    inserted=t0,
                    removed=t0 + pd.Timedelta(hours=48),
                    reinsertion_time=pd.NaT,
                    death_time=t0 + pd.Timedelta(hours=48),
                    icu_out=t0 + pd.Timedelta(hours=72),
                ),
            ]
        )
        self.episodes = cdp.add_episode_endpoints(self.episodes)
        self.panel = cdp.build_base_panel(self.episodes)

        cauti_episodes = self.episodes.copy()
        cauti_episodes.loc[
            cauti_episodes["stay_id"].eq(1), "cauti_time"
        ] = t0 + pd.Timedelta(hours=12)
        cauti_episodes.loc[
            cauti_episodes["stay_id"].eq(2), "cauti_time"
        ] = t0 + pd.Timedelta(hours=36)
        self.cauti_panel = cdp.build_base_panel(cauti_episodes)

    def episode_rows(self, stay_id):
        return self.panel.loc[self.panel["stay_id"].eq(stay_id)].sort_values(
            ["period_start", "period_end"]
        )

    def test_earliest_event_resolves_episode_reason(self):
        reasons = self.episodes.set_index("stay_id")["episode_end_reason"].to_dict()
        self.assertEqual(reasons[1], "death")
        self.assertEqual(reasons[2], "reinsertion")
        self.assertEqual(reasons[3], "icu_exit_alive")
        self.assertEqual(reasons[4], "death")

    def test_episode_starts_at_or_after_terminal_event_are_excluded(self):
        candidates = pd.DataFrame(
            [
                {
                    "stay_id": 10,
                    "inserted": self.t0,
                    "death_time": self.t0 + pd.Timedelta(hours=48),
                    "ICU_out": self.t0 + pd.Timedelta(hours=72),
                },
                {
                    "stay_id": 11,
                    "inserted": self.t0 + pd.Timedelta(hours=48),
                    "death_time": self.t0 + pd.Timedelta(hours=48),
                    "ICU_out": self.t0 + pd.Timedelta(hours=72),
                },
                {
                    "stay_id": 12,
                    "inserted": self.t0 + pd.Timedelta(hours=80),
                    "death_time": pd.NaT,
                    "ICU_out": self.t0 + pd.Timedelta(hours=72),
                },
            ]
        )
        retained = cdp.exclude_post_terminal_episode_starts(candidates)
        self.assertEqual(retained["stay_id"].tolist(), [10])

    def test_no_rows_extend_past_terminal_time(self):
        self.assertTrue(
            self.panel["period_end"].le(self.panel["episode_end_time"]).all()
        )
        final_end = self.panel.groupby(["stay_id", "inserted"])[
            "period_end"
        ].max()
        expected_end = self.episodes.set_index(["stay_id", "inserted"])[
            "episode_end_time"
        ]
        pd.testing.assert_series_equal(
            final_end.sort_index(),
            expected_end.sort_index(),
            check_names=False,
        )

    def test_death_before_removal_has_no_out_or_removal_rows(self):
        rows = self.episode_rows(1)
        self.assertTrue(rows["catheter_state"].eq("in").all())
        self.assertEqual(int(rows["removed_in_period"].sum()), 0)
        self.assertEqual(int(rows["death_in_period"].sum()), 1)
        self.assertEqual(rows.iloc[-1]["period_end"], self.t0 + pd.Timedelta(hours=60))

    def test_terminal_events_are_mutually_exclusive_and_on_final_rows(self):
        terminal_cols = [
            "death_in_period",
            "reinsertion_in_period",
            "icu_exit_alive_in_period",
        ]
        terminal_count = self.panel[terminal_cols].sum(axis=1)
        self.assertEqual(int(terminal_count.sum()), len(self.episodes))
        self.assertTrue(terminal_count.le(1).all())

        for stay_id in self.episodes["stay_id"]:
            rows = self.episode_rows(stay_id)
            self.assertEqual(int(rows.iloc[:-1][terminal_cols].to_numpy().sum()), 0)
            self.assertTrue(rows.iloc[:-1]["episode_end_reason"].isna().all())
            self.assertEqual(int(rows.iloc[-1][terminal_cols].sum()), 1)
            self.assertTrue(pd.notna(rows.iloc[-1]["episode_end_reason"]))

    def test_chart_extraction_windows_stop_at_terminal_time(self):
        windows = cdp.build_chart_extraction_windows(self.episodes)
        actual_end = windows.set_index("stay_id")["window_end"].sort_index()
        expected_end = self.episodes.set_index("stay_id")[
            "episode_end_time"
        ].sort_index()
        pd.testing.assert_series_equal(actual_end, expected_end, check_names=False)

    def test_death_icu_exit_tie_is_death_only(self):
        final = self.episode_rows(4).iloc[-1]
        self.assertEqual(final["episode_end_reason"], "death")
        self.assertEqual(int(final["death_in_period"]), 1)
        self.assertEqual(int(final["icu_exit_alive_in_period"]), 0)

    def test_removal_tied_with_death_is_not_an_action(self):
        rows = self.episode_rows(5)
        self.assertEqual(int(rows["removed_in_period"].sum()), 0)
        self.assertEqual(int(rows["death_in_period"].sum()), 1)

    def test_every_cauti_event_row_is_at_risk(self):
        event_rows = self.cauti_panel.loc[
            self.cauti_panel["cauti_in_period"].eq(1)
        ]
        self.assertEqual(len(event_rows), 2)
        self.assertTrue(event_rows["at_risk_cauti"].eq(1).all())

    def test_no_later_episode_row_is_at_risk_for_cauti(self):
        for _, episode_rows in self.cauti_panel.groupby(
            ["stay_id", "inserted"], sort=False
        ):
            rows = episode_rows.sort_values(["period_start", "period_end"])
            prior_cauti = rows["cauti_in_period"].cumsum().sub(
                rows["cauti_in_period"]
            ).gt(0)
            self.assertTrue(rows.loc[prior_cauti, "at_risk_cauti"].eq(0).all())

    def test_episodes_without_cauti_retain_state_window_risk_set(self):
        expected = (
            self.panel["catheter_state"].eq("in")
            | (
                self.panel["catheter_state"].eq("out")
                & self.panel["periods_in_state"].le(
                    cdp.POST_REMOVE_RISK_PERIODS
                )
            )
        ).astype(int)
        pd.testing.assert_series_equal(
            self.panel["at_risk_cauti"], expected, check_names=False
        )

    def test_cauti_does_not_change_other_outcomes_or_terminal_state(self):
        unchanged_cols = [
            col
            for col in self.panel.columns
            if col not in {"cauti_in_period", "at_risk_cauti"}
        ]
        pd.testing.assert_frame_equal(
            self.cauti_panel[unchanged_cols],
            self.panel[unchanged_cols],
        )


if __name__ == "__main__":
    unittest.main()
