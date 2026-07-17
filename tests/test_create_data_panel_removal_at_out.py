import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from create_data_panel_removal_at_out import (
    add_transition_columns,
    base_feature_cols,
    build_base_panel,
    cleanup_chart_intermediate,
)


def make_episode(icu_out="2162-06-22 20:52:00"):
    return pd.DataFrame([{
        "subject_id": 13180007,
        "hadm_id": 27543152,
        "stay_id": 30000213,
        "inserted": pd.Timestamp("2162-06-21 06:45:00"),
        "removed": pd.Timestamp("2162-06-22 11:02:00"),
        "reinsertion_time": pd.NaT,
        "cauti_time": pd.NaT,
        "death_time": pd.NaT,
        "ICU_out": pd.Timestamp(icu_out),
        "ICU_in": pd.Timestamp("2162-06-20 00:00:00"),
        "gender": "M",
        "age": 55,
        "ethnicity_group": "WHITE",
    }])


class RemovalAtOutTests(unittest.TestCase):
    def test_exact_episode_places_removal_on_first_out_row(self):
        panel = build_base_panel(make_episode())
        add_transition_columns(panel)

        expected = pd.DataFrame({
            "period_start": pd.to_datetime([
                "2162-06-21 06:45:00",
                "2162-06-22 06:45:00",
                "2162-06-22 11:02:00",
            ]),
            "period_end": pd.to_datetime([
                "2162-06-22 06:45:00",
                "2162-06-22 11:02:00",
                "2162-06-22 20:52:00",
            ]),
            "catheter_state": ["in", "in", "out"],
            "removed_in_period": [0, 0, 1],
            "observed_action": pd.Series(["keep", "keep", "remove"], dtype="string"),
            "action_remove": pd.Series([0, 0, 1], dtype="Int64"),
            "is_decision_row": [1, 1, 1],
            "icu_end_in_period": [0, 0, 1],
        })

        pd.testing.assert_frame_equal(
            panel[expected.columns].reset_index(drop=True),
            expected,
            check_dtype=False,
        )
        self.assertEqual(panel["next_state"].tolist(), [
            "no_event_continue",
            "no_event_continue",
            "icu_exit_alive",
        ])

    def test_later_out_rows_are_nondecision_rows_with_na_action_remove(self):
        panel = build_base_panel(make_episode(icu_out="2162-06-24 20:52:00"))
        add_transition_columns(panel)

        out_rows = panel.loc[panel["catheter_state"] == "out"].reset_index(drop=True)
        self.assertEqual(out_rows.loc[0, "period_start"], pd.Timestamp("2162-06-22 11:02:00"))
        self.assertEqual(out_rows.loc[0, "removed_in_period"], 1)
        self.assertEqual(out_rows.loc[0, "observed_action"], "remove")
        self.assertEqual(out_rows.loc[0, "action_remove"], 1)
        self.assertEqual(out_rows.loc[0, "is_decision_row"], 1)

        later_out_rows = out_rows.iloc[1:]
        self.assertTrue((later_out_rows["removed_in_period"] == 0).all())
        self.assertTrue((later_out_rows["observed_action"] == "out").all())
        self.assertTrue(later_out_rows["action_remove"].isna().all())
        self.assertTrue((later_out_rows["is_decision_row"] == 0).all())

    def test_no_out_row_is_fabricated_when_removal_equals_followup_end(self):
        episode = make_episode(icu_out="2162-06-22 11:02:00")
        panel = build_base_panel(episode)
        add_transition_columns(panel)

        self.assertEqual(len(panel), 2)
        self.assertTrue((panel["catheter_state"] == "in").all())
        self.assertEqual(panel["removed_in_period"].sum(), 0)
        self.assertTrue((panel["observed_action"] == "keep").all())
        self.assertTrue((panel["action_remove"] == 0).all())
        self.assertTrue((panel["is_decision_row"] == 1).all())

    def test_decision_indicator_is_not_a_model_feature(self):
        columns = pd.DataFrame(columns=["is_decision_row", "age", "itemid_1__mean"])
        self.assertEqual(base_feature_cols(columns), ["itemid_1__mean", "age"])

    def test_chart_intermediate_cleanup_defaults_to_removing_regenerable_file(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw_chart_covariates.csv"
            path.write_bytes(b"regenerable")
            removed_bytes = cleanup_chart_intermediate(path)

            self.assertEqual(removed_bytes, len(b"regenerable"))
            self.assertFalse(path.exists())

    def test_chart_intermediate_cleanup_can_be_disabled(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw_chart_covariates.csv"
            path.write_bytes(b"keep")
            removed_bytes = cleanup_chart_intermediate(path, keep_intermediates=True)

            self.assertEqual(removed_bytes, 0)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
