import unittest

import pandas as pd

import fit_nuisance_models as fnm


class NuisanceCautiRiskSetTests(unittest.TestCase):
    def test_binary_values_reject_missing_and_invalid_inputs(self):
        for value in (None, "bad", 2, -1, 0.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "only 0/1"):
                fnm.binary_values(pd.Series([value], name="cauti_in_period"))
        self.assertEqual(fnm.binary_values(pd.Series(["0", "1"])).tolist(), [0, 1])

    def test_observed_cauti_training_masks_use_at_risk_flag(self):
        panel = pd.DataFrame(
            {
                fnm.STATE_COL: ["in", "in", "out", "out"],
                fnm.AT_RISK_CAUTI: [1, 0, 1, 0],
            }
        )

        self.assertEqual(
            fnm.outcome_risk_mask(panel, "in", "cauti").tolist(),
            [True, False, False, False],
        )
        self.assertEqual(
            fnm.outcome_risk_mask(panel, "out", "cauti").tolist(),
            [False, False, True, False],
        )


if __name__ == "__main__":
    unittest.main()
