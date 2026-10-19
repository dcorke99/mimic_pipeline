import unittest

import pandas as pd

import fit_nuisance_models as fnm


class NuisanceCautiRiskSetTests(unittest.TestCase):
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
