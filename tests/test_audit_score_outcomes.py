import unittest

import pandas as pd

from audit_score_outcomes import scenario_counts


class ScoreOutcomeAuditTests(unittest.TestCase):
    def test_counts_low_score_with_later_reversion(self):
        frame = pd.DataFrame({
            "touched_target_band": [True, True, False],
            "minutes_until_touch": [30, 15, None],
            "reversion_to_touch_band_pct": [1.17, 0.2, None],
            "score_15m": [51.5, 75, 80],
        })
        result = scenario_counts(frame, 30, 0.5)
        self.assertEqual(result["observed_cases"], 1)
        self.assertEqual(result["observed_with_score_15m_lt_70"], 1)
        self.assertEqual(result["first_bar_time_ambiguous"], 0)


if __name__ == "__main__":
    unittest.main()
