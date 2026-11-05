import unittest

import pandas as pd

from trial_variants import add_low_volatility, opening_reversal_table


class TrialVariantsTests(unittest.TestCase):
    def test_low_volatility_uses_only_earlier_sessions(self):
        days = pd.date_range("2026-08-01", periods=12).strftime("%Y-%m-%d")
        history = pd.DataFrame({"session_date": days, "ticker": "AAA",
                                "prior_bb_width_pct": [1.0] * 10 + [5.0, .8]})
        model = pd.DataFrame({"session_date": days[-2:], "ticker": "AAA",
                              "prior_width_pct": [5.0, .8],
                              "trial_candidate_20s": [True, True],
                              "trial_signal_status_20s": ["Señal", "Señal"],
                              "trial_failed_conditions_20s": ["", ""]})
        actual = add_low_volatility(model, history, max_width_pct=1.5)
        self.assertEqual(actual.trial_width_reference_pct.tolist(), [1.0, 1.0])
        self.assertEqual(actual.trial_candidate_20s.tolist(), [False, True])
        self.assertEqual(actual.trial_width_status.iloc[0],
                         "Ancho BB supera el máximo absoluto")

    def test_reversal_is_observed_at_requested_cut_and_works_both_directions(self):
        setups = pd.DataFrame({"ticker": ["UP", "DOWN", "LATE"],
                               "prior_close": [100.0, 100.0, 100.0]})
        observations = pd.DataFrame([
            {"ticker": "UP", "window_seconds": 20, "first_price": 102.0,
             "price": 101.5, "source": "test", "first_time": "09:30:00",
             "last_time": "09:30:19", "event_count": 2},
            {"ticker": "DOWN", "window_seconds": 20, "first_price": 98.0,
             "price": 98.5, "source": "test", "first_time": "09:30:00",
             "last_time": "09:30:19", "event_count": 2},
            {"ticker": "LATE", "window_seconds": 30, "first_price": 102.0,
             "price": 101.0, "source": "test", "first_time": "09:30:00",
             "last_time": "09:30:29", "event_count": 2},
        ])
        result = opening_reversal_table(setups, observations, 20, 1.0)
        by_ticker = result.set_index("ticker")
        self.assertTrue(by_ticker.loc["UP", "candidate"])
        self.assertTrue(by_ticker.loc["DOWN", "candidate"])
        self.assertFalse(by_ticker.loc["LATE", "candidate"])
        self.assertGreater(by_ticker.loc["UP", "reversal_pct"], 0)
        self.assertGreater(by_ticker.loc["DOWN", "reversal_pct"], 0)
        self.assertEqual(set(result.ticker.tolist()[:2]), {"UP", "DOWN"})


if __name__ == "__main__":
    unittest.main()
