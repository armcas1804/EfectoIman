import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import pandas as pd

from trial_variants import (add_low_volatility, live_quote_observations,
                            opening_reversal_table, read_live_cut_observations,
                            score_home_trial_b)


class TrialVariantsTests(unittest.TestCase):
    def test_live_cuts_ignore_quotes_received_after_each_deadline(self):
        opened = datetime(2026, 10, 22, 9, 30,
                          tzinfo=ZoneInfo("America/New_York"))
        quotes = pd.DataFrame([{
            "ticker": "AAA", "price": price,
            "quote_event_time": (opened + timedelta(seconds=second)).isoformat(),
            "quote_received_at": (opened + timedelta(seconds=second)).isoformat(),
        } for second, price in ((2, 102.0), (25, 101.5), (45, 100.5))])
        thirty = live_quote_observations(quotes, opened, 30)
        sixty = live_quote_observations(quotes, opened, 60)
        self.assertEqual(thirty.price.tolist(), [101.5])
        self.assertEqual(sixty.price.tolist(), [100.5])
        setups = pd.DataFrame([{"ticker": "AAA", "prior_close": 100.0}])
        self.assertTrue(opening_reversal_table(setups, thirty, 30).candidate.iloc[0])
        with TemporaryDirectory() as folder:
            quotes.to_csv(Path(folder) / "apertura_flujo_60s_2026-10-22.csv",
                          index=False)
            recovered = read_live_cut_observations("2026-10-22", 30, Path(folder))
            ten = read_live_cut_observations("2026-10-22", 10, Path(folder))
        self.assertEqual(recovered.price.tolist(), [101.5])
        self.assertEqual(ten.price.tolist(), [102.0])

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

    def test_home_b_score_rewards_low_stable_width_and_penalizes_expansion(self):
        setups = pd.DataFrame([
            {"ticker": "FLAT", "prior_close": 100.0,
             "prior_bb_upper": 100.0, "prior_bb_lower": 98.8,
             "prior_bb_width_pct": 1.2, "prior_bb_width_change": -0.02,
             "prior_target_band_slope_pct": 0.0,
             "cutoff_bb_upper": 99.9, "cutoff_bb_lower": 98.9,
             "target_angle_cutoff_deg": 0.0,
             "angle_cutoff_event_span_seconds": 20.0},
            {"ticker": "WIDE", "prior_close": 100.0,
             "prior_bb_upper": 100.5, "prior_bb_lower": 97.5,
             "prior_bb_width_pct": 3.0, "prior_bb_width_change": 0.10,
             "prior_target_band_slope_pct": 0.10,
             "cutoff_bb_upper": 100.5, "cutoff_bb_lower": 95.7,
             "target_angle_cutoff_deg": 30.0,
             "angle_cutoff_event_span_seconds": 20.0},
            {"ticker": "INSIDE", "prior_close": 100.0,
             "prior_bb_upper": 103.0, "prior_bb_lower": 99.0,
             "prior_bb_width_pct": 1.0, "prior_bb_width_change": 0.0,
             "prior_target_band_slope_pct": 0.0,
             "cutoff_bb_upper": 103.0, "cutoff_bb_lower": 102.0,
             "target_angle_cutoff_deg": 0.0,
             "angle_cutoff_event_span_seconds": 20.0},
            {"ticker": "REENTER", "prior_close": 100.0,
             "prior_bb_upper": 100.0, "prior_bb_lower": 98.8,
             "prior_bb_width_pct": 1.2, "prior_bb_width_change": 0.0,
             "prior_target_band_slope_pct": 0.0,
             "cutoff_bb_upper": 101.5, "cutoff_bb_lower": 100.5,
             "target_angle_cutoff_deg": 0.0,
             "angle_cutoff_event_span_seconds": 20.0},
            {"ticker": "DOWN", "prior_close": 100.0,
             "prior_bb_upper": 101.0, "prior_bb_lower": 99.5,
             "prior_bb_width_pct": 1.5, "prior_bb_width_change": 0.0,
             "prior_target_band_slope_pct": 0.0,
             "cutoff_bb_upper": 100.2, "cutoff_bb_lower": 99.2,
             "target_angle_cutoff_deg": 0.0,
             "angle_cutoff_event_span_seconds": 20.0},
            {"ticker": "MISSING", "prior_close": 100.0},
        ])
        observations = pd.DataFrame([
            {"ticker": ticker, "window_seconds": 30,
             "first_price": 98.0 if ticker == "DOWN" else 102.0,
             "price": 99.0 if ticker == "DOWN" else 101.0,
             "source": "test", "first_time": "09:30:01",
             "last_time": "09:30:29", "event_count": 2}
            for ticker in setups.ticker
        ])
        result = score_home_trial_b(
            setups, opening_reversal_table(setups, observations, 30))
        by_ticker = result.set_index("ticker")
        self.assertEqual(by_ticker.loc["FLAT", "score_home_b"], 100.0)
        self.assertTrue(by_ticker.loc["WIDE", "home_b_candidate"])
        self.assertLess(by_ticker.loc["WIDE", "score_home_b"], 60.0)
        self.assertGreater(by_ticker.loc["WIDE", "penalty_prior_lateral"], 0)
        self.assertGreater(by_ticker.loc["WIDE", "penalty_live_lateral"], 0)
        self.assertGreater(by_ticker.loc["WIDE", "penalty_live_expansion"], 0)
        self.assertTrue(by_ticker.loc["INSIDE", "candidate"])
        self.assertFalse(by_ticker.loc["INSIDE", "home_b_candidate"])
        self.assertTrue(pd.isna(by_ticker.loc["INSIDE", "score_home_b"]))
        self.assertTrue(by_ticker.loc["REENTER", "candidate"])
        self.assertFalse(by_ticker.loc["REENTER", "home_b_candidate"])
        self.assertLess(by_ticker.loc["REENTER", "distance_cut_outside_bb_pct"], 0)
        self.assertTrue(by_ticker.loc["DOWN", "home_b_candidate"])
        self.assertGreater(by_ticker.loc["DOWN", "distance_cut_outside_bb_pct"], 0)
        self.assertTrue(pd.isna(by_ticker.loc["MISSING", "score_home_b"]))


if __name__ == "__main__":
    unittest.main()
