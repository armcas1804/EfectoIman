"""Los resultados descriptivos respetan dirección y momento de confirmación."""

import unittest

import pandas as pd

from outcomes import build_outcomes


class OutcomeTests(unittest.TestCase):
    def test_down_gap_uses_upward_excursion_and_skips_opening_bar_when_confirmed(self) -> None:
        setups = pd.DataFrame([{
            "session_date": "2026-10-06", "source": "real", "ticker": "TEST",
            "gap_direction": "DOWN", "candidate_at_open": True, "confirmed": 1,
            "touched_target_band": True, "touch_time": "10:00",
            "touch_status": "OBSERVED_15M", "opening_price": 98.0,
            "reversion_to_touch_band_pct": 1.0,
            "band_at_or_beyond_open_at_touch": False,
        }])
        candles = pd.DataFrame([
            {"session_date": "2026-10-06", "source": "real", "ticker": "TEST",
             "session": "ACTUAL", "bar_index": 26, "high": 110.0,
             "low": 96.0, "close": 97.0},
            {"session_date": "2026-10-06", "source": "real", "ticker": "TEST",
             "session": "ACTUAL", "bar_index": 27, "high": 100.0,
             "low": 97.0, "close": 99.0},
        ])
        result = build_outcomes(setups, candles).iloc[0]
        self.assertAlmostEqual(result.move_to_last_close_pct, (99 / 98 - 1) * 100)
        self.assertAlmostEqual(result.best_observed_excursion_pct, (100 / 98 - 1) * 100)


if __name__ == "__main__":
    unittest.main()
