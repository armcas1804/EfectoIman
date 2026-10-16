"""La reproducción sólo toma cierres que ya existían en la hora de decisión."""

import unittest

import pandas as pd

from analyze_early_window import replay
from test_opening_scan import candles


class EarlyWindowReplayTests(unittest.TestCase):
    def test_future_minute_does_not_change_earlier_angle(self) -> None:
        prior = candles(prior_step=0.01).loc[lambda frame: frame.session == "PREVIA"]
        prior = prior.assign(day="2026-10-09")
        rows = [{"ticker": "TEST", "day": "2026-10-09", "time": f"09:{minute:02d}",
                 "open": 103.0, "high": 103.2, "low": 102.8,
                 "close": 103.0} for minute in range(30, 36)]
        initial = replay(pd.DataFrame(rows), prior)
        changed = pd.DataFrame(rows)
        changed.loc[changed.time == "09:34", ["open", "high", "low", "close"]] = 110
        revised = replay(changed, prior)
        self.assertFalse(initial.empty)
        a = initial.loc[initial.decision_time == "09:33"].iloc[0]
        b = revised.loc[revised.decision_time == "09:33"].iloc[0]
        self.assertEqual(a.angle_from_open_deg, b.angle_from_open_deg)
        self.assertEqual(a.angle_latest_minute_deg, b.angle_latest_minute_deg)


if __name__ == "__main__":
    unittest.main()
