"""La vista breve conserva el cálculo de Bollinger anterior al recorte."""

import unittest

import pandas as pd

from daily_chart import build_daily_chart


class DailyChartTests(unittest.TestCase):
    def test_twenty_second_marker_is_visible_with_historical_session_candles(self):
        candles = pd.DataFrame([{
            "ticker": "AAL", "bar_index": index,
            "session": "PREVIA" if index < 25 else "ACTUAL",
            "time": "15:45" if index < 25 else "09:30",
            "open": 100.0 if index < 25 else 102.0,
            "high": 101.0 if index < 25 else 103.0,
            "low": 99.0 if index < 25 else 101.0,
            "close": 100.0 if index < 25 else 102.5,
        } for index in range(26)])
        chart = build_daily_chart(candles, "AAL", 2.0, -1, float("nan"),
                                  cutoff_seconds=20, cutoff_price=102.2,
                                  cutoff_upper=101.4, cutoff_lower=98.6,
                                  focus_opening=True)
        traces = {trace.name: trace for trace in chart.data}

        self.assertIn("CORTE 20 S", chart.layout.title.text)
        self.assertEqual(float(traces["Precio 20 s"].y[0]), 102.2)
        self.assertEqual(float(traces["Superior provisional 20 s"].y[0]), 101.4)
        self.assertLess(float(traces["Precio 20 s"].x[0]), 25)

    def test_opening_focus_keeps_four_prior_and_eight_current_candles(self):
        candles = pd.DataFrame([
            {
                "ticker": "AAL", "bar_index": index,
                "session": "PREVIA" if index < 26 else "ACTUAL",
                "time": f"{9 + (index % 26) // 4:02d}:{((index % 26) % 4) * 15:02d}",
                "open": 100.0 if index < 26 else 102.0,
                "high": 100.5 if index < 26 else 102.5,
                "low": 99.5 if index < 26 else 101.5,
                "close": 100.0 if index < 26 else 102.0,
            }
            for index in range(52)
        ])
        full = build_daily_chart(candles, "AAL", 2.0, -1, float("nan"))
        focused = build_daily_chart(candles, "AAL", 2.0, -1, float("nan"),
                                    focus_opening=True)

        self.assertEqual(list(focused.data[0].x), list(range(22, 34)))
        self.assertEqual(len(full.data[0].x), 52)
        self.assertAlmostEqual(float(focused.data[1].y[0]),
                               float(full.data[1].y[22]))
        self.assertEqual(list(focused.layout.xaxis.range), [21.5, 33.5])


if __name__ == "__main__":
    unittest.main()
