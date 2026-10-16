"""La vista breve conserva el cálculo de Bollinger anterior al recorte."""

import unittest

import pandas as pd

from daily_chart import build_daily_chart


class DailyChartTests(unittest.TestCase):
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
