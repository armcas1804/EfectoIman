"""La gráfica de Home no debe usar minutos posteriores al instante observado."""

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from home_chart import build_home_chart, provisional_15m_bar


NY = ZoneInfo("America/New_York")


def prior_candles() -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": "AAL", "bar_index": index, "session": "PREVIA",
        "time": f"{9 + index // 4:02d}:{index % 4 * 15:02d}",
        "open": 100.0, "high": 100.3, "low": 99.7, "close": 100.0,
    } for index in range(26)])


class HomeChartTests(unittest.TestCase):
    def test_provisional_candle_uses_only_complete_minutes(self):
        minutes = pd.DataFrame([
            {"day": "2026-10-22", "time": "09:30", "open": 102.0,
             "high": 103.0, "low": 101.0, "close": 102.5},
            {"day": "2026-10-22", "time": "09:31", "open": 102.5,
             "high": 104.0, "low": 102.0, "close": 103.5},
            {"day": "2026-10-22", "time": "09:32", "open": 103.5,
             "high": 200.0, "low": 100.0, "close": 200.0},
        ])
        now = datetime(2026, 10, 22, 9, 32, 30, tzinfo=NY)
        bar = provisional_15m_bar(prior_candles(), "AAL", minutes, now)
        self.assertEqual(bar["last_minute"], "09:31")
        self.assertEqual(bar["high"], 104.0)
        self.assertEqual(bar["close"], 103.5)
        self.assertGreater(bar["bb_upper"], 100.0)
        self.assertLess(bar["bb_upper"], 110.0)

    def test_cutoff_is_a_point_and_bands_extend_from_prior_session(self):
        chart = build_home_chart(prior_candles(), "AAL", 30, 102.5,
                                 101.8, 98.9, 2.5)
        traces = {trace.name: trace for trace in chart.data}
        self.assertEqual(list(traces["Precio del corte"].y), [102.5])
        self.assertEqual(list(traces["Superior al corte"].y)[-1], 101.8)
        self.assertIn("Banda superior · BB(20,2)", traces)
        self.assertNotIn("Vela en formación", " ".join(traces))


if __name__ == "__main__":
    unittest.main()
