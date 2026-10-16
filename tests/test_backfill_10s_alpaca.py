"""La reconstrucción no utiliza operaciones posteriores al corte."""

import unittest

from backfill_10s_alpaca import market_window, summarize_trades


class HistoricalTenSecondTests(unittest.TestCase):
    def test_last_trade_before_cutoff_and_missing_symbol(self):
        trades = {"AAPL": [
            {"t": "2026-10-09T13:30:02.000Z", "p": 100, "c": ["@"]},
            {"t": "2026-10-09T13:30:09.500Z", "p": 101, "c": ["@"]},
            {"t": "2026-10-09T13:30:11.000Z", "p": 102, "c": ["@"]},
        ], "MSFT": [{"t": "2026-10-09T13:30:12.000Z", "p": 200}]}
        result = summarize_trades("2026-10-09", trades)
        self.assertEqual(result.ticker.tolist(), ["AAPL"])
        self.assertEqual(result.price_at_10s.tolist(), [101])
        self.assertEqual(result.trade_count.tolist(), [2])

    def test_market_window_uses_new_york_timezone(self):
        start, end = market_window("2026-10-09")
        self.assertEqual(start, "2026-10-09T13:30:00+00:00")
        self.assertEqual(end, "2026-10-09T13:30:10+00:00")


if __name__ == "__main__":
    unittest.main()
