"""Sólo los precios recibidos dentro de diez segundos cuentan como históricos."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from subminute_history import ten_second_coverage


class TenSecondCoverageTests(unittest.TestCase):
    def test_missing_and_late_quotes_are_not_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "study.db"
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("CREATE TABLE daily_sessions (session_date TEXT, source TEXT)")
                db.execute("CREATE TABLE daily_diagnostics "
                           "(session_date TEXT, source TEXT, ticker TEXT)")
                db.executemany("INSERT INTO daily_sessions VALUES (?, 'real')",
                               [("2026-10-08",), ("2026-10-09",)])
                db.executemany("INSERT INTO daily_diagnostics VALUES (?, 'real', ?)",
                               [("2026-10-09", "AAPL"), ("2026-10-09", "MSFT")])
            (root / "apertura_10s_2026-10-09.csv").write_text(
                "ticker,price_at_cutoff,quote_event_time,quote_received_at\n"
                "AAPL,100,2026-10-09T09:30:04-04:00,2026-10-09T09:30:05-04:00\n"
                "MSFT,200,2026-10-09T09:30:11-04:00,2026-10-09T09:30:12-04:00\n",
                encoding="utf-8")
            (root / "apertura_historica_10s_2026-10-09.csv").write_text(
                "ticker,source,price_at_10s,last_trade_at_et\n"
                "AAPL,alpaca_sip_historical_trades,101,2026-10-09T09:30:09-04:00\n",
                encoding="utf-8")
            coverage = ten_second_coverage(database, root)
            self.assertEqual(coverage.ten_second_quotes.tolist(), [1, 0])
            self.assertEqual(coverage.status.tolist(), ["CAPTURA_10S", "SIN_CAPTURA"])
            self.assertEqual(coverage.historical_10s_quotes.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
