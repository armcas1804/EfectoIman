"""The opening audit must use timely quotes and exclude later candles."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

import audit_live_10s
from opening_scan import opening_setups
from test_opening_scan import candles


class AuditLiveTenSecondsTests(unittest.TestCase):
    day = "2026-10-13"

    def make_quote_file(self, folder: Path, seconds: int) -> None:
        opened = datetime(2026, 10, 13, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        path = folder / f"apertura_10s_{self.day}.csv"
        pd.DataFrame([{
            "ticker": "TEST", "price_at_cutoff": 102.0,
            "quote_event_time": opened.isoformat(),
            "quote_received_at": (opened + timedelta(seconds=seconds)).isoformat(),
        }]).to_csv(path, index=False)

    def test_quote_received_after_cutoff_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            self.make_quote_file(folder, 11)
            with patch.object(audit_live_10s, "SNAPSHOTS", folder):
                with self.assertRaisesRegex(ValueError, "fuera de los diez segundos"):
                    audit_live_10s.timely_quotes(self.day)

    def test_future_candle_does_not_change_frozen_opening_score(self) -> None:
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            self.make_quote_file(folder, 9)
            prior = candles(prior_step=0.002)
            future = candles(prior_step=0.002, first=(102, 110, 90, 105))
            expected = opening_setups(prior, opening_prices={"TEST": 102.0},
                                      cutoff_seconds=10).iloc[0]
            with patch.object(audit_live_10s, "SNAPSHOTS", folder), patch.object(
                    audit_live_10s, "DATA_DIR", folder), patch.object(
                    audit_live_10s, "TICKERS", ("TEST",)), patch.object(
                    audit_live_10s, "load_daily_session",
                    return_value=(pd.DataFrame(), future, "updated")):
                frozen, _ = audit_live_10s.freeze_opening(self.day)
            self.assertEqual(frozen.iloc[0].score_open, expected.score_open)
            self.assertEqual(frozen.iloc[0].score_target_angle_component, 0.0)


if __name__ == "__main__":
    unittest.main()
