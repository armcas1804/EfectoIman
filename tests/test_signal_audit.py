"""El registro conserva la primera hora disponible de una señal preliminar."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

import signal_audit


class SignalAuditTests(unittest.TestCase):
    def test_records_preliminary_and_preserves_first_time(self) -> None:
        with TemporaryDirectory() as folder, patch.object(
                signal_audit, "AUDIT_PATH", Path(folder) / "audit.csv"):
            started = datetime(2026, 10, 9, 9, 30, 10,
                               tzinfo=ZoneInfo("America/New_York"))
            setups = pd.DataFrame([
                {"ticker": "CVS", "candidate_at_open": True,
                 "candidate": False, "score_open": 82, "signal_stage": "09:30"},
                {"ticker": "OTHER", "candidate_at_open": False,
                 "candidate": False, "score_open": 50, "signal_stage": "09:30"},
            ])
            signal_audit.record_availability("2026-10-09", "corte_10s", setups,
                                             started)
            signal_audit.record_availability("2026-10-09", "velas_1m_15m", setups,
                                             started)
            records = pd.read_csv(signal_audit.AUDIT_PATH)
            self.assertEqual(list(records.ticker), ["CVS", "CVS"])
            self.assertEqual(signal_audit.first_availability("2026-10-09")["CVS"][1],
                             "corte_10s")
            self.assertNotIn("OTHER", signal_audit.first_availability("2026-10-09"))


if __name__ == "__main__":
    unittest.main()
