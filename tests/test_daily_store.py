"""Persistencia de diagnósticos y actualización entre etapas."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import daily_store
from opening_scan import opening_setups
from test_opening_scan import candles, minute


class DailyStoreTests(unittest.TestCase):
    def test_roundtrip_and_stage_update(self) -> None:
        with TemporaryDirectory() as folder, patch.object(
                daily_store, "DATABASE_PATH", Path(folder) / "scanner.db"):
            early = opening_setups(candles(), opening_prices={"TEST": 102})
            daily_store.save_daily_session("2026-10-06", early, candles())
            loaded, _, _ = daily_store.load_daily_session("2026-10-06")
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.iloc[0].confirmed, -1)
            self.assertTrue(pd.isna(loaded.iloc[0].post_open_bb_width_pct))
            late_candles = candles(first=(102, 102.2, 101, 101.5))
            late = opening_setups(late_candles, minute(102, 101.8))
            daily_store.save_daily_session("2026-10-06", late, late_candles)
            loaded, saved_candles, _ = daily_store.load_daily_session("2026-10-06")
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.iloc[0].confirmed, 1)
            self.assertTrue(pd.notna(loaded.iloc[0].post_open_bb_width_pct))
            self.assertEqual(len(saved_candles), 27)
            exported, _ = daily_store.export_history()
            self.assertEqual(len(exported), 1)
            self.assertIn("failed_conditions", exported)


if __name__ == "__main__":
    unittest.main()
