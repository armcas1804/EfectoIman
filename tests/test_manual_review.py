"""Etiquetas persistentes y cortes históricos sin información futura."""

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import pandas as pd

from backfill_30s_alpaca import summarize_trajectory, summarize_windows
from manual_review import (early_features, early_observations, early_trajectory,
                           load_reviews, save_review)


class ManualReviewTests(unittest.TestCase):
    def test_labels_are_explicit_and_persistent(self):
        with TemporaryDirectory() as folder:
            database = Path(folder) / "reviews.db"
            save_review("2026-10-09", "AAL", "Buena", database)
            self.assertEqual(load_reviews(database).iloc[0].label, "Buena")
            save_review("2026-10-09", "AAL", "Excelente", database)
            self.assertEqual(load_reviews(database).iloc[0].label, "Excelente")
            save_review("2026-10-09", "AAL", "Mala", database)
            self.assertEqual(load_reviews(database).iloc[0].label, "Mala")
            save_review("2026-10-09", "AAL", None, database)
            self.assertTrue(load_reviews(database).empty)

    def test_historical_windows_never_use_later_trade(self):
        with TemporaryDirectory() as folder:
            snapshot_dir = Path(folder)
            ny = ZoneInfo("America/New_York")
            opened = datetime(2026, 10, 9, 9, 30, tzinfo=ny)
            trades = [{"t": opened.isoformat(), "p": 100},
                      {"t": opened.replace(second=9).isoformat(), "p": 101},
                      {"t": opened.replace(second=19).isoformat(), "p": 102},
                      {"t": opened.replace(second=29).isoformat(), "p": 103}]
            result = summarize_windows("2026-10-09", {"AAL": trades})
            self.assertEqual(result.price_at_cutoff.tolist(), [101, 102, 103])
            result.to_csv(snapshot_dir / "apertura_historica_30s_2026-10-09.csv",
                          index=False)
            observed = early_observations("2026-10-09", snapshot_dir)
            self.assertEqual(observed.price.tolist(), [101, 102, 103])
            self.assertEqual(observed.window_seconds.tolist(), [10, 20, 30])
            path = summarize_trajectory("2026-10-09", {"AAL": trades})
            path.to_csv(snapshot_dir / "apertura_historica_trayectoria_30s_2026-10-09.csv",
                        index=False)
            early_path = early_trajectory("2026-10-09", "AAL", snapshot_dir)
            self.assertEqual(early_path.loc[
                early_path.second_after_open <= 10, "price"].tolist(), [100, 101])

    def test_bands_at_ten_seconds_ignore_later_prices(self):
        with TemporaryDirectory() as folder:
            snapshot_dir = Path(folder)
            ny = ZoneInfo("America/New_York")
            opened = datetime(2026, 10, 9, 9, 30, tzinfo=ny)
            bars = pd.DataFrame([{
                "ticker": "AAL", "bar_index": i, "session": "PREVIA",
                "time": f"{9 + i // 4:02d}:{(i % 4) * 15:02d}",
                "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            } for i in range(26)])
            result = summarize_windows("2026-10-09", {"AAL": [
                {"t": opened.isoformat(), "p": 101},
                {"t": opened.replace(second=9).isoformat(), "p": 102},
                {"t": opened.replace(second=20).isoformat(), "p": 200},
            ]})
            result.to_csv(snapshot_dir / "apertura_historica_30s_2026-10-09.csv",
                          index=False)
            observations = early_observations("2026-10-09", snapshot_dir)
            features = early_features("2026-10-09", "AAL", bars, observations)
            ten = features.loc[features.window_seconds == 10].iloc[0]
            self.assertEqual(ten.price, 102)
            self.assertLess(ten.upper_provisional, 110)


if __name__ == "__main__":
    unittest.main()
