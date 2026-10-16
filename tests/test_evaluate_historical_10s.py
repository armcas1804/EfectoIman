import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd
import numpy as np

from evaluate_historical_10s import load_joined_examples, prior_band_dynamics


def _fixture(tmp_path, trade_time="2026-10-09T09:30:09-04:00"):
    folder = tmp_path / "snapshots"
    folder.mkdir()
    pd.DataFrame([{
        "date": "2026-10-09", "ticker": "AAL",
        "last_trade_at_et": trade_time, "price_at_10s": 102.0,
        "last_trade_conditions": '["@", "I"]',
    }]).to_csv(folder / "apertura_historica_10s_2026-10-09.csv", index=False)
    database = tmp_path / "diagnostics.db"
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("CREATE TABLE daily_diagnostics "
                   "(session_date TEXT, ticker TEXT, payload_json TEXT)")
        db.execute("CREATE TABLE daily_candles "
                   "(session_date TEXT, ticker TEXT, session TEXT, "
                   "bar_index INTEGER, close REAL)")
        db.execute("INSERT INTO daily_diagnostics VALUES (?, ?, ?)", (
            "2026-10-09", "AAL", json.dumps({
                "score_15m": 72, "prior_close": 100,
                "prior_bb_upper": 101, "prior_bb_lower": 99,
                "prior_bb_width_pct": 2, "prior_bb_width_change": 0,
                "gap_direction": "DOWN",  # Debe recalcularse con el trade de 10 s.
            })))
    return folder, database


class HistoricalTenSecondTests(unittest.TestCase):
    def test_uses_ten_second_trade_and_prior_bands(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, database = _fixture(Path(directory))
            row = load_joined_examples(folder, database).iloc[0]
            self.assertAlmostEqual(row.gap_pct_10s, 2)
            self.assertAlmostEqual(row.distance_to_band_pct_10s,
                                   (102 / 101 - 1) * 100)
            self.assertEqual(row.direction_10s, "UP")
            self.assertEqual(row.label_15m_ge_70, 1)
            self.assertTrue(row.last_trade_odd_lot)
            self.assertAlmostEqual(row.last_trade_elapsed_s, 9)

    def test_rejects_trade_after_ten_second_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, database = _fixture(Path(directory),
                                        "2026-10-09T09:30:11-04:00")
            with self.assertRaisesRegex(ValueError,
                                        "fuera de los primeros 10 segundos"):
                load_joined_examples(folder, database)

    def test_provisional_angle_uses_previous_fifteen_minute_closes(self):
        prior = np.array([100 + i * 0.01 for i in range(24)])
        last = prior[-20:]
        upper = last.mean() + 2 * last.std(ddof=0)
        lower = last.mean() - 2 * last.std(ddof=0)
        slope, angle = prior_band_dynamics(prior, 102, upper, lower, "UP")
        self.assertGreater(slope, 0)
        self.assertGreater(angle, 0)
        _, flat_angle = prior_band_dynamics(prior, 100.24, upper, lower, "UP")
        self.assertLess(flat_angle, angle)


if __name__ == "__main__":
    unittest.main()
