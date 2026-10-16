"""El estudio temporal no debe aprender del mismo día que evalúa."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from evaluate_early_score import chronological_split, load_examples


class EarlyScoreValidationTests(unittest.TestCase):
    def test_load_excludes_unfinished_day_and_inside_band(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "study.db"
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("CREATE TABLE daily_diagnostics "
                           "(session_date TEXT, ticker TEXT, payload_json TEXT)")
                base = {"ticker": "TEST", "score_15m": 72,
                        "score_open": 48, "gap_pct": 1.5,
                        "gap_direction": "UP", "distance_to_band_pct": 0.8,
                        "prior_bb_width_pct": 1.0,
                        "prior_target_band_slope_pct": 0.01,
                        "prior_bb_width_change": 0.01}
                for day, distance in (("2026-10-09", 0.8),
                                      ("2026-10-08", -0.2),
                                      ("2026-10-12", 0.8)):
                    row = dict(base, distance_to_band_pct=distance)
                    db.execute("INSERT INTO daily_diagnostics VALUES (?,?,?)",
                               (day, row["ticker"], json.dumps(row)))
            frame = load_examples(database, today="2026-10-12")
            self.assertEqual(frame.date.tolist(), ["2026-10-09"])
            self.assertEqual(frame.label_15m_ge_70.tolist(), [1])

    def test_split_separates_entire_dates(self):
        frame = pd.DataFrame({"date": ["1", "1", "2", "2", "3", "3"],
                              "label_15m_ge_70": [0, 1, 1, 0, 0, 1]})
        train, test = chronological_split(frame, test_days=1, min_train_days=2)
        self.assertEqual(set(train.date), {"1", "2"})
        self.assertEqual(set(test.date), {"3"})


if __name__ == "__main__":
    unittest.main()
