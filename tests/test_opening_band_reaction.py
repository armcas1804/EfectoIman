"""Opening band reaction distinguishes a prior-band jump from a live slope."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

import opening_band_reaction
from test_opening_scan import candles


class OpeningBandReactionTests(unittest.TestCase):
    day = "2026-10-13"

    def write_capture(self, folder: Path) -> datetime:
        opened = datetime(2026, 10, 13, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        snapshots = folder / "snapshots"
        snapshots.mkdir()
        pd.DataFrame([{
            "ticker": "TEST", "price_at_cutoff": 102.0,
            "quote_event_time": (opened + timedelta(seconds=8)).isoformat(),
            "quote_received_at": (opened + timedelta(seconds=9)).isoformat(),
        }]).to_csv(snapshots / f"apertura_10s_{self.day}.csv", index=False)
        return opened

    def test_cutoff_only_has_band_reaction_but_no_post_open_slope(self) -> None:
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            self.write_capture(folder)
            with patch.object(opening_band_reaction, "DATA_DIR", folder), patch.object(
                    opening_band_reaction, "load_daily_session",
                    return_value=(pd.DataFrame(), candles(), "updated")):
                result = opening_band_reaction.build_reaction(self.day, ("TEST",))
            self.assertEqual(len(result), 1)
            self.assertGreater(result.iloc[0].directional_reaction_pct, 0)
            self.assertTrue(pd.isna(result.iloc[0].target_slope_pct_per_min))

    def test_slope_uses_only_distinct_events_received_by_each_deadline(self) -> None:
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            opened = self.write_capture(folder)
            records = []
            for seconds, received, price in ((0, 1, 101), (8, 9, 102), (20, 21, 103)):
                records.append({
                    "ticker": "TEST", "price": price,
                    "quote_event_time": (opened + timedelta(seconds=seconds)).isoformat(),
                    "quote_received_at": (opened + timedelta(seconds=received)).isoformat(),
                })
            pd.DataFrame(records).to_csv(
                folder / "snapshots" / f"apertura_flujo_30s_{self.day}.csv", index=False)
            with patch.object(opening_band_reaction, "DATA_DIR", folder), patch.object(
                    opening_band_reaction, "load_daily_session",
                    return_value=(pd.DataFrame(), candles(), "updated")):
                result = opening_band_reaction.build_reaction(self.day, ("TEST",))
            ten, twenty, thirty = list(result.itertuples(index=False))
            self.assertEqual([ten.quote_count, twenty.quote_count, thirty.quote_count],
                             [2, 2, 3])
            self.assertAlmostEqual(ten.event_span_seconds, 8)
            self.assertAlmostEqual(twenty.event_span_seconds, 8)
            self.assertAlmostEqual(thirty.event_span_seconds, 20)
            self.assertGreater(ten.target_slope_pct_per_min, 0)
            self.assertGreater(thirty.target_slope_pct_per_min, 0)

    def test_repeated_market_timestamp_does_not_invent_a_slope(self) -> None:
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            opened = self.write_capture(folder)
            pd.DataFrame([{
                "ticker": "TEST", "price": price,
                "quote_event_time": opened.isoformat(),
                "quote_received_at": (opened + timedelta(seconds=received)).isoformat(),
            } for received, price in ((1, 101), (9, 102))]).to_csv(
                folder / "snapshots" / f"apertura_flujo_30s_{self.day}.csv", index=False)
            with patch.object(opening_band_reaction, "DATA_DIR", folder), patch.object(
                    opening_band_reaction, "load_daily_session",
                    return_value=(pd.DataFrame(), candles(), "updated")):
                ten = opening_band_reaction.build_reaction(self.day, ("TEST",)).iloc[0]
            self.assertEqual(ten.measurement_status, "NO_DISTINCT_EVENT_TIMES")
            self.assertTrue(pd.isna(ten.target_slope_pct_per_min))

    def test_later_window_can_measure_without_a_ten_second_quote(self) -> None:
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            opened = self.write_capture(folder)
            pd.DataFrame(columns=["ticker", "price_at_cutoff", "quote_event_time",
                                  "quote_received_at"]).to_csv(
                folder / "snapshots" / f"apertura_10s_{self.day}.csv", index=False)
            pd.DataFrame([{
                "ticker": "TEST", "price": price,
                "quote_event_time": (opened + timedelta(seconds=event)).isoformat(),
                "quote_received_at": (opened + timedelta(seconds=received)).isoformat(),
            } for event, received, price in ((11, 12, 101), (17, 18, 102))]).to_csv(
                folder / "snapshots" / f"apertura_flujo_30s_{self.day}.csv", index=False)
            with patch.object(opening_band_reaction, "DATA_DIR", folder), patch.object(
                    opening_band_reaction, "load_daily_session",
                    return_value=(pd.DataFrame(), candles(), "updated")):
                ten, twenty, thirty = opening_band_reaction.build_reaction(
                    self.day, ("TEST",)).itertuples(index=False)
            self.assertEqual(ten.measurement_status, "NO_QUOTE")
            self.assertTrue(pd.isna(ten.target_angle_deg_5m_equiv))
            self.assertEqual(twenty.gap_direction_at_window, "UP")
            self.assertTrue(pd.notna(twenty.target_angle_deg_5m_equiv))
            self.assertTrue(pd.notna(thirty.target_angle_deg_5m_equiv))


if __name__ == "__main__":
    unittest.main()
