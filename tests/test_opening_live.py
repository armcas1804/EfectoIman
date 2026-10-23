"""El corte de diez segundos usa únicamente precios observados a tiempo."""

from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

import opening_live
from opening_live import (CaptureProgress, OpeningCapture, capture_times,
                          next_capture_day, parse_quote)
from test_opening_scan import candles


class OpeningLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.day = date(2026, 10, 9)
        _, self.opened, self.cutoff = capture_times(self.day)
        self.symbols = {"TEST": "TEST"}

    def message(self, seconds: int, price: float = 103) -> dict:
        return {"id": "TEST", "price": price,
                "time": int((self.opened + timedelta(seconds=seconds)).timestamp() * 1000)}

    def test_next_session_is_scheduled_before_browser_needs_to_refresh(self) -> None:
        new_york = ZoneInfo("America/New_York")
        self.assertEqual(next_capture_day(self.opened.replace(
            day=11, hour=18, minute=0, tzinfo=new_york)), date(2026, 10, 12))
        self.assertEqual(next_capture_day(self.opened.replace(
            day=12, hour=9, minute=25, tzinfo=new_york)), date(2026, 10, 12))
        self.assertEqual(next_capture_day(self.opened.replace(
            day=12, hour=9, minute=30, second=11, tzinfo=new_york)),
            date(2026, 10, 13))

    def test_quote_must_be_observed_and_received_in_first_ten_seconds(self) -> None:
        valid = parse_quote(self.message(7), self.opened + timedelta(seconds=8),
                            self.day, self.symbols)
        self.assertEqual(valid.price, 103)
        self.assertIsNone(parse_quote(self.message(-1), self.opened + timedelta(seconds=1),
                                      self.day, self.symbols))
        self.assertIsNone(parse_quote(self.message(5), self.cutoff + timedelta(seconds=1),
                                      self.day, self.symbols))
        self.assertIsNone(parse_quote(self.message(11), self.cutoff + timedelta(seconds=1),
                                      self.day, self.symbols))
        self.assertEqual(parse_quote(self.message(8), self.opened + timedelta(seconds=7),
                                     self.day, self.symbols).price, 103)
        self.assertIsNone(parse_quote(self.message(8), self.opened + timedelta(seconds=4),
                                      self.day, self.symbols))
        self.assertIsNone(parse_quote(self.message(5, float("nan")), self.cutoff,
                                      self.day, self.symbols))
        self.assertIsNotNone(parse_quote(self.message(15),
                                           self.opened + timedelta(seconds=16),
                                           self.day, self.symbols, window_seconds=30))
        self.assertIsNotNone(parse_quote(self.message(89),
                                           self.opened + timedelta(seconds=89),
                                           self.day, self.symbols, window_seconds=90))
        self.assertIsNone(parse_quote(self.message(91),
                                      self.opened + timedelta(seconds=91),
                                      self.day, self.symbols, window_seconds=90))

    def test_later_quotes_are_recorded_without_changing_ten_second_cut(self) -> None:
        capture = OpeningCapture(self.day, ("TEST",))
        capture._prior_candles = candles().loc[
            lambda frame: frame.session == "PREVIA"].copy()
        capture.accept_message(self.message(2, 102), self.opened + timedelta(seconds=3))
        capture.accept_message(self.message(8, 103), self.opened + timedelta(seconds=9))
        with TemporaryDirectory() as folder, patch.object(
                opening_live, "DATA_DIR", Path(folder)), patch.object(
                opening_live, "save_daily_session"), patch.object(
                opening_live, "record_availability"):
            capture._freeze()
            capture.accept_message(self.message(15, 104),
                                   self.opened + timedelta(seconds=16))
            capture.accept_message(self.message(45, 105),
                                   self.opened + timedelta(seconds=46))
            capture._finish_observation()
            cutoff = pd.read_csv(Path(folder) / "snapshots" /
                                 "apertura_10s_2026-10-09.csv")
            sequence = pd.read_csv(Path(folder) / "snapshots" /
                                   "apertura_flujo_30s_2026-10-09.csv")
            sequence_minute = pd.read_csv(Path(folder) / "snapshots" /
                                          "apertura_flujo_60s_2026-10-09.csv")
            sequence_ninety = pd.read_csv(Path(folder) / "snapshots" /
                                          "apertura_flujo_90s_2026-10-09.csv")
        self.assertEqual(float(cutoff.iloc[0].price_at_cutoff), 103)
        self.assertEqual(sequence.price.tolist(), [102, 103, 104])
        self.assertEqual(sequence_minute.price.tolist(), [102, 103, 104, 105])
        self.assertEqual(sequence_ninety.price.tolist(), [102, 103, 104, 105])
        self.assertEqual(float(capture.snapshot()[0].iloc[0].opening_price), 103)

    def test_freeze_uses_latest_timely_quote_and_provisional_band(self) -> None:
        capture = OpeningCapture(self.day, ("TEST",))
        prior = candles().loc[lambda frame: frame.session == "PREVIA"].copy()
        prior.loc[:, ["open", "high", "low", "close"]] = [
            [100 + i * .02] * 4 for i in range(len(prior))]
        capture._prior_candles = prior
        capture.accept_message(self.message(2, 102), self.opened + timedelta(seconds=3))
        capture.accept_message(self.message(8, 103), self.opened + timedelta(seconds=9))
        capture.accept_message(self.message(9, 104), self.cutoff + timedelta(seconds=1))
        with TemporaryDirectory() as folder, patch.object(
                opening_live, "DATA_DIR", Path(folder)), patch.object(
                    opening_live, "save_daily_session") as save, patch.object(
                    opening_live, "record_availability") as audit:
            capture._freeze()
            setups, saved_candles, status, version = capture.snapshot()
            self.assertEqual(version, 1)
            self.assertEqual(float(setups.iloc[0].opening_price), 103)
            self.assertTrue(pd.notna(setups.iloc[0].cutoff_bb_upper))
            self.assertTrue(pd.notna(setups.iloc[0].target_angle_10s_deg))
            self.assertEqual(float(setups.iloc[0].angle_10s_event_span_seconds), 6)
            self.assertEqual(setups.iloc[0].signal_stage, "09:30")
            self.assertTrue(pd.isna(setups.iloc[0].target_angle_5m_deg))
            self.assertTrue(saved_candles.session.eq("PREVIA").all())
            self.assertIn("1/1", status)
            self.assertTrue((Path(folder) / "snapshots" /
                             "apertura_10s_2026-10-09.csv").exists())
            save.assert_called_once()
            audit.assert_called_once()
            progress = capture.progress()
            self.assertEqual(progress.frozen_count, 1)
            self.assertEqual(progress.prior_count, 1)
            self.assertIsNotNone(progress.ready_at)

    def test_thirty_sixty_and_ninety_second_lists_are_independent(self) -> None:
        capture = OpeningCapture(self.day, ("TEST",))
        prior = candles().loc[lambda frame: frame.session == "PREVIA"].copy()
        capture._prior_candles = prior
        for seconds, price in ((2, 102), (25, 103), (45, 104), (55, 105),
                               (80, 106), (95, 107)):
            capture.accept_message(self.message(seconds, price),
                                   self.opened + timedelta(seconds=seconds))
        events = capture._quote_events.copy()
        with TemporaryDirectory() as folder, patch.object(
                opening_live, "DATA_DIR", Path(folder)), patch.object(
                    opening_live, "record_availability"):
            capture._build_window(30, [q for q in events if q.received_at <=
                                       self.opened + timedelta(seconds=30)], prior)
            thirty, _, _ = capture.snapshot_at(30)
            capture._build_window(60, [q for q in events if q.received_at <=
                                       self.opened + timedelta(seconds=60)], prior)
            sixty, _, _ = capture.snapshot_at(60)
            capture._build_window(90, [q for q in events if q.received_at <=
                                       self.opened + timedelta(seconds=90)], prior)
            ninety, _, _ = capture.snapshot_at(90)
            folder_path = Path(folder) / "snapshots"
            thirty_quotes = pd.read_csv(folder_path / "apertura_30s_2026-10-09.csv")
            sixty_quotes = pd.read_csv(folder_path / "apertura_60s_2026-10-09.csv")
            ninety_quotes = pd.read_csv(folder_path / "apertura_90s_2026-10-09.csv")
            thirty_candidates = pd.read_csv(
                folder_path / "candidatos_apertura_30s_2026-10-09.csv")
        self.assertEqual(thirty_quotes.iloc[0].price_at_cutoff, 103)
        self.assertEqual(sixty_quotes.iloc[0].price_at_cutoff, 105)
        self.assertEqual(ninety_quotes.iloc[0].price_at_cutoff, 106)
        self.assertEqual(thirty.iloc[0].opening_price, 103)
        self.assertEqual(sixty.iloc[0].opening_price, 105)
        self.assertEqual(ninety.iloc[0].opening_price, 106)
        self.assertEqual(thirty.iloc[0].cutoff_seconds, 30)
        self.assertEqual(sixty.iloc[0].cutoff_seconds, 60)
        self.assertEqual(ninety.iloc[0].cutoff_seconds, 90)
        self.assertTrue(pd.notna(thirty.iloc[0].cutoff_bb_upper))
        self.assertTrue(pd.notna(sixty.iloc[0].target_angle_cutoff_deg))
        self.assertTrue(pd.notna(ninety.iloc[0].target_angle_cutoff_deg))
        self.assertEqual(len(thirty_candidates), int(thirty.iloc[0].candidate_at_open))

    def test_visual_status_distinguishes_capture_analysis_and_ready(self) -> None:
        from app import capture_message

        capture = OpeningCapture(date(2026, 10, 12), ("TEST",))
        self.assertEqual(capture.prepare_at.astimezone(
            ZoneInfo("America/Hermosillo")).strftime("%H:%M"), "06:25")
        self.assertEqual(capture.cutoff.astimezone(
            ZoneInfo("America/Hermosillo")).strftime("%H:%M:%S"), "06:30:10")
        progress = CaptureProgress("Preparando", 1, 0, None, True,
                                   None, None, None, None, None)
        tone, message = capture_message(capture.opened_at - timedelta(seconds=1),
                                        capture, progress)
        self.assertEqual(tone, "ready")
        self.assertIn("Preparación lista", message)
        progress = CaptureProgress("Capturando", 1, 1, None, True,
                                   None, None, None, None, None)
        tone, message = capture_message(capture.opened_at + timedelta(seconds=5),
                                        capture, progress)
        self.assertEqual(tone, "active")
        self.assertIn("1/1", message)
        self.assertIn("faltan 25 s", message)
        progress = CaptureProgress("Calculando", 1, 1, 1, False,
                                   capture.cutoff, None, None, None, None)
        tone, message = capture_message(capture.cutoff, capture, progress)
        self.assertEqual(tone, "active")
        self.assertIn("faltan 20 s", message)
        progress = CaptureProgress("Listo", 1, 1, 1, False,
                                   capture.cutoff, capture.cutoff + timedelta(seconds=2),
                                   1, None, None)
        tone, message = capture_message(capture.cutoff + timedelta(seconds=2),
                                        capture, progress)
        self.assertEqual(tone, "active")
        self.assertIn("30 s", message)
        empty = CaptureProgress("Listo", 1, 0, 0, False,
                                capture.cutoff, capture.cutoff + timedelta(seconds=2),
                                0, None, None)
        tone, message = capture_message(capture.cutoff + timedelta(seconds=2),
                                        capture, empty)
        self.assertEqual(tone, "error")
        self.assertIn("0/1", message)
        thirty = CaptureProgress("Listo", 1, 1, 1, True,
                                 capture.cutoff, capture.cutoff,
                                 1, None, None,
                                 {30: capture.opened_at + timedelta(seconds=34)},
                                 {30: 1}, {30: 1})
        tone, message = capture_message(capture.opened_at + timedelta(seconds=35),
                                        capture, thirty)
        self.assertEqual(tone, "active")
        self.assertIn("Lista de 30 s", message)
        minute = CaptureProgress("Listo", 1, 1, 1, False,
                                 capture.cutoff, capture.cutoff,
                                 1, None, None,
                                 {30: capture.opened_at + timedelta(seconds=34),
                                  60: capture.opened_at + timedelta(seconds=64)},
                                 {30: 1, 60: 1}, {30: 1, 60: 1})
        tone, message = capture_message(capture.opened_at + timedelta(seconds=65),
                                        capture, minute)
        self.assertEqual(tone, "active")
        self.assertIn("Lista de 1 minuto", message)
        ninety = CaptureProgress("Listo", 1, 1, 1, False,
                                 capture.cutoff, capture.cutoff,
                                 1, None, None,
                                 {30: capture.opened_at + timedelta(seconds=34),
                                  60: capture.opened_at + timedelta(seconds=64),
                                  90: capture.opened_at + timedelta(seconds=94)},
                                 {30: 1, 60: 1, 90: 1}, {30: 1, 60: 1, 90: 1})
        tone, message = capture_message(capture.opened_at + timedelta(seconds=95),
                                        capture, ninety)
        self.assertEqual(tone, "ready")
        self.assertIn("Lista de 1 minuto y medio", message)

    def test_missing_timely_quote_stays_missing_in_diagnostic(self) -> None:
        capture = OpeningCapture(self.day, ("TEST",))
        capture._prior_candles = candles().loc[
            lambda frame: frame.session == "PREVIA"].copy()
        with TemporaryDirectory() as folder, patch.object(
                opening_live, "DATA_DIR", Path(folder)), patch.object(
                    opening_live, "save_daily_session") as save:
            capture._freeze()
            setups, _, status, _ = capture.snapshot()
            self.assertEqual(setups.iloc[0].data_status, "MISSING_OPEN")
            self.assertEqual(setups.iloc[0].failed_conditions, "missing_open")
            self.assertIn("0/1", status)
            save.assert_not_called()

    def test_late_start_attempts_remaining_cuts(self) -> None:
        capture = OpeningCapture(self.day, ("TEST",))
        capture._started = True
        with patch.object(opening_live.threading.Thread, "start"):
            capture.poll(self.cutoff + timedelta(seconds=1))
        self.assertIn("Inicio tardío", capture.status)
        self.assertTrue(capture._prepared)
        self.assertIsNone(capture.snapshot()[0])

    def test_start_after_all_cuts_does_not_create_empty_snapshots(self) -> None:
        capture = OpeningCapture(self.day, ("TEST",))
        capture.observe_until = datetime.now(opening_live.NEW_YORK) - timedelta(seconds=1)
        capture._prior_candles = candles().loc[
            lambda frame: frame.session == "PREVIA"].copy()
        with TemporaryDirectory() as folder, patch.object(
                opening_live, "DATA_DIR", Path(folder)):
            capture.start()
            capture.poll()
            self.assertFalse(capture._started)
            self.assertEqual(capture.status,
                             "Los cortes de apertura no se capturaron en este proceso")
            self.assertIsNone(capture.snapshot_at(90)[0])
            self.assertFalse((Path(folder) / "snapshots").exists())

    def test_passed_cut_without_received_quotes_cannot_overwrite_snapshot(self) -> None:
        capture = OpeningCapture(self.day, ("TEST",))
        capture._prior_candles = candles().loc[
            lambda frame: frame.session == "PREVIA"].copy()
        with TemporaryDirectory() as folder, patch.object(
                opening_live, "DATA_DIR", Path(folder)):
            capture._start_window_builds(self.opened + timedelta(seconds=90))
            capture._finish_observation()
            self.assertEqual(capture._window_building, set())
            self.assertFalse((Path(folder) / "snapshots").exists())

    def test_poll_prepares_only_once_when_timer_is_late(self) -> None:
        capture = OpeningCapture(self.day, ("TEST",))
        capture._started = True
        with patch.object(opening_live.threading.Thread, "start") as start:
            capture.poll(self.opened - timedelta(minutes=1))
            capture.poll(self.opened - timedelta(seconds=30))
        self.assertEqual(start.call_count, 2)
        self.assertTrue(capture._prepared)


if __name__ == "__main__":
    unittest.main()
