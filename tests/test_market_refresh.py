"""La descarga periódica no bloquea la interfaz ni arranca dos veces."""

import unittest
from unittest.mock import patch

import pandas as pd

import market_refresh
from market_refresh import MarketRefresh


class MarketRefreshTests(unittest.TestCase):
    def test_start_schedules_only_one_background_worker(self) -> None:
        refresh = MarketRefresh("2026-10-09", ("TEST",))
        with patch.object(market_refresh.threading.Thread, "start") as start:
            refresh.start()
            refresh.start()
        start.assert_called_once()
        self.assertIn("segundo plano", refresh.snapshot()[0])

    def test_worker_saves_available_day_and_reports_missing_bars(self) -> None:
        refresh = MarketRefresh("2026-10-09", ("TEST",))
        with patch.object(market_refresh, "download_bars", side_effect=[
                ({"TEST": pd.DataFrame()}, ()), ({}, ("TEST",))]), patch.object(
                    market_refresh, "available_days", return_value=["2026-10-09"]), patch.object(
                    market_refresh, "scan_real_day", return_value=(
                        pd.DataFrame({"ticker": ["TEST"]}),
                        pd.DataFrame({"ticker": ["TEST"]}))), patch.object(
                    market_refresh, "load_daily_session", return_value=(
                        pd.DataFrame(), pd.DataFrame(), None)), patch.object(
                    market_refresh, "save_daily_session") as save, patch.object(
                    market_refresh, "record_availability"):
            refresh._run()
        status, missing, version = refresh.snapshot()
        self.assertEqual(status, "Velas actualizadas")
        self.assertEqual(missing, ("1m TEST",))
        self.assertEqual(version, 1)
        save.assert_called_once()

    def test_partial_download_keeps_existing_diagnostics(self) -> None:
        refresh = MarketRefresh("2026-10-09", ("TEST", "OLD"))
        recent = pd.DataFrame([{"ticker": "TEST", "signal_stage": "09:30", "score": 60}])
        previous = pd.DataFrame([{"ticker": "OLD", "signal_stage": "09:35", "score": 70},
                                 {"ticker": "TEST", "signal_stage": "09:35", "score": 80}])
        with patch.object(market_refresh, "download_bars", side_effect=[
                ({"TEST": pd.DataFrame()}, ("OLD",)), ({}, ("TEST", "OLD"))]), patch.object(
                    market_refresh, "available_days", return_value=["2026-10-09"]), patch.object(
                    market_refresh, "scan_real_day", return_value=(
                        recent, pd.DataFrame({"ticker": ["TEST"]}))), patch.object(
                    market_refresh, "load_daily_session", return_value=(
                        previous, pd.DataFrame(), "saved")), patch.object(
                    market_refresh, "save_daily_session") as save, patch.object(
                    market_refresh, "record_availability"):
            refresh._run()
        merged = save.call_args.args[1].set_index("ticker")
        self.assertEqual(set(merged.index), {"OLD", "TEST"})
        self.assertEqual(merged.loc["TEST", "score"], 80)


if __name__ == "__main__":
    unittest.main()
