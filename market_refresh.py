"""Actualización de velas de la operativa sin bloquear el dashboard."""

import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from config import TICKERS
from daily_store import load_daily_session, save_daily_session
from real_data import available_days, download_bars, scan_real_day
from signal_audit import record_availability


NEW_YORK = ZoneInfo("America/New_York")


class MarketRefresh:
    def __init__(self, day: str, tickers: tuple[str, ...] = TICKERS):
        self.day = day
        self.tickers = tickers
        self._lock = threading.Lock()
        self._running = False
        self._last_finished = 0.0
        self._version = 0
        self._missing: tuple[str, ...] = ()
        self._status = "Pendiente de actualización de velas"

    def start(self, force: bool = False) -> None:
        with self._lock:
            if self._running or (not force and
                    time.monotonic() - self._last_finished < 60):
                return
            self._running = True
            self._status = "Actualizando velas de mercado en segundo plano"
        threading.Thread(target=self._run, daemon=True,
                         name="efecto-iman-market-refresh").start()

    def _run(self) -> None:
        started_at = datetime.now(NEW_YORK)
        try:
            bars, missing_15m = download_bars(period="10d", tickers=self.tickers)
            try:
                minute_bars, missing_1m = download_bars(
                    period="1d", tickers=self.tickers, interval="1m")
            except Exception:
                minute_bars, missing_1m = {}, self.tickers
            missing = tuple(f"15m {ticker}" for ticker in missing_15m) + tuple(
                f"1m {ticker}" for ticker in missing_1m)
            available = set(available_days(bars)) | set(available_days(minute_bars))
            if self.day in available:
                setups, candles = scan_real_day(bars, self.day, minute_bars)
                if not candles.empty:
                    fresh_setups = setups
                    previous, _, _ = load_daily_session(self.day)
                    if not previous.empty:
                        stages = {"PRE_OPEN": 0, "09:30": 1, "09:31": 2,
                                  "09:35": 3, "09:40": 4, "09:45": 5}
                        rows = {row["ticker"]: row for row in previous.to_dict("records")}
                        for row in setups.to_dict("records"):
                            old = rows.get(row["ticker"])
                            if old is None or stages.get(str(row.get("signal_stage")), 0) >= stages.get(
                                    str(old.get("signal_stage")), 0):
                                rows[row["ticker"]] = row
                        setups = setups.__class__.from_records(list(rows.values()),
                                                               columns=list(setups.columns))
                    save_daily_session(self.day, setups, candles)
                    try:
                        record_availability(self.day, "velas_1m_15m", fresh_setups,
                                            started_at)
                    except OSError:
                        pass  # El registro auxiliar no debe impedir actualizar las velas.
            with self._lock:
                self._missing = missing
                self._version += 1
                self._status = ("Velas actualizadas" if self.day in available else
                                "Aún no hay velas de la sesión actual")
        except Exception as exc:
            with self._lock:
                self._status = f"No se pudieron actualizar las velas: {exc}"
        finally:
            with self._lock:
                self._running = False
                self._last_finished = time.monotonic()

    def snapshot(self) -> tuple[str, tuple[str, ...], int]:
        with self._lock:
            return self._status, self._missing, self._version
