"""Captura local de la apertura y diagnósticos a 10, 30, 60 y 90 segundos."""

import csv
import math
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config import DATA_DIR, PROVIDER_SYMBOLS
from daily_store import save_daily_session
from opening_scan import opening_setups
from real_data import download_bars, session_candles
from signal_audit import record_availability


NEW_YORK = ZoneInfo("America/New_York")
OPEN_TIME = time(9, 30)
PREPARE_MINUTES = 5
CAPTURE_SECONDS = 10
DISPLAY_WINDOWS = (30, 60, 90)
OBSERVE_SECONDS = max(DISPLAY_WINDOWS)
MAX_PROVIDER_CLOCK_SKEW_SECONDS = 3


@dataclass(frozen=True)
class OpeningQuote:
    ticker: str
    price: float
    event_time: datetime
    received_at: datetime


@dataclass(frozen=True)
class CaptureProgress:
    status: str
    prior_count: int
    quote_count: int
    frozen_count: int | None
    stream_connected: bool
    frozen_at: datetime | None
    ready_at: datetime | None
    candidate_count: int | None
    prior_error: str | None
    stream_error: str | None
    window_ready_at: dict[int, datetime] | None = None
    window_counts: dict[int, int] | None = None
    window_candidates: dict[int, int] | None = None


def capture_times(day: date) -> tuple[datetime, datetime, datetime]:
    opened = datetime.combine(day, OPEN_TIME, NEW_YORK)
    return (opened - timedelta(minutes=PREPARE_MINUTES), opened,
            opened + timedelta(seconds=CAPTURE_SECONDS))


def next_capture_day(now: datetime) -> date:
    """Programa el próximo día entre semana sin depender del refresco del navegador."""
    local = now.astimezone(NEW_YORK)
    day = local.date()
    if local >= capture_times(day)[2]:
        day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def parse_quote(message: dict, received_at: datetime, day: date,
                symbol_map: dict[str, str],
                window_seconds: int = CAPTURE_SECONDS) -> OpeningQuote | None:
    """Acepta precios recibidos en la ventana con desfase acotado del reloj Yahoo."""
    ticker = symbol_map.get(str(message.get("id", "")))
    if ticker is None or received_at.tzinfo is None:
        return None
    try:
        price = float(message["price"])
        epoch = float(message["time"])
        event_time = datetime.fromtimestamp(
            epoch / 1000 if epoch > 100_000_000_000 else epoch, NEW_YORK)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None
    _, opened, _ = capture_times(day)
    cutoff = opened + timedelta(seconds=window_seconds)
    received_at = received_at.astimezone(NEW_YORK)
    # La disponibilidad la determina la recepción local. Yahoo puede fechar
    # un mensaje unos segundos por delante del reloj del equipo.
    if (not math.isfinite(price) or price <= 0 or
            not opened <= received_at <= cutoff or
            not opened <= event_time <= received_at + timedelta(
                seconds=MAX_PROVIDER_CLOCK_SKEW_SECONDS)):
        return None
    return OpeningQuote(ticker, price, event_time, received_at)


class OpeningCapture:
    """Prepara velas previas y congela cortes independientes de apertura."""

    def __init__(self, day: date, tickers: tuple[str, ...]):
        self.day = day
        self.tickers = tickers
        self.prepare_at, self.opened_at, self.cutoff = capture_times(day)
        self.observe_until = self.opened_at + timedelta(seconds=OBSERVE_SECONDS)
        self._symbol_map = {PROVIDER_SYMBOLS.get(ticker, ticker): ticker
                            for ticker in tickers}
        self._lock = threading.RLock()
        self._quotes: dict[str, OpeningQuote] = {}
        self._quote_events: list[OpeningQuote] = []
        self._frozen_quotes: dict[str, OpeningQuote] | None = None
        self._sequence_saved = False
        self._prior_candles: pd.DataFrame | None = None
        self._setups: pd.DataFrame | None = None
        self._window_setups: dict[int, pd.DataFrame] = {}
        self._window_quotes: dict[int, dict[str, OpeningQuote]] = {}
        self._window_ready_at: dict[int, datetime] = {}
        self._window_building: set[int] = set()
        self._socket = None
        self._stream_connected = False
        self._frozen_at: datetime | None = None
        self._ready_at: datetime | None = None
        self._prior_error: str | None = None
        self._stream_error: str | None = None
        self._building = False
        self._started = False
        self._prepared = False
        self._missed_10s = False
        self.version = 0
        self.status = "Programado para las 09:30:10 ET"

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            now = datetime.now(NEW_YORK)
            if self.day.weekday() >= 5:
                self.status = "Sin apertura programada en fin de semana"
                return
            if now >= self.observe_until:
                self._started = False
                self.status = "Los cortes de apertura no se capturaron en este proceso"
                return
        threading.Thread(target=self._run, daemon=True, name="efecto-iman-open-windows").start()

    def _run(self) -> None:
        wait = max(0, (self.prepare_at - datetime.now(NEW_YORK)).total_seconds())
        if wait:
            threading.Event().wait(wait)
        self.poll()
        for seconds in (CAPTURE_SECONDS, *DISPLAY_WINDOWS):
            deadline = self.opened_at + timedelta(seconds=seconds)
            wait = max(0, (deadline - datetime.now(NEW_YORK)).total_seconds())
            if wait:
                threading.Event().wait(wait)
            self.poll()

    def poll(self, now: datetime | None = None) -> None:
        """Respaldo por hora real si el temporizador quedó suspendido."""
        now = now or datetime.now(NEW_YORK)
        with self._lock:
            if not self._started or self.day.weekday() >= 5:
                return
            if self._frozen_quotes is not None:
                finish = now >= self.observe_until and not self._sequence_saved
                freeze = prepare = False
            elif now >= self.cutoff:
                finish = False
                if not self._prepared:
                    self._prepared = True
                    self._missed_10s = True
                    self.status = "Inicio tardío: preparando los cortes que aún sean posibles"
                    freeze = False
                    prepare = True
                else:
                    freeze = not self._missed_10s
                    prepare = False
            else:
                finish = False
                freeze = False
                prepare = now >= self.prepare_at and not self._prepared
                if prepare:
                    self._prepared = True
                    self.status = "Preparando bandas previas y flujo de cotizaciones"
        if prepare:
            threading.Thread(target=self._load_prior, daemon=True,
                             name="efecto-iman-prior-bars").start()
            threading.Thread(target=self._listen, daemon=True,
                             name="efecto-iman-yahoo-quotes").start()
        if freeze:
            self._freeze()
        self._start_window_builds(now)
        if finish or now >= self.observe_until:
            self._finish_observation()

    def _load_prior(self) -> None:
        try:
            bars, _ = download_bars(period="10d", tickers=self.tickers)
            candles = session_candles(bars, self.day.isoformat())
            candles = candles.loc[candles.session == "PREVIA"].copy()
            if candles.empty:
                raise RuntimeError("No hay velas previas de 15 minutos")
            with self._lock:
                self._prior_candles = candles
                self._prior_error = None
            self._build_if_ready()
            self._start_window_builds(datetime.now(NEW_YORK))
        except Exception as exc:
            with self._lock:
                self._prior_error = str(exc)
                self.status = f"Sin bandas previas: {exc}"

    def _listen(self) -> None:
        symbols = list(self._symbol_map)
        while datetime.now(NEW_YORK) < self.observe_until:
            socket = None
            try:
                socket = yf.WebSocket(verbose=False)
                with self._lock:
                    self._socket = socket
                socket.subscribe(symbols)
                with self._lock:
                    if self._frozen_quotes is None:
                        self._stream_connected = True
                        self._stream_error = None
                        self.status = "Esperando cotizaciones de la apertura"
                socket.listen(self.accept_message)
            except Exception as exc:
                with self._lock:
                    if self._frozen_quotes is None:
                        self._stream_error = str(exc)
                        self.status = f"Reconectando cotizaciones: {exc}"
            finally:
                if socket is not None:
                    try:
                        socket.close()
                    except Exception:
                        pass
                with self._lock:
                    self._stream_connected = False
                    if self._socket is socket:
                        self._socket = None
            remaining = (self.observe_until - datetime.now(NEW_YORK)).total_seconds()
            if remaining > 0:
                threading.Event().wait(min(5, remaining))

    def accept_message(self, message: dict,
                       received_at: datetime | None = None) -> None:
        received_at = received_at or datetime.now(NEW_YORK)
        quote = parse_quote(message, received_at, self.day, self._symbol_map,
                            OBSERVE_SECONDS)
        if quote is None:
            return
        with self._lock:
            if not self._sequence_saved:
                self._quote_events.append(quote)
            if self._frozen_quotes is None and quote.received_at <= self.cutoff:
                previous = self._quotes.get(quote.ticker)
                if previous is None or quote.event_time >= previous.event_time:
                    self._quotes[quote.ticker] = quote

    def _freeze(self) -> None:
        with self._lock:
            if self._frozen_quotes is not None:
                return
            self._frozen_quotes = self._quotes.copy()
            self._frozen_at = datetime.now(NEW_YORK)
            self.status = (f"Corte 09:30:10 ET: {len(self._frozen_quotes)}/"
                           f"{len(self.tickers)} cotizaciones recibidas a tiempo")
        self._build_if_ready()

    def _finish_observation(self) -> None:
        """Guarda el flujo recibido hasta cada corte sin cambiar los anteriores."""
        with self._lock:
            if self._sequence_saved:
                return
            self._sequence_saved = True
            events = self._quote_events.copy()
            socket = self._socket
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass
        if not events:
            return
        try:
            folder = DATA_DIR / "snapshots"
            folder.mkdir(parents=True, exist_ok=True)
            for seconds in DISPLAY_WINDOWS:
                path = folder / f"apertura_flujo_{seconds}s_{self.day.isoformat()}.csv"
                deadline = self.opened_at + timedelta(seconds=seconds)
                with path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(("ticker", "price", "quote_event_time", "quote_received_at"))
                    for quote in sorted(events, key=lambda item: (item.received_at, item.ticker)):
                        if quote.received_at <= deadline:
                            writer.writerow((quote.ticker, quote.price,
                                             quote.event_time.isoformat(),
                                             quote.received_at.isoformat()))
        except OSError as exc:
            with self._lock:
                self.status = f"{self.status}; no se pudo guardar flujo de apertura: {exc}"

    def _start_window_builds(self, now: datetime) -> None:
        with self._lock:
            if self._prior_candles is None:
                return
            for seconds in DISPLAY_WINDOWS:
                deadline = self.opened_at + timedelta(seconds=seconds)
                if (now < deadline or seconds in self._window_building or
                        seconds in self._window_setups):
                    continue
                events = [quote for quote in self._quote_events
                          if quote.received_at <= deadline]
                if not events:
                    continue
                self._window_building.add(seconds)
                candles = self._prior_candles.copy()
                threading.Thread(target=self._build_window,
                                 args=(seconds, events, candles), daemon=True,
                                 name=f"efecto-iman-open-{seconds}s").start()

    def _build_window(self, seconds: int, events: list[OpeningQuote],
                      candles: pd.DataFrame) -> None:
        deadline = self.opened_at + timedelta(seconds=seconds)
        latest: dict[str, OpeningQuote] = {}
        opening_events: dict[str, list[tuple[datetime, float]]] = {}
        for quote in events:
            previous = latest.get(quote.ticker)
            if previous is None or quote.event_time >= previous.event_time:
                latest[quote.ticker] = quote
            if quote.event_time <= deadline:
                opening_events.setdefault(quote.ticker, []).append(
                    (quote.event_time, quote.price))
        try:
            started_at = datetime.now(NEW_YORK)
            setups = opening_setups(candles,
                                    opening_prices={ticker: quote.price
                                                    for ticker, quote in latest.items()},
                                    cutoff_seconds=seconds,
                                    opening_events=opening_events)
            folder = DATA_DIR / "snapshots"
            folder.mkdir(parents=True, exist_ok=True)
            prefix = f"{seconds}s_{self.day.isoformat()}"
            quotes_path = folder / f"apertura_{prefix}.csv"
            with quotes_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("ticker", "price_at_cutoff", "quote_event_time",
                                 "quote_received_at"))
                for quote in sorted(latest.values(), key=lambda item: item.ticker):
                    writer.writerow((quote.ticker, quote.price,
                                     quote.event_time.isoformat(),
                                     quote.received_at.isoformat()))
            setups.to_csv(folder / f"diagnostico_apertura_{prefix}.csv", index=False)
            chosen = setups.loc[setups.candidate_at_open == True].sort_values(
                "score_open", ascending=False)
            chosen.to_csv(folder / f"candidatos_apertura_{prefix}.csv", index=False)
            try:
                record_availability(self.day.isoformat(), f"corte_{seconds}s",
                                    setups, started_at)
            except OSError:
                pass
            with self._lock:
                self._window_quotes[seconds] = latest
                self._window_setups[seconds] = setups
                self._window_ready_at[seconds] = datetime.now(NEW_YORK)
                self.version += 1
                label = {30: "30 segundos", 60: "1 minuto",
                         90: "1 minuto y medio"}[seconds]
                self.status = (f"Corte de {label} listo: {len(latest)}/{len(self.tickers)} "
                               f"cotizaciones; {len(chosen)} candidatos preliminares")
        except Exception as exc:
            with self._lock:
                self.status = f"No se pudo calcular el corte de {seconds} s: {exc}"
        finally:
            with self._lock:
                self._window_building.discard(seconds)

    def _build_if_ready(self) -> None:
        with self._lock:
            if (self._building or self._setups is not None or
                    self._frozen_quotes is None or self._prior_candles is None):
                return
            self._building = True
            quotes = self._frozen_quotes.copy()
            candles = self._prior_candles.copy()
            opening_events: dict[str, list[tuple[datetime, float]]] = {}
            for quote in self._quote_events:
                if quote.received_at <= self.cutoff and quote.event_time <= self.cutoff:
                    opening_events.setdefault(quote.ticker, []).append(
                        (quote.event_time, quote.price))
        try:
            started_at = datetime.now(NEW_YORK)
            setups = opening_setups(candles, opening_prices={ticker: quote.price
                                                              for ticker, quote in quotes.items()},
                                    cutoff_seconds=CAPTURE_SECONDS,
                                    opening_events=opening_events)
            path = DATA_DIR / "snapshots" / f"apertura_10s_{self.day.isoformat()}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("ticker", "price_at_cutoff", "quote_event_time",
                                 "quote_received_at"))
                for quote in sorted(quotes.values(), key=lambda item: item.ticker):
                    writer.writerow((quote.ticker, quote.price,
                                     quote.event_time.isoformat(),
                                     quote.received_at.isoformat()))
            if quotes:
                save_daily_session(self.day.isoformat(), setups, candles)
                try:
                    record_availability(self.day.isoformat(), "corte_10s", setups, started_at)
                except OSError:
                    pass  # El registro auxiliar no debe invalidar el corte ya guardado.
            with self._lock:
                self._setups = setups
                self._ready_at = datetime.now(NEW_YORK)
                self.version += 1
                self.status = (f"Corte 09:30:10 ET: {len(quotes)}/{len(self.tickers)} "
                               "cotizaciones; diagnóstico preliminar listo")
        except Exception as exc:
            with self._lock:
                self.status = f"No se pudo calcular el corte: {exc}"
        finally:
            with self._lock:
                self._building = False

    def snapshot(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str, int]:
        with self._lock:
            return self._setups, self._prior_candles, self.status, self.version

    def snapshot_at(self, seconds: int) -> tuple[pd.DataFrame | None,
                                                pd.DataFrame | None, datetime | None]:
        with self._lock:
            return (self._window_setups.get(seconds), self._prior_candles,
                    self._window_ready_at.get(seconds))

    def progress(self) -> CaptureProgress:
        """Estado real de preparación, recepción y cálculo para la pantalla."""
        with self._lock:
            return CaptureProgress(
                status=self.status,
                prior_count=(int(self._prior_candles.ticker.nunique())
                             if self._prior_candles is not None else 0),
                quote_count=len({quote.ticker for quote in self._quote_events}),
                frozen_count=(len(self._frozen_quotes)
                              if self._frozen_quotes is not None else None),
                stream_connected=self._stream_connected,
                frozen_at=self._frozen_at,
                ready_at=self._ready_at,
                candidate_count=(int(self._setups.candidate_at_open.fillna(False).sum())
                                 if self._setups is not None else None),
                prior_error=self._prior_error,
                stream_error=self._stream_error,
                window_ready_at=self._window_ready_at.copy(),
                window_counts={seconds: len(quotes)
                               for seconds, quotes in self._window_quotes.items()},
                window_candidates={seconds: int(
                    setups.candidate_at_open.fillna(False).sum())
                    for seconds, setups in self._window_setups.items()},
            )
