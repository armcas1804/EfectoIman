"""Velas públicas de 15 y 1 minuto; proveedor Yahoo vía yfinance."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config import PROVIDER_SYMBOLS, TICKERS
from opening_scan import opening_setups, upper_angle_openings


NEW_YORK = ZoneInfo("America/New_York")


def download_bars(period: str = "59d", tickers: tuple[str, ...] = TICKERS,
                  interval: str = "15m"
                  ) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    """Descarga el universo en lotes e informa explícitamente símbolos sin velas."""
    result = {}
    for start in range(0, len(tickers), 4):
        batch = tickers[start:start + 4]
        provider_names = [PROVIDER_SYMBOLS.get(ticker, ticker) for ticker in batch]
        raw = yf.download(provider_names, period=period, interval=interval, group_by="ticker",
                          progress=False, auto_adjust=False, prepost=False, threads=False)
        if raw.empty:
            continue
        for ticker, provider in zip(batch, provider_names):
            try:
                source = raw[provider] if isinstance(raw.columns, pd.MultiIndex) else raw
                table = source[["Open", "High", "Low", "Close"]].dropna().copy()
            except KeyError:
                continue
            if table.empty:
                continue
            times = pd.DatetimeIndex(table.index)
            if times.tz is None:
                times = times.tz_localize("UTC")
            times = times.tz_convert(NEW_YORK)
            table.index = times
            table = table.sort_index()
            table = table.loc[~table.index.duplicated(keep="last")]
            table.columns = ["open", "high", "low", "close"]
            table = table.between_time("09:30", "15:59" if interval == "1m" else "15:45")
            if table.empty:
                continue
            table["day"] = table.index.strftime("%Y-%m-%d")
            table["time"] = table.index.strftime("%H:%M")
            result[ticker] = table
    missing = tuple(ticker for ticker in tickers if ticker not in result)
    if not result:
        raise RuntimeError("La fuente no devolvió velas. Revisa conexión o disponibilidad del proveedor.")
    return result, missing


def available_days(bars: dict[str, pd.DataFrame]) -> list[str]:
    return sorted(set().union(*(set(table.day) for table in bars.values())))


def session_candles(bars: dict[str, pd.DataFrame], day: str) -> pd.DataFrame:
    """Entrega sesión previa y actual con el formato OHLC del escáner."""
    records = []
    for ticker, table in bars.items():
        days = sorted(set(table.day))
        preceding = [value for value in days if value < day]
        if not preceding:
            continue
        prior_day = preceding[-1]
        prior = table.loc[table.day == prior_day].tail(26)
        current = table.loc[table.day == day]
        for index, (_, row) in enumerate(prior.iterrows()):
            records.append({"ticker": ticker, "bar_index": index, "session": "PREVIA",
                            "time": row.time, "open": float(row.open), "high": float(row.high),
                            "low": float(row.low), "close": float(row.close)})
        for index, (_, row) in enumerate(current.iterrows(), start=len(prior)):
            records.append({"ticker": ticker, "bar_index": index, "session": "ACTUAL",
                            "time": row.time, "open": float(row.open), "high": float(row.high),
                            "low": float(row.low), "close": float(row.close)})
    return pd.DataFrame.from_records(records, columns=["ticker", "bar_index", "session", "time",
                                                      "open", "high", "low", "close"])


def first_minute_rows(minute_bars: dict[str, pd.DataFrame], day: str) -> pd.DataFrame:
    rows = []
    for ticker, table in minute_bars.items():
        first = table.loc[(table.day == day) & (table.time == "09:30")]
        if not first.empty:
            bar = first.iloc[0]
            rows.append({"ticker": ticker, "open": float(bar.open), "high": float(bar.high),
                         "low": float(bar.low), "close": float(bar.close)})
    return pd.DataFrame.from_records(rows, columns=["ticker", "open", "high", "low", "close"])


def early_minute_rows(minute_bars: dict[str, pd.DataFrame], day: str,
                      latest_start: str = "09:44") -> pd.DataFrame:
    """Minutos ya cerrados posteriores a la confirmación 09:31."""
    rows = []
    for ticker, table in minute_bars.items():
        selected = table.loc[(table.day == day) & (table.time >= "09:31") &
                             (table.time <= latest_start) & (table.time <= "09:44")]
        for _, bar in selected.iterrows():
            rows.append({"ticker": ticker, "time": str(bar.time), "open": float(bar.open),
                         "high": float(bar.high), "low": float(bar.low), "close": float(bar.close)})
    return pd.DataFrame.from_records(rows, columns=["ticker", "time", "open", "high", "low", "close"])


def five_minute_rows(five_minute_bars: dict[str, pd.DataFrame], day: str) -> pd.DataFrame:
    """Primera y segunda vela de 5m, cerradas a las 09:35 y 09:40 ET."""
    rows = []
    for ticker, table in five_minute_bars.items():
        selected = table.loc[(table.day == day) & table.time.isin(["09:30", "09:35"])]
        for bar in selected.itertuples():
            rows.append({"ticker": ticker, "time": str(bar.time), "open": float(bar.open),
                         "high": float(bar.high), "low": float(bar.low), "close": float(bar.close)})
    return pd.DataFrame.from_records(rows, columns=["ticker", "time", "open", "high", "low", "close"])


def scan_upper_angle_day(bars: dict[str, pd.DataFrame], day: str,
                         minute_bars: dict[str, pd.DataFrame] | None = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Escaneo histórico o en vivo con minutos ya cerrados, sin guardar en SQLite."""
    candles = session_candles(bars, day)
    minute_rows = []
    now = datetime.now(NEW_YORK)
    live = day == now.date().isoformat()
    latest_complete_start = (now - timedelta(minutes=1)).strftime("%H:%M")
    for ticker, table in (minute_bars or {}).items():
        selected = table.loc[(table.day == day) & (table.time >= "09:30") &
                             (table.time <= "09:43")]
        if live:
            selected = selected.loc[selected.time <= latest_complete_start]
        minute_rows.extend({"ticker": ticker, "time": str(bar.time),
                            "open": float(bar.open), "close": float(bar.close)}
                           for bar in selected.itertuples())
    minutes = pd.DataFrame.from_records(minute_rows, columns=["ticker", "time", "open", "close"])
    return upper_angle_openings(candles, minutes, day,
                                include_final_15m=not live or now.strftime("%H:%M") >= "09:45")


def scan_real_day(bars: dict[str, pd.DataFrame], day: str,
                  minute_bars: dict[str, pd.DataFrame] | None = None,
                  five_minute_bars: dict[str, pd.DataFrame] | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    candles = session_candles(bars, day)
    if candles.empty:
        return opening_setups(candles), candles
    opening_prices = {ticker: float(rows.iloc[0].open)
                      for ticker, rows in candles.loc[candles.session == "ACTUAL"].groupby("ticker")}
    raw_minutes = first_minute_rows(minute_bars or {}, day)
    for row in raw_minutes.itertuples():
        opening_prices[row.ticker] = float(row.open)
    minutes = raw_minutes
    early = early_minute_rows(minute_bars or {}, day)
    live = day == datetime.now(NEW_YORK).date().isoformat()
    five = None if live else five_minute_rows(five_minute_bars or {}, day)
    if live:
        now = datetime.now(NEW_YORK)
        first_complete = now >= datetime.combine(now.date(), datetime.min.time(), NEW_YORK).replace(
            hour=9, minute=31)
        if not first_complete:
            minutes = raw_minutes.iloc[0:0]
        latest_complete_start = (now - timedelta(minutes=1)).strftime("%H:%M")
        early = early_minute_rows(minute_bars or {}, day, latest_complete_start)
        # No usar la vela que el proveedor aún pueda estar formando.
        cutoff = now - timedelta(minutes=15)
        candles = candles.loc[(candles.session == "PREVIA") |
                              (candles.time <= cutoff.strftime("%H:%M"))].copy()
    return opening_setups(candles, minutes, opening_prices=opening_prices,
                          early_minutes=early, five_minutes=five), candles
