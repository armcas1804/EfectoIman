"""Cobertura verificable de cortes reales de apertura de diez segundos."""

import sqlite3
from contextlib import closing
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config import DATABASE_PATH, DATA_DIR, HISTORY_SESSION_LIMIT


NEW_YORK = ZoneInfo("America/New_York")
REQUIRED_COLUMNS = {"ticker", "price_at_cutoff", "quote_event_time", "quote_received_at"}


def ten_second_coverage(database: Path = DATABASE_PATH,
                        snapshot_dir: Path = DATA_DIR / "snapshots",
                        limit: int = HISTORY_SESSION_LIMIT) -> pd.DataFrame:
    """Cuenta sólo precios realmente recibidos en el corte, nunca velas reconstruidas."""
    with closing(sqlite3.connect(database)) as db:
        sessions = db.execute(
            "SELECT s.session_date, COUNT(d.ticker) FROM daily_sessions s "
            "LEFT JOIN daily_diagnostics d ON d.session_date=s.session_date "
            "AND d.source=s.source WHERE s.source='real' "
            "GROUP BY s.session_date ORDER BY s.session_date DESC LIMIT ?", (limit,)
        ).fetchall()
    rows = []
    for day, historical_tickers in sessions:
        path = snapshot_dir / f"apertura_10s_{day}.csv"
        historical_path = snapshot_dir / f"apertura_historica_10s_{day}.csv"
        valid_quotes = 0
        historical_quotes = 0
        status = "SIN_CAPTURA"
        historical_status = "SIN_HISTORICO_SUBMINUTO"
        if path.is_file():
            try:
                quotes = pd.read_csv(path)
                if not REQUIRED_COLUMNS.issubset(quotes.columns):
                    raise ValueError("faltan columnas del corte")
                received = pd.to_datetime(quotes.quote_received_at, utc=True,
                                          errors="coerce").dt.tz_convert(NEW_YORK)
                event = pd.to_datetime(quotes.quote_event_time, utc=True,
                                       errors="coerce").dt.tz_convert(NEW_YORK)
                prices = pd.to_numeric(quotes.price_at_cutoff, errors="coerce")
                opened = datetime.combine(datetime.fromisoformat(day).date(),
                                          time(9, 30), NEW_YORK)
                elapsed = (received - opened).dt.total_seconds()
                valid = (quotes.ticker.notna() & prices.gt(0) &
                         elapsed.between(0, 10, inclusive="both") &
                         event.notna())
                valid_quotes = int(quotes.loc[valid, "ticker"].nunique())
                status = "CAPTURA_10S" if valid_quotes else "SIN_COTIZACIONES_VALIDAS"
            except (OSError, ValueError, pd.errors.ParserError, KeyError, AttributeError):
                status = "ARCHIVO_INVALIDO"
        if historical_path.is_file():
            try:
                trades = pd.read_csv(historical_path)
                required = {"ticker", "price_at_10s", "last_trade_at_et", "source"}
                if not required.issubset(trades.columns):
                    raise ValueError("faltan columnas del histórico")
                stamps = pd.to_datetime(trades.last_trade_at_et, utc=True,
                                        errors="coerce").dt.tz_convert(NEW_YORK)
                opened = datetime.combine(datetime.fromisoformat(day).date(),
                                          time(9, 30), NEW_YORK)
                elapsed = (stamps - opened).dt.total_seconds()
                prices = pd.to_numeric(trades.price_at_10s, errors="coerce")
                valid = (trades.ticker.notna() & prices.gt(0) &
                         trades.source.eq("alpaca_sip_historical_trades") &
                         elapsed.between(0, 10, inclusive="both"))
                historical_quotes = int(trades.loc[valid, "ticker"].nunique())
                historical_status = ("HISTORICO_SIP_10S" if historical_quotes
                                     else "SIN_OPERACIONES_VALIDAS")
            except (OSError, ValueError, pd.errors.ParserError, KeyError, AttributeError):
                historical_status = "ARCHIVO_HISTORICO_INVALIDO"
        rows.append({"date": day, "historical_tickers": historical_tickers,
                     "ten_second_quotes": valid_quotes, "status": status,
                     "snapshot_file": path.name if path.is_file() else "",
                     "historical_10s_quotes": historical_quotes,
                     "historical_status": historical_status,
                     "historical_file": historical_path.name if historical_path.is_file() else ""})
    return pd.DataFrame.from_records(rows, columns=[
        "date", "historical_tickers", "ten_second_quotes", "status", "snapshot_file",
        "historical_10s_quotes", "historical_status", "historical_file"])
