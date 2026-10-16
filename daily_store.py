"""Sesiones diagnósticas y velas locales en SQLite."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd

from config import DATABASE_PATH, UNIVERSE_ID
from opening_scan import SETUP_COLUMNS


CANDLE_COLUMNS = ("ticker", "bar_index", "session", "time", "open", "high", "low", "close")
SOURCE = "real"


def initialize_daily_store() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE_PATH)) as db, db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS daily_sessions (
                session_date TEXT NOT NULL, source TEXT NOT NULL, updated_at TEXT NOT NULL,
                universe_id TEXT NOT NULL, PRIMARY KEY(session_date, source));
            CREATE TABLE IF NOT EXISTS daily_candles (
                session_date TEXT NOT NULL, source TEXT NOT NULL, ticker TEXT NOT NULL,
                bar_index INTEGER NOT NULL, session TEXT NOT NULL, time TEXT NOT NULL,
                open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
                PRIMARY KEY(session_date, source, ticker, bar_index));
            CREATE TABLE IF NOT EXISTS daily_diagnostics (
                session_date TEXT NOT NULL, source TEXT NOT NULL, ticker TEXT NOT NULL,
                score REAL, gap_pct REAL, candidate INTEGER NOT NULL,
                confirmed INTEGER NOT NULL, payload_json TEXT NOT NULL,
                PRIMARY KEY(session_date, source, ticker));
        """)
        columns = {row[1] for row in db.execute("PRAGMA table_info(daily_sessions)")}
        if "universe_id" not in columns:
            db.execute("ALTER TABLE daily_sessions ADD COLUMN universe_id TEXT NOT NULL DEFAULT 'legacy'")
        if db.execute("PRAGMA user_version").fetchone()[0] < 2:
            for rowid, payload in list(db.execute(
                    "SELECT rowid, payload_json FROM daily_diagnostics")):
                row = json.loads(payload)
                if row.get("distance_to_band_pct") is not None and row["distance_to_band_pct"] <= 0:
                    row.update(touched_target_band=False, touch_time=None, bars_until_touch=None,
                               minutes_until_touch=None, target_band_price_at_touch=None,
                               touch_status="NOT_APPLICABLE_INSIDE_BAND")
                opening, target = row.get("opening_price"), row.get("target_band_price_at_touch")
                if row.get("touched_target_band") and opening and target:
                    sign = 1 if row.get("gap_direction") == "UP" else -1
                    reversion = sign * (opening - target) / opening * 100
                    row["reversion_to_touch_band_pct"] = reversion
                    row["band_at_or_beyond_open_at_touch"] = bool(reversion <= 0)
                else:
                    row["reversion_to_touch_band_pct"] = None
                    row["band_at_or_beyond_open_at_touch"] = None
                db.execute("UPDATE daily_diagnostics SET payload_json=? WHERE rowid=?",
                           (json.dumps(row, ensure_ascii=False, allow_nan=False), rowid))
            db.execute("PRAGMA user_version=2")


def save_daily_session(day: str, setups: pd.DataFrame, candles: pd.DataFrame,
                       replace_existing: bool = False) -> None:
    """Actualiza métricas por fase y conserva el universo observado de la sesión."""
    initialize_daily_store()
    updated = datetime.now().astimezone().isoformat(timespec="seconds")
    payloads = json.loads(setups.to_json(orient="records")) if not setups.empty else []
    with closing(sqlite3.connect(DATABASE_PATH)) as db, db:
        previous = db.execute(
            "SELECT universe_id FROM daily_sessions WHERE session_date=? AND source=?",
            (day, SOURCE)).fetchone()
        if previous is not None and (previous[0] != UNIVERSE_ID or replace_existing):
            db.execute("DELETE FROM daily_candles WHERE session_date=? AND source=?", (day, SOURCE))
        db.execute("DELETE FROM daily_diagnostics WHERE session_date=? AND source=?", (day, SOURCE))
        db.execute(
            "INSERT INTO daily_sessions (session_date, source, updated_at, universe_id) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(session_date, source) DO UPDATE SET "
            "updated_at=excluded.updated_at, universe_id=excluded.universe_id",
            (day, SOURCE, updated, UNIVERSE_ID))
        db.executemany(
            "INSERT INTO daily_diagnostics VALUES (?,?,?,?,?,?,?,?)",
            [(day, SOURCE, row["ticker"], row.get("score"), row.get("gap_pct"),
              int(bool(row.get("candidate"))), int(row.get("confirmed", -1)),
              json.dumps(row, ensure_ascii=False, allow_nan=False))
             for row in payloads])
        db.executemany(
            "INSERT INTO daily_candles VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(session_date, source, ticker, bar_index) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close",
            [(day, SOURCE, *(row[column] for column in CANDLE_COLUMNS))
             for row in candles.to_dict("records")])


def session_dates() -> list[str]:
    initialize_daily_store()
    with closing(sqlite3.connect(DATABASE_PATH)) as db:
        return [row[0] for row in db.execute(
            "SELECT session_date FROM daily_sessions WHERE source=? AND universe_id=? "
            "ORDER BY session_date DESC", (SOURCE, UNIVERSE_ID))]


def load_daily_session(day: str) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    initialize_daily_store()
    with closing(sqlite3.connect(DATABASE_PATH)) as db:
        updated = db.execute(
            "SELECT updated_at FROM daily_sessions "
            "WHERE session_date=? AND source=? AND universe_id=?",
            (day, SOURCE, UNIVERSE_ID)).fetchone()
        if updated is None:
            return pd.DataFrame(columns=SETUP_COLUMNS), pd.DataFrame(columns=CANDLE_COLUMNS), None
        payloads = [json.loads(row[0]) for row in db.execute(
            "SELECT payload_json FROM daily_diagnostics WHERE session_date=? AND source=? "
            "ORDER BY score DESC, ticker", (day, SOURCE))]
        candles = pd.read_sql_query(
            "SELECT " + ",".join(CANDLE_COLUMNS) + " FROM daily_candles "
            "WHERE session_date=? AND source=? ORDER BY ticker, bar_index",
            db, params=(day, SOURCE))
    return pd.DataFrame.from_records(payloads, columns=SETUP_COLUMNS), candles, updated[0]


def export_history(days: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = session_dates()[:days]
    if not dates:
        return pd.DataFrame(columns=["session_date", "source"] + SETUP_COLUMNS), pd.DataFrame()
    placeholders = ",".join("?" for _ in dates)
    with closing(sqlite3.connect(DATABASE_PATH)) as db:
        setups = []
        for day, source, payload in db.execute(
            f"SELECT session_date, source, payload_json FROM daily_diagnostics "
            f"WHERE source=? AND session_date IN ({placeholders}) "
            "ORDER BY session_date DESC, score DESC, ticker", (SOURCE, *dates)):
            setups.append({"session_date": day, "source": source, **json.loads(payload)})
        candles = pd.read_sql_query(
            f"SELECT * FROM daily_candles WHERE source=? AND session_date IN ({placeholders}) "
            "ORDER BY session_date DESC, ticker, bar_index", db, params=(SOURCE, *dates))
    return pd.DataFrame.from_records(setups, columns=["session_date", "source"] + SETUP_COLUMNS), candles
