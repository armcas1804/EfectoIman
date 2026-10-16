"""Reconstruye aperturas históricas de 10 s con operaciones SIP de Alpaca.

Requiere APCA_API_KEY_ID y APCA_API_SECRET_KEY en el entorno. Los timestamps
históricos son de mercado; no prueban cuándo habría recibido datos este equipo.
"""

import argparse
import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import date, datetime, time as day_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import DATABASE_PATH, DATA_DIR, TICKERS


NEW_YORK = ZoneInfo("America/New_York")
ENDPOINT = "https://data.alpaca.markets/v2/stocks/trades"
OUTPUT_COLUMNS = (
    "date", "ticker", "source", "first_trade_at_et", "last_trade_at_et",
    "price_first_10s", "price_at_10s", "trade_count", "last_trade_conditions",
)
# SPX es un índice, no un título con operaciones bursátiles propias.
TRADE_SYMBOLS = tuple(ticker for ticker in TICKERS if ticker != "SPX")


def completed_session_dates(limit: int = 50,
                            database: Path = DATABASE_PATH) -> list[str]:
    today = datetime.now(NEW_YORK).date().isoformat()
    with closing(sqlite3.connect(database)) as db:
        rows = db.execute(
            "SELECT DISTINCT session_date FROM daily_sessions "
            "WHERE source='real' AND session_date < ? "
            "ORDER BY session_date DESC LIMIT ?", (today, limit)).fetchall()
    return [row[0] for row in reversed(rows)]


def market_window(day: str) -> tuple[str, str]:
    opened = datetime.combine(date.fromisoformat(day), day_time(9, 30), NEW_YORK)
    return (opened.astimezone(timezone.utc).isoformat(),
            (opened + timedelta(seconds=10)).astimezone(timezone.utc).isoformat())


def summarize_trades(day: str, trades_by_symbol: dict) -> pd.DataFrame:
    """Última operación hasta 09:30:10 ET; conserva trazabilidad del primer print."""
    opened = datetime.combine(date.fromisoformat(day), day_time(9, 30), NEW_YORK)
    cutoff = opened + timedelta(seconds=10)
    rows = []
    for ticker, trades in trades_by_symbol.items():
        if ticker not in TRADE_SYMBOLS:
            continue
        valid = []
        for trade in trades:
            try:
                stamp = datetime.fromisoformat(trade["t"].replace("Z", "+00:00"))
                local = stamp.astimezone(NEW_YORK)
                price = float(trade["p"])
            except (KeyError, ValueError, TypeError):
                continue
            if opened <= local <= cutoff and price > 0:
                valid.append((stamp, price, trade.get("c", [])))
        if not valid:
            continue
        valid.sort(key=lambda item: item[0])
        first, last = valid[0], valid[-1]
        rows.append({
            "date": day, "ticker": ticker, "source": "alpaca_sip_historical_trades",
            "first_trade_at_et": first[0].astimezone(NEW_YORK).isoformat(),
            "last_trade_at_et": last[0].astimezone(NEW_YORK).isoformat(),
            "price_first_10s": first[1], "price_at_10s": last[1],
            "trade_count": len(valid),
            "last_trade_conditions": json.dumps(last[2], ensure_ascii=False),
        })
    return pd.DataFrame.from_records(rows, columns=OUTPUT_COLUMNS).sort_values(
        "ticker").reset_index(drop=True)


def fetch_day(session: requests.Session, day: str,
              symbols: tuple[str, ...] = TRADE_SYMBOLS) -> pd.DataFrame:
    start, end = market_window(day)
    params = {"symbols": ",".join(symbols), "start": start, "end": end,
              "feed": "sip", "sort": "asc", "limit": 10000}
    gathered: dict[str, list] = {}
    seen_tokens = set()
    for _ in range(1000):
        for attempt in range(4):
            response = session.get(ENDPOINT, params=params, timeout=30)
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            if attempt == 3:
                response.raise_for_status()
            time.sleep(min(30, 2 ** attempt + 1))
        response.raise_for_status()
        payload = response.json()
        batches = payload.get("trades") or {}
        if not isinstance(batches, dict):
            raise ValueError("Respuesta inesperada: 'trades' no es un mapa de símbolos")
        for ticker, trades in batches.items():
            gathered.setdefault(ticker, []).extend(trades)
        token = payload.get("next_page_token")
        if not token:
            return summarize_trades(day, gathered)
        if token in seen_tokens:
            raise RuntimeError("Paginación repetida; se descarta este día incompleto")
        seen_tokens.add(token)
        params["page_token"] = token
    raise RuntimeError("Límite de páginas alcanzado; se descarta este día incompleto")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=50)
    parser.add_argument("--date", type=date.fromisoformat,
                        help="consultar una fecha cuyo corte ya tenga 15 minutos")
    parser.add_argument("--refresh", action="store_true",
                        help="volver a consultar días que ya tienen archivo completo")
    args = parser.parse_args()
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Faltan APCA_API_KEY_ID y APCA_API_SECRET_KEY en el entorno. "
                         "No introduzcas las claves en el código ni en el chat.")
    days = ([args.date.isoformat()] if args.date else completed_session_dates(args.days))
    if not days:
        raise SystemExit("No hay sesiones históricas locales para consultar.")
    if args.date:
        cutoff = datetime.fromisoformat(market_window(days[0])[1])
        if datetime.now(timezone.utc) - cutoff < timedelta(minutes=15):
            raise SystemExit("El corte SIP debe tener al menos 15 minutos de antigüedad.")
    folder = DATA_DIR / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers.update({"APCA-API-KEY-ID": key,
                                "APCA-API-SECRET-KEY": secret})
        for day in days:
            output = folder / f"apertura_historica_10s_{day}.csv"
            if output.exists() and not args.refresh:
                print(f"{day}: ya guardado", flush=True)
                continue
            try:
                frame = fetch_day(session, day)
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                print(f"{day}: descarga incompleta ({type(exc).__name__}: {exc})", flush=True)
                continue
            temporary = output.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(output)
            print(f"{day}: {len(frame)}/{len(TRADE_SYMBOLS)} símbolos con operación "
                  "en los primeros 10 s", flush=True)


if __name__ == "__main__":
    main()
