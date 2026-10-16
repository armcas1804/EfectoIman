"""Completa cortes históricos de 10, 20 y 30 s con operaciones SIP de Alpaca.

Requiere APCA_API_KEY_ID y APCA_API_SECRET_KEY en el entorno. Los tiempos
históricos son de mercado; no prueban cuándo habrían llegado al equipo.
"""

import argparse
import os
import time
from datetime import date, datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from backfill_10s_alpaca import ENDPOINT, TRADE_SYMBOLS, completed_session_dates
from config import DATA_DIR


NY = ZoneInfo("America/New_York")
OUTPUT_COLUMNS = ("date", "ticker", "source", "window_seconds", "first_trade_at_et",
                  "last_trade_at_et", "price_first", "price_at_cutoff", "trade_count")
TRAJECTORY_COLUMNS = ("date", "ticker", "second_after_open", "last_trade_at_et",
                      "price", "trade_count")


def summarize_windows(day: str, trades_by_symbol: dict) -> pd.DataFrame:
    opened = datetime.combine(date.fromisoformat(day), clock_time(9, 30), NY)
    records = []
    for ticker, trades in trades_by_symbol.items():
        if ticker not in TRADE_SYMBOLS:
            continue
        valid = []
        for trade in trades:
            try:
                stamp = datetime.fromisoformat(trade["t"].replace("Z", "+00:00"))
                price = float(trade["p"])
            except (KeyError, TypeError, ValueError):
                continue
            elapsed = (stamp.astimezone(NY) - opened).total_seconds()
            if 0 <= elapsed <= 30 and price > 0:
                valid.append((stamp, price, elapsed))
        valid.sort(key=lambda item: item[0])
        for seconds in (10, 20, 30):
            eligible = [trade for trade in valid if trade[2] <= seconds]
            if not eligible:
                continue
            first, last = eligible[0], eligible[-1]
            records.append({
                "date": day, "ticker": ticker,
                "source": "alpaca_sip_historical_trades",
                "window_seconds": seconds,
                "first_trade_at_et": first[0].astimezone(NY).isoformat(),
                "last_trade_at_et": last[0].astimezone(NY).isoformat(),
                "price_first": first[1], "price_at_cutoff": last[1],
                "trade_count": len(eligible),
            })
    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS).sort_values(
        ["ticker", "window_seconds"]).reset_index(drop=True)


def summarize_trajectory(day: str, trades_by_symbol: dict) -> pd.DataFrame:
    """Un precio observado por segundo: último negocio de ese segundo."""
    opened = datetime.combine(date.fromisoformat(day), clock_time(9, 30), NY)
    buckets = {}
    for ticker, trades in trades_by_symbol.items():
        if ticker not in TRADE_SYMBOLS:
            continue
        for trade in trades:
            try:
                stamp = datetime.fromisoformat(trade["t"].replace("Z", "+00:00"))
                price = float(trade["p"])
            except (KeyError, TypeError, ValueError):
                continue
            elapsed = (stamp.astimezone(NY) - opened).total_seconds()
            if 0 <= elapsed <= 30 and price > 0:
                key = (ticker, int(elapsed))
                bucket = buckets.setdefault(key, [])
                bucket.append((stamp, price))
    records = []
    for (ticker, second), trades in sorted(buckets.items()):
        last = max(trades, key=lambda item: item[0])
        records.append((day, ticker, second, last[0].astimezone(NY).isoformat(),
                        last[1], len(trades)))
    return pd.DataFrame.from_records(records, columns=TRAJECTORY_COLUMNS)


def fetch_day(session: requests.Session, day: str,
              symbols: tuple[str, ...] = TRADE_SYMBOLS) -> tuple[pd.DataFrame, pd.DataFrame]:
    opened = datetime.combine(date.fromisoformat(day), clock_time(9, 30), NY)
    params = {"symbols": ",".join(symbols),
              "start": opened.astimezone(timezone.utc).isoformat(),
              "end": (opened + timedelta(seconds=30)).astimezone(timezone.utc).isoformat(),
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
            return summarize_windows(day, gathered), summarize_trajectory(day, gathered)
        if token in seen_tokens:
            raise RuntimeError("Paginación repetida; se descarta este día incompleto")
        seen_tokens.add(token)
        params["page_token"] = token
    raise RuntimeError("Límite de páginas alcanzado; se descarta este día incompleto")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=50)
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Faltan APCA_API_KEY_ID y APCA_API_SECRET_KEY en el entorno.")
    days = ([args.date.isoformat()] if args.date else completed_session_dates(args.days))
    if not days:
        raise SystemExit("No hay sesiones históricas locales para consultar.")
    folder = DATA_DIR / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers.update({"APCA-API-KEY-ID": key,
                                "APCA-API-SECRET-KEY": secret})
        for day in days:
            output = folder / f"apertura_historica_30s_{day}.csv"
            if output.exists() and not args.refresh:
                print(f"{day}: ya guardado", flush=True)
                continue
            opened = datetime.combine(date.fromisoformat(day), clock_time(9, 30), NY)
            if datetime.now(timezone.utc) - opened.astimezone(timezone.utc) < timedelta(minutes=15):
                print(f"{day}: corte SIP demasiado reciente", flush=True)
                continue
            try:
                frame, trajectory = fetch_day(session, day)
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                print(f"{day}: descarga incompleta ({type(exc).__name__}: {exc})", flush=True)
                continue
            trajectory_output = folder / f"apertura_historica_trayectoria_30s_{day}.csv"
            trajectory_temporary = trajectory_output.with_suffix(".csv.tmp")
            trajectory.to_csv(trajectory_temporary, index=False)
            trajectory_temporary.replace(trajectory_output)
            temporary = output.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(output)
            counts = frame.groupby("window_seconds").ticker.nunique().to_dict()
            print(f"{day}: 10 s={counts.get(10, 0)}, 20 s={counts.get(20, 0)}, "
                  f"30 s={counts.get(30, 0)} símbolos", flush=True)


if __name__ == "__main__":
    main()
