"""BB(20,2) de 15m provisional en los primeros 10, 20 y 30 segundos.

La reacción frente a la banda previa requiere un precio oportuno. Una pendiente
dentro de la apertura requiere dos eventos posteriores a las 09:30 con horas de
mercado distintas y recibidos antes del corte correspondiente; de lo contrario
queda vacía. No se utilizan velas posteriores ni se modifica el score.
"""

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import DATA_DIR, TICKERS
from daily_store import load_daily_session
from indicators import bollinger_bands
from opening_scan import _provisional_bands, normalized_band_angle_deg


NEW_YORK = ZoneInfo("America/New_York")
WINDOWS = (10, 20, 30)
COLUMNS = (
    "date", "ticker", "window_seconds", "gap_direction_at_10s",
    "gap_direction_at_window",
    "quote_count", "price_at_window", "last_event_time", "last_received_at",
    "prior_upper_band", "prior_lower_band", "upper_band_provisional",
    "lower_band_provisional", "target_band_previous", "target_band_provisional",
    "target_change_from_prior_pct", "directional_reaction_pct",
    "first_event_time", "event_span_seconds", "target_slope_pct_per_min",
    "target_angle_deg_5m_equiv", "measurement_status",
)


def _events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "price", "quote_event_time",
                                     "quote_received_at", "event", "received"])
    events = pd.read_csv(path)
    expected = {"ticker", "price", "quote_event_time", "quote_received_at"}
    if not expected.issubset(events.columns):
        raise ValueError(f"Flujo incompleto: {path}")
    events["event"] = pd.to_datetime(events.quote_event_time, utc=True)
    events["received"] = pd.to_datetime(events.quote_received_at, utc=True)
    return events


def build_reaction(day: str, tickers: tuple[str, ...] = TICKERS) -> pd.DataFrame:
    snapshot = DATA_DIR / "snapshots"
    capture = pd.read_csv(snapshot / f"apertura_10s_{day}.csv").set_index("ticker")
    stream_path = snapshot / f"apertura_flujo_30s_{day}.csv"
    events = _events(stream_path)
    _, candles, _ = load_daily_session(day)
    prior = candles.loc[candles.session == "PREVIA"]
    if prior.empty:
        raise RuntimeError("No hay velas previas guardadas para la fecha")
    opened = datetime.combine(date.fromisoformat(day), time(9, 30), NEW_YORK)
    rows = []
    for ticker in tickers:
        bars = prior.loc[prior.ticker == ticker].sort_values("bar_index")
        if len(bars) < 20:
            continue
        bands = bollinger_bands(bars).iloc[-1]
        upper, lower = float(bands.bb_upper), float(bands.bb_lower)
        observed = ticker in capture.index
        cutoff_price = float(capture.loc[ticker, "price_at_cutoff"]) if observed else np.nan
        cutoff_sign = (1 if cutoff_price >= float(bands.close) else -1) if observed else 0
        direction = ("UP" if cutoff_sign == 1 else "DOWN") if cutoff_sign else ""
        ticker_events = events.loc[events.ticker == ticker]
        for window in (WINDOWS if stream_path.exists() else (10,)):
            deadline = opened + timedelta(seconds=window)
            available = ticker_events.loc[
                (ticker_events.received <= deadline) & (ticker_events.event <= deadline)]
            # The original ten-second file is the authoritative frozen price.
            if window == 10:
                last_price = cutoff_price
                last_event = str(capture.loc[ticker, "quote_event_time"]) if observed else ""
                last_received = str(capture.loc[ticker, "quote_received_at"]) if observed else ""
            elif not available.empty:
                last = available.sort_values(["event", "received"]).iloc[-1]
                last_price = float(last.price)
                last_event = str(last.quote_event_time)
                last_received = str(last.quote_received_at)
            else:
                last_price, last_event, last_received = np.nan, "", ""
            sign = (cutoff_sign if observed else
                    (1 if last_price >= float(bands.close) else -1)
                    if np.isfinite(last_price) else 0)
            row = dict.fromkeys(COLUMNS, np.nan)
            row.update(date=day, ticker=ticker, window_seconds=window,
                       gap_direction_at_10s=direction,
                       gap_direction_at_window=("UP" if sign == 1 else "DOWN") if sign else "",
                       quote_count=(len(available) if stream_path.exists()
                                    else int(observed)),
                       price_at_window=last_price, last_event_time=last_event,
                       last_received_at=last_received, prior_upper_band=upper,
                       prior_lower_band=lower, first_event_time="",
                       measurement_status="NO_QUOTE" if not np.isfinite(last_price)
                       else "ONLY_CUTOFF_QUOTE")
            if np.isfinite(last_price):
                provisional_upper, provisional_lower = _provisional_bands(bars, last_price)
                row.update(upper_band_provisional=provisional_upper,
                           lower_band_provisional=provisional_lower)
                if sign:
                    before = upper if sign == 1 else lower
                    after = provisional_upper if sign == 1 else provisional_lower
                    change = (after / before - 1) * 100
                    row.update(target_band_previous=before,
                               target_band_provisional=after,
                               target_change_from_prior_pct=change,
                               directional_reaction_pct=sign * change)
            if len(available) >= 2 and sign:
                ordered = available.sort_values(["event", "received"])
                first, last = ordered.iloc[0], ordered.iloc[-1]
                elapsed = (last.event - first.event).total_seconds()
                row["first_event_time"] = str(first.quote_event_time)
                row["event_span_seconds"] = elapsed
                if elapsed > 0:
                    first_upper, first_lower = _provisional_bands(bars, float(first.price))
                    last_upper, last_lower = _provisional_bands(bars, float(last.price))
                    first_band = first_upper if sign == 1 else first_lower
                    last_band = last_upper if sign == 1 else last_lower
                    minutes = elapsed / 60
                    row["target_slope_pct_per_min"] = (
                        sign * (last_band / first_band - 1) * 100 / minutes)
                    row["target_angle_deg_5m_equiv"] = (
                        sign * normalized_band_angle_deg(first_band, last_band, minutes))
                    row["measurement_status"] = "POST_OPEN_SLOPE_OBSERVED"
                else:
                    row["measurement_status"] = "NO_DISTINCT_EVENT_TIMES"
            elif not available.empty:
                row["measurement_status"] = "ONE_EVENT_ONLY"
            rows.append(row)
    return pd.DataFrame.from_records(rows, columns=COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure provisional opening Bollinger bands")
    parser.add_argument("day", nargs="?", default=datetime.now(NEW_YORK).date().isoformat())
    day = parser.parse_args().day
    result = build_reaction(day)
    path = DATA_DIR / "snapshots" / f"reaccion_bollinger_apertura_{day}.csv"
    result.to_csv(path, index=False, float_format="%.8f")
    chosen_path = DATA_DIR / "snapshots" / f"candidatos_apertura_10s_{day}.csv"
    chosen = set(pd.read_csv(chosen_path).ticker) if chosen_path.exists() else set()
    current = result.loc[(result.window_seconds == 10) & result.ticker.isin(chosen)]
    print(current[["ticker", "gap_direction_at_10s",
                   "target_change_from_prior_pct", "directional_reaction_pct",
                   "target_slope_pct_per_min", "measurement_status"]].to_string(index=False))
    print(f"Guardado: {path}")


if __name__ == "__main__":
    main()
