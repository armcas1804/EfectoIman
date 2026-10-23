"""Revisión humana de aperturas, separada del score y del scanner."""

import sqlite3
from contextlib import closing
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import DATA_DIR, DATABASE_PATH
from indicators import bollinger_bands
from opening_scan import (_linear_slope, _provisional_bands,
                          normalized_band_angle_deg)


NY = ZoneInfo("America/New_York")
LABELS = ("Excelente", "Buena", "Mala", "Dudosa")
EARLY_COLUMNS = ("session_date", "ticker", "window_seconds", "source", "price",
                 "first_price", "first_time", "last_time", "event_count")


def initialize_reviews(database: Path = DATABASE_PATH) -> None:
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS manual_reviews (
            session_date TEXT NOT NULL, ticker TEXT NOT NULL,
            label TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY(session_date, ticker))""")


def load_reviews(database: Path = DATABASE_PATH) -> pd.DataFrame:
    initialize_reviews(database)
    with closing(sqlite3.connect(database)) as db:
        return pd.read_sql_query("SELECT session_date, ticker, label, updated_at "
                                 "FROM manual_reviews ORDER BY session_date DESC, ticker", db)


def save_review(day: str, ticker: str, label: str | None,
                database: Path = DATABASE_PATH) -> None:
    if label is not None and label not in LABELS:
        raise ValueError("Etiqueta no válida")
    initialize_reviews(database)
    with closing(sqlite3.connect(database)) as db, db:
        if label is None:
            db.execute("DELETE FROM manual_reviews WHERE session_date=? AND ticker=?",
                       (day, ticker))
        else:
            db.execute("INSERT INTO manual_reviews VALUES (?,?,?,?) "
                       "ON CONFLICT(session_date,ticker) DO UPDATE SET "
                       "label=excluded.label, updated_at=excluded.updated_at",
                       (day, ticker, label, datetime.now(NY).isoformat(timespec="seconds")))


def _live_windows(day: str, path: Path, windows: tuple[int, ...]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=EARLY_COLUMNS)
    quotes = pd.read_csv(path)
    needed = {"ticker", "price", "quote_event_time", "quote_received_at"}
    if not needed.issubset(quotes):
        return pd.DataFrame(columns=EARLY_COLUMNS)
    quotes["received"] = pd.to_datetime(quotes.quote_received_at, utc=True,
                                         errors="coerce").dt.tz_convert(NY)
    quotes["event"] = pd.to_datetime(quotes.quote_event_time, utc=True,
                                      errors="coerce").dt.tz_convert(NY)
    quotes["price"] = pd.to_numeric(quotes.price, errors="coerce")
    opened = pd.Timestamp(datetime.combine(datetime.fromisoformat(day).date(), time(9, 30), NY))
    quotes["elapsed"] = (quotes.received - opened).dt.total_seconds()
    quotes = quotes.loc[quotes.elapsed.between(0, max(windows)) & quotes.price.gt(0) &
                        quotes.event.notna()].sort_values(["ticker", "received"])
    records = []
    for ticker, rows in quotes.groupby("ticker"):
        for seconds in windows:
            eligible = rows.loc[rows.elapsed <= seconds]
            if eligible.empty:
                continue
            first, last = eligible.iloc[0], eligible.iloc[-1]
            records.append((day, ticker, seconds, "captura_local", float(last.price),
                            float(first.price), first.event.isoformat(),
                            last.event.isoformat(), len(eligible)))
    return pd.DataFrame.from_records(records, columns=EARLY_COLUMNS)


def _historical_ten_seconds(day: str, path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=EARLY_COLUMNS)
    trades = pd.read_csv(path)
    needed = {"ticker", "price_at_10s", "price_first_10s", "first_trade_at_et",
              "last_trade_at_et", "trade_count", "source"}
    if not needed.issubset(trades):
        return pd.DataFrame(columns=EARLY_COLUMNS)
    trades = trades.loc[trades.source.eq("alpaca_sip_historical_trades")].copy()
    if trades.empty:
        return pd.DataFrame(columns=EARLY_COLUMNS)
    result = pd.DataFrame({
        "session_date": day, "ticker": trades.ticker, "window_seconds": 10,
        "source": "alpaca_sip_historical_trades", "price": trades.price_at_10s,
        "first_price": trades.price_first_10s, "first_time": trades.first_trade_at_et,
        "last_time": trades.last_trade_at_et, "event_count": trades.trade_count,
    })
    result["price"] = pd.to_numeric(result.price, errors="coerce")
    return result.loc[result.price.gt(0), list(EARLY_COLUMNS)]


def _historical_thirty_seconds(day: str, path: Path,
                               max_seconds: int = 30) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=EARLY_COLUMNS)
    trades = pd.read_csv(path)
    needed = {"ticker", "window_seconds", "price_at_cutoff", "price_first",
              "first_trade_at_et", "last_trade_at_et", "trade_count", "source"}
    if not needed.issubset(trades):
        return pd.DataFrame(columns=EARLY_COLUMNS)
    trades = trades.loc[trades.source.eq("alpaca_sip_historical_trades")].copy()
    if trades.empty:
        return pd.DataFrame(columns=EARLY_COLUMNS)
    result = pd.DataFrame({
        "session_date": day, "ticker": trades.ticker,
        "window_seconds": pd.to_numeric(trades.window_seconds, errors="coerce"),
        "source": f"alpaca_sip_historical_trades_{max_seconds}s",
        "price": trades.price_at_cutoff,
        "first_price": trades.price_first, "first_time": trades.first_trade_at_et,
        "last_time": trades.last_trade_at_et, "event_count": trades.trade_count,
    })
    result["price"] = pd.to_numeric(result.price, errors="coerce")
    return result.loc[result.price.gt(0) &
                      result.window_seconds.isin(range(10, max_seconds + 1, 10)),
                      list(EARLY_COLUMNS)]


def early_observations(day: str, snapshot_dir: Path = DATA_DIR / "snapshots") -> pd.DataFrame:
    """Usa sólo operaciones/quotes con hora dentro del corte; no interpola."""
    historical = _historical_ten_seconds(day,
                                          snapshot_dir / f"apertura_historica_10s_{day}.csv")
    historical_30 = _historical_thirty_seconds(day,
                                                snapshot_dir / f"apertura_historica_30s_{day}.csv")
    historical_60 = _historical_thirty_seconds(day,
                                                snapshot_dir / f"apertura_historica_60s_{day}.csv",
                                                max_seconds=60)
    sixty_path = snapshot_dir / f"apertura_flujo_60s_{day}.csv"
    live = (_live_windows(day, sixty_path, (10, 20, 30, 60)) if sixty_path.is_file()
            else _live_windows(day, snapshot_dir / f"apertura_flujo_30s_{day}.csv",
                               (10, 20, 30)))
    frames = [frame for frame in (historical, historical_30, historical_60, live)
              if not frame.empty]
    result = (pd.concat(frames, ignore_index=True)
              if frames else pd.DataFrame(columns=EARLY_COLUMNS))
    if result.empty:
        return result
    # La captura local tiene precedencia para el corte de 10 s si hay ambas fuentes.
    result["priority"] = result.source.map({
        "captura_local": 0,
        "alpaca_sip_historical_trades_60s": 1,
        "alpaca_sip_historical_trades_30s": 2,
        "alpaca_sip_historical_trades": 3})
    return (result.sort_values("priority").drop_duplicates(
        ["session_date", "ticker", "window_seconds"]).drop(columns="priority")
        .sort_values(["ticker", "window_seconds"]).reset_index(drop=True))


def early_trajectory(day: str, ticker: str,
                     snapshot_dir: Path = DATA_DIR / "snapshots") -> pd.DataFrame:
    """Trayectoria subminuto real, cuando se conservó; jamás usa velas de 1m."""
    sixty_path = snapshot_dir / f"apertura_flujo_60s_{day}.csv"
    live_path = (sixty_path if sixty_path.is_file() else
                 snapshot_dir / f"apertura_flujo_30s_{day}.csv")
    if live_path.is_file():
        raw = pd.read_csv(live_path)
        if {"ticker", "price", "quote_received_at"}.issubset(raw):
            rows = raw.loc[raw.ticker == ticker].copy()
            opened = pd.Timestamp(datetime.combine(datetime.fromisoformat(day).date(),
                                                     time(9, 30), NY))
            received = pd.to_datetime(rows.quote_received_at, utc=True,
                                      errors="coerce").dt.tz_convert(NY)
            rows["second_after_open"] = (received - opened).dt.total_seconds()
            rows["price"] = pd.to_numeric(rows.price, errors="coerce")
            rows = rows.loc[rows.second_after_open.between(0, 60) & rows.price.gt(0)]
            if not rows.empty:
                return rows[["second_after_open", "price"]].sort_values(
                    "second_after_open").reset_index(drop=True)
    historical_path = snapshot_dir / f"apertura_historica_trayectoria_60s_{day}.csv"
    if not historical_path.is_file():
        historical_path = snapshot_dir / f"apertura_historica_trayectoria_30s_{day}.csv"
    if historical_path.is_file():
        rows = pd.read_csv(historical_path)
        if {"ticker", "second_after_open", "price"}.issubset(rows):
            rows = rows.loc[rows.ticker == ticker, ["second_after_open", "price"]].copy()
            rows["second_after_open"] = pd.to_numeric(rows.second_after_open,
                                                       errors="coerce")
            rows["price"] = pd.to_numeric(rows.price, errors="coerce")
            return rows.loc[rows.second_after_open.between(0, 60) &
                            rows.price.gt(0)].sort_values("second_after_open")
    return pd.DataFrame(columns=["second_after_open", "price"])


def early_features(day: str, ticker: str, candles: pd.DataFrame,
                   observations: pd.DataFrame) -> pd.DataFrame:
    """BB(20,2) 15m provisional: 19 cierres previos + precio conocido al corte."""
    prior = candles.loc[(candles.ticker == ticker) &
                        (candles.session == "PREVIA")].sort_values("bar_index")
    samples = observations.loc[observations.ticker == ticker].sort_values("window_seconds")
    if len(prior) < 24 or samples.empty:
        return pd.DataFrame()
    bands = bollinger_bands(prior)
    tail = bands.tail(5)
    previous_close = float(prior.iloc[-1].close)
    prior_upper = float(bands.iloc[-1].bb_upper)
    prior_lower = float(bands.iloc[-1].bb_lower)
    prior_middle = float(bands.iloc[-1].bb_middle)
    prior_slope = _linear_slope(tail.bb_upper, prior_upper)
    lower_slope = _linear_slope(tail.bb_lower, prior_lower)
    widths = (tail.bb_upper - tail.bb_lower) / tail.bb_middle * 100
    width_change = _linear_slope(widths, 100.0)
    width_growth = (_linear_slope(widths, float(widths.iloc[0]))
                    if widths.iloc[0] > 0 else np.nan)
    deviation = (prior_upper - prior_middle) / 2
    records = []
    for sample in samples.itertuples():
        upper, lower = _provisional_bands(prior, float(sample.price))
        first_upper, first_lower = _provisional_bands(prior, float(sample.first_price))
        first_time = pd.to_datetime(sample.first_time, utc=True, errors="coerce")
        last_time = pd.to_datetime(sample.last_time, utc=True, errors="coerce")
        event_span = ((last_time - first_time).total_seconds()
                      if pd.notna(first_time) and pd.notna(last_time) else np.nan)
        upper_angle = (normalized_band_angle_deg(first_upper, upper, event_span / 60)
                       if np.isfinite(event_span) and event_span > 0 else np.nan)
        lower_angle = (normalized_band_angle_deg(first_lower, lower, event_span / 60)
                       if np.isfinite(event_span) and event_span > 0 else np.nan)
        is_up = float(sample.price) >= previous_close
        target = upper if is_up else lower
        # Positivo siempre significa que la banda objetivo persigue el gap.
        target_angle = upper_angle if is_up else -lower_angle
        distance = ((float(sample.price) / target - 1) * 100 if is_up else
                    (target / float(sample.price) - 1) * 100)
        prior_target = prior_upper if is_up else prior_lower
        prior_distance = ((float(sample.price) / prior_target - 1) * 100 if is_up else
                          (prior_target / float(sample.price) - 1) * 100)
        records.append({
            "session_date": day, "ticker": ticker, "window_seconds": int(sample.window_seconds),
            "source": sample.source, "price": float(sample.price),
            "previous_close": previous_close,
            "gap_pct": (float(sample.price) / previous_close - 1) * 100,
            "prior_upper": prior_upper, "prior_lower": prior_lower,
            "prior_width_pct": (prior_upper - prior_lower) / prior_middle * 100,
            "prior_width_change_pp_per_bar": width_change,
            "prior_width_growth_pct_per_bar": width_growth,
            "prior_upper_slope_pct_per_bar": prior_slope,
            "prior_lower_slope_pct_per_bar": lower_slope,
            "target_band": "UPPER" if is_up else "LOWER",
            "prior_target_slope_pct_per_bar": prior_slope if is_up else lower_slope,
            "upper_provisional": upper, "lower_provisional": lower,
            "distance_above_upper_pct": (float(sample.price) / upper - 1) * 100,
            "target_provisional": target, "distance_outside_target_pct": distance,
            "distance_outside_prior_target_pct": prior_distance,
            "target_band_reaction_pct": (1 if is_up else -1) *
                                        (target / prior_target - 1) * 100,
            "z_at_cutoff": ((float(sample.price) - prior_middle) / deviation
                            if deviation > 0 else np.nan),
            "upper_angle_deg_5m_equiv": upper_angle,
            "lower_angle_deg_5m_equiv": lower_angle,
            "target_angle_deg_5m_equiv": target_angle,
            "event_span_seconds": event_span,
            "event_count": int(sample.event_count),
        })
    return pd.DataFrame.from_records(records)


def build_early_chart(day: str, ticker: str, candles: pd.DataFrame,
                      features: pd.DataFrame, window_seconds: int,
                      trajectory: pd.DataFrame | None = None) -> go.Figure | None:
    visible = features.loc[features.window_seconds <= window_seconds].sort_values("window_seconds")
    prior = candles.loc[(candles.ticker == ticker) &
                        (candles.session == "PREVIA")].sort_values("bar_index").tail(20)
    if prior.empty or visible.empty or window_seconds not in set(visible.window_seconds):
        return None
    bands = bollinger_bands(candles.loc[(candles.ticker == ticker) &
                                        (candles.session == "PREVIA")].sort_values("bar_index"))
    bands = bands.tail(20)
    x = list(range(len(prior)))
    figure = go.Figure()
    figure.add_trace(go.Candlestick(x=x, open=prior.open, high=prior.high,
                                    low=prior.low, close=prior.close,
                                    name="Velas 15 min previas",
                                    increasing_line_color="#4c4c4c",
                                    decreasing_line_color="#aa3535"))
    figure.add_trace(go.Scatter(x=x, y=bands.bb_upper, name="Banda superior previa",
                                line=dict(color="#272727", width=1.5)))
    figure.add_trace(go.Scatter(x=x, y=bands.bb_lower, name="Banda inferior previa",
                                line=dict(color="#8d8d8d", width=1)))
    if trajectory is not None and not trajectory.empty:
        path = trajectory.loc[trajectory.second_after_open <= window_seconds].copy()
        if not path.empty:
            positions = len(prior) + path.second_after_open / 5
            figure.add_trace(go.Scatter(x=positions, y=path.price,
                                        name="Precio observado", mode="lines+markers",
                                        line=dict(color="#ac2626", width=1.5),
                                        marker=dict(size=4)))
            bands_at_path = [_provisional_bands(prior, float(price))
                             for price in path.price]
            figure.add_trace(go.Scatter(x=positions, y=[p[0] for p in bands_at_path],
                                        name="Superior provisional", mode="lines",
                                        line=dict(color="#272727", width=1, dash="dot")))
            figure.add_trace(go.Scatter(x=positions, y=[p[1] for p in bands_at_path],
                                        name="Inferior provisional", mode="lines",
                                        line=dict(color="#8d8d8d", width=1, dash="dot")))
    for row in visible.itertuples():
        position = len(prior) + row.window_seconds / 5
        figure.add_trace(go.Scatter(x=[position], y=[row.price], mode="markers+text",
                                    text=[f"{row.window_seconds} s"],
                                    textposition="top center", name=f"Precio {row.window_seconds} s",
                                    marker=dict(size=9, color="#ac2626")))
        figure.add_trace(go.Scatter(x=[position], y=[row.upper_provisional],
                                    mode="markers", name=f"Superior provisional {row.window_seconds} s",
                                    marker=dict(size=8, color="#272727", symbol="diamond")))
        figure.add_trace(go.Scatter(x=[position], y=[row.lower_provisional],
                                    mode="markers", name=f"Inferior provisional {row.window_seconds} s",
                                    marker=dict(size=7, color="#8d8d8d", symbol="diamond")))
    figure.add_vline(x=len(prior) - .5, line_dash="dash", line_color="#888")
    figure.update_layout(title=f"{day} · {ticker} · hasta {window_seconds} s",
                         height=330, margin=dict(l=8, r=8, t=45, b=15),
                         paper_bgcolor="#f7f7f5", plot_bgcolor="#f7f7f5",
                         font=dict(family="Georgia, serif", size=10),
                         xaxis=dict(tickmode="array", tickvals=[0, 5, 10, 15, 19, 20, 22, 24, 26, 32],
                                    ticktext=["Prev. -20", "Prev. -15", "Prev. -10",
                                              "Prev. -5", "Cierre", "Apertura", "10 s", "20 s",
                                              "30 s", "1 min"],
                                    rangeslider=dict(visible=False)),
                         yaxis=dict(gridcolor="#e4e4e0"),
                         legend=dict(orientation="h", y=-.13, font=dict(size=8)))
    return figure
