"""Gráfica de Home: velas cerradas de 15m y corte provisional de apertura."""

import math
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go

from indicators import BOLLINGER_DEVIATIONS, BOLLINGER_WINDOW, bollinger_bands


def provisional_15m_bar(candles: pd.DataFrame, ticker: str,
                        minute_bars: pd.DataFrame, now: datetime) -> dict | None:
    """Construye sólo la vela actual con minutos completos ya disponibles."""
    if minute_bars.empty:
        return None
    rows = candles.loc[candles.ticker == ticker].sort_values("bar_index")
    if len(rows) < BOLLINGER_WINDOW - 1:
        return None
    day = now.strftime("%Y-%m-%d")
    latest_complete = (now - timedelta(minutes=1)).strftime("%H:%M")
    minutes = minute_bars.loc[(minute_bars.day == day) &
                              (minute_bars.time >= "09:30") &
                              (minute_bars.time <= latest_complete)].sort_values("time")
    if minutes.empty:
        return None
    last_minute = str(minutes.iloc[-1].time)
    hour, minute = map(int, last_minute.split(":"))
    start = f"{hour:02d}:{(minute // 15) * 15:02d}"
    current = rows.loc[rows.session == "ACTUAL"]
    if not current.empty:
        last_confirmed = str(current.iloc[-1].time)
        last_hour, last_minute_value = map(int, last_confirmed.split(":"))
        expected = f"{last_hour + (last_minute_value + 15) // 60:02d}:" \
                   f"{(last_minute_value + 15) % 60:02d}"
    else:
        expected = "09:30"
    if start != expected:
        return None
    part = minutes.loc[minutes.time >= start]
    closes = pd.concat([rows.close.astype(float).tail(BOLLINGER_WINDOW - 1),
                        pd.Series([float(part.iloc[-1].close)])], ignore_index=True)
    middle = float(closes.mean())
    deviation = float(closes.std(ddof=0))
    return {"bar_index": float(rows.iloc[-1].bar_index) + 1,
            "time": start, "last_minute": last_minute,
            "open": float(part.iloc[0].open), "high": float(part.high.max()),
            "low": float(part.low.min()), "close": float(part.iloc[-1].close),
            "bb_middle": middle,
            "bb_upper": middle + BOLLINGER_DEVIATIONS * deviation,
            "bb_lower": middle - BOLLINGER_DEVIATIONS * deviation}


def build_home_chart(candles: pd.DataFrame, ticker: str, cutoff_seconds: int,
                     cutoff_price: float, cutoff_upper: float, cutoff_lower: float,
                     gap_pct: float, minute_bars: pd.DataFrame | None = None,
                     now: datetime | None = None) -> go.Figure:
    """No dibuja una vela de 15m hasta que exista su OHLC completo."""
    rows = candles.loc[candles.ticker == ticker].sort_values("bar_index").copy()
    figure = go.Figure()
    if rows.empty:
        return figure
    series = bollinger_bands(rows)
    current = series.loc[series.session == "ACTUAL"]
    opening_index = (int(current.iloc[0].bar_index) if not current.empty
                     else int(series.iloc[-1].bar_index) + 1)
    visible = series.loc[series.bar_index.between(opening_index - 8,
                                                  opening_index + 11)].copy()
    x = visible.bar_index.astype(float)
    figure.add_trace(go.Scatter(
        x=x, y=visible.bb_upper, mode="lines", name="Banda superior · BB(20,2)",
        line=dict(color="#c4cbd0", width=1.5), hovertemplate="Superior %{y:.2f}<extra></extra>"))
    figure.add_trace(go.Scatter(
        x=x, y=visible.bb_lower, mode="lines", name="Banda inferior · BB(20,2)",
        line=dict(color="#c4cbd0", width=1.5), fill="tonexty",
        fillcolor="rgba(171,185,194,0.15)", hovertemplate="Inferior %{y:.2f}<extra></extra>"))
    figure.add_trace(go.Scatter(
        x=x, y=visible.bb_middle, mode="lines", name="Media 20",
        line=dict(color="#82939d", width=1, dash="dot"),
        hovertemplate="Media %{y:.2f}<extra></extra>"))
    figure.add_trace(go.Candlestick(
        x=x, open=visible.open, high=visible.high, low=visible.low,
        close=visible.close, name="Velas 15 min",
        increasing_line_color="#35a690", increasing_fillcolor="#35a690",
        decreasing_line_color="#d05c60", decreasing_fillcolor="#d05c60"))
    provisional = (provisional_15m_bar(candles, ticker, minute_bars, now)
                   if minute_bars is not None and now is not None else None)
    if provisional:
        px = provisional["bar_index"]
        figure.add_trace(go.Candlestick(
            x=[px], open=[provisional["open"]], high=[provisional["high"]],
            low=[provisional["low"]], close=[provisional["close"]],
            name=f"Vela en formación · hasta {provisional['last_minute']}",
            increasing_line_color="#57c5ac", increasing_fillcolor="#57c5ac",
            decreasing_line_color="#ee7779", decreasing_fillcolor="#ee7779",
            opacity=0.65))
        for name, value, color in (
                ("Superior provisional actual", provisional["bb_upper"], "#ecf1f3"),
                ("Inferior provisional actual", provisional["bb_lower"], "#ecf1f3"),
                ("Media provisional actual", provisional["bb_middle"], "#82939d")):
            figure.add_trace(go.Scatter(
                x=[float(rows.iloc[-1].bar_index), px],
                y=[float(series.iloc[-1][{"Superior provisional actual": "bb_upper",
                                         "Inferior provisional actual": "bb_lower",
                                         "Media provisional actual": "bb_middle"}[name]]), value],
                mode="lines+markers", name=name, showlegend=False,
                line=dict(color=color, width=1.2, dash="dot"),
                marker=dict(size=6)))

    # Los tres valores pertenecen al corte elegido. Un punto evita inventar
    # el máximo, mínimo y cierre definitivo de una vela todavía abierta.
    cutoff_x = opening_index - 0.22
    prior = series.loc[series.session == "PREVIA"]
    if not prior.empty:
        last = prior.iloc[-1]
        for label, previous, band_at_cut, color in (
                ("Superior al corte", float(last.bb_upper), cutoff_upper, "#5cc2dd"),
                ("Inferior al corte", float(last.bb_lower), cutoff_lower, "#b9a5e8")):
            if math.isfinite(previous) and math.isfinite(band_at_cut):
                figure.add_trace(go.Scatter(
                    x=[float(last.bar_index), cutoff_x], y=[previous, band_at_cut],
                    mode="lines+markers", name=label, showlegend=False,
                    line=dict(color=color, width=3, dash="dash"),
                    marker=dict(color=color, size=[5, 12], symbol="diamond"),
                    hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"))
    for label, price, color, symbol in (
            ("Precio del corte", cutoff_price, "#f3cc70", "diamond"),
            ("Banda superior provisional", cutoff_upper, "#5cc2dd", "circle"),
            ("Banda inferior provisional", cutoff_lower, "#b9a5e8", "circle")):
        if math.isfinite(price):
            figure.add_trace(go.Scatter(
                x=[cutoff_x], y=[price], mode="markers", name=label,
                showlegend=label == "Precio del corte",
                marker=dict(color=color, size=11 if label == "Precio del corte" else 8,
                            symbol=symbol, line=dict(color="#1c252d", width=1)),
                hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"))
    figure.add_vline(x=opening_index - 0.5, line_color="#73828b",
                     line_width=1, line_dash="dash")
    figure.add_annotation(x=cutoff_x, y=cutoff_price,
                          text=f"{cutoff_seconds} s · gap {gap_pct:+.2f}%",
                          showarrow=True, arrowhead=2, ax=75, ay=-38,
                          font=dict(color="#f0f2f3", size=11),
                          bgcolor="#28343d", bordercolor="#55636b")
    tick_rows = visible.iloc[::max(1, len(visible) // 7)]
    figure.update_layout(
        title=dict(text=f"{ticker}  |  apertura y BB(20,2) · 15 min",
                   font=dict(size=17, color="#f0f2f3")),
        height=470, margin=dict(l=12, r=18, t=72, b=38),
        paper_bgcolor="#172027", plot_bgcolor="#172027",
        font=dict(family="Arial, sans-serif", color="#c4cbd0", size=11),
        legend=dict(orientation="h", y=1.04, x=0,
                    font=dict(size=10, color="#dbe3e7"),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified", dragmode="pan",
        xaxis=dict(tickmode="array", tickvals=tick_rows.bar_index,
                   ticktext=[f"{'Día previo' if row.session == 'PREVIA' else 'Hoy'} {row.time}"
                             for row in tick_rows.itertuples()],
                   tickangle=-20, gridcolor="#2b3942", zeroline=False,
                   rangeslider=dict(visible=False),
                   range=[max(float(visible.bar_index.min()) - 0.6,
                              opening_index - 8.6),
                          max(float(visible.bar_index.max()) + 0.7,
                              opening_index + 0.7,
                              provisional["bar_index"] + 0.7 if provisional else 0)]),
        yaxis=dict(title="Precio", side="right", gridcolor="#2b3942",
                   zeroline=False, tickformat=".2f", fixedrange=False),
    )
    return figure
