"""Gráfica compacta y neutra para operativa e historial."""

import pandas as pd
import plotly.graph_objects as go

from indicators import bollinger_bands
from opening_scan import candle_close_time


def build_daily_chart(candles: pd.DataFrame, ticker: str, gap_pct: float,
                      confirmation_bar_index: int, first_minute_close: float,
                      target_band: str = "UPPER", touch_time: str | None = None,
                      touch_price: float = float("nan"),
                      cutoff_seconds: int | None = None,
                      cutoff_price: float = float("nan"),
                      cutoff_upper: float = float("nan"),
                      cutoff_lower: float = float("nan")) -> go.Figure:
    rows = candles.loc[candles.ticker == ticker].sort_values("bar_index")
    series = bollinger_bands(rows)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=series.bar_index, y=series.bb_lower, mode="lines",
                                name="Banda inferior", line=dict(color="#777777", width=1)))
    figure.add_trace(go.Scatter(x=series.bar_index, y=series.bb_upper, mode="lines",
                                name="Banda superior", line=dict(color="#2b2b2b", width=1.5),
                                fill="tonexty", fillcolor="rgba(100,100,100,.08)"))
    figure.add_trace(go.Scatter(x=series.bar_index, y=series.bb_middle, mode="lines",
                                name="Media 20", line=dict(color="#aaaaaa", width=1, dash="dot")))
    figure.add_trace(go.Candlestick(x=series.bar_index, open=series.open, high=series.high,
                                    low=series.low, close=series.close, name="Velas 15 min",
                                    increasing_line_color="#474747", decreasing_line_color="#b72828",
                                    increasing_fillcolor="#dedede", decreasing_fillcolor="#b72828"))
    current = series.loc[series.session == "ACTUAL"]
    if not current.empty:
        first = current.iloc[0]
        figure.add_vline(x=float(first.bar_index) - .5, line_width=1, line_dash="dash", line_color="#777777")
        figure.add_annotation(x=first.bar_index, y=first.open, text=f"Gap {gap_pct:+.2f}%",
                              showarrow=True, arrowhead=2, ay=-35, bgcolor="#f7f7f7")
    first_index = int(current.iloc[0].bar_index) if not current.empty else len(series)
    if current.empty and not series.empty:
        opening_price = float(series.iloc[-1].close) * (1 + gap_pct / 100)
        figure.add_annotation(x=first_index, y=opening_price, text=f"Gap {gap_pct:+.2f}%",
                              showarrow=True, arrowhead=2, ay=-35, bgcolor="#f7f7f7")
    if cutoff_seconds in (30, 60) and current.empty and pd.notna(cutoff_price):
        label = "30 s" if cutoff_seconds == 30 else "1 min"
        figure.add_trace(go.Scatter(x=[first_index], y=[cutoff_price],
                                    mode="markers+text", text=[label],
                                    textposition="top right", name=f"Precio {label}",
                                    marker=dict(size=11, color="#aa3535")))
        for band_name, band_price, color in (("Superior provisional", cutoff_upper, "#272727"),
                                              ("Inferior provisional", cutoff_lower, "#777777")):
            if pd.notna(band_price):
                figure.add_trace(go.Scatter(x=[first_index], y=[band_price],
                                            mode="markers", name=f"{band_name} {label}",
                                            marker=dict(size=9, color=color, symbol="diamond")))
    if pd.notna(first_minute_close):
        figure.add_trace(go.Scatter(x=[first_index], y=[first_minute_close],
                                    mode="markers", name="09:31 ET",
                                    marker=dict(color="#405c50", size=9, symbol="circle")))
    if confirmation_bar_index >= 0:
        figure.add_vline(x=float(first_index), line_width=1, line_dash="dot",
                         line_color="#405c50")
        figure.add_annotation(x=first_index, y=first_minute_close, text="CONF. 1 MIN",
                              showarrow=True, arrowhead=2, ay=-24, bgcolor="#f7f7f7",
                              font=dict(color="#405c50", size=9))
    if touch_time and pd.notna(touch_time) and pd.notna(touch_price):
        touch_bar = next((int(row.bar_index) for row in current.itertuples()
                          if candle_close_time(str(row.time)) >= str(touch_time)), None)
        if touch_bar is not None:
            figure.add_trace(go.Scatter(x=[touch_bar], y=[touch_price], mode="markers+text",
                                        name="Toque", text=[f"TOQUE {touch_time}"],
                                        textposition="top center",
                                        marker=dict(color="#b72828", size=11, symbol="diamond")))
    tick_indexes = set(range(0, len(series), max(1, len(series) // 4)))
    tick_indexes.add(len(series) - 1)
    if not current.empty:
        opening_index = int(current.index[0])
        tick_indexes = {index for index in tick_indexes
                        if index >= opening_index or opening_index - index > 3}
        tick_indexes.add(opening_index)
    ticks = series.iloc[sorted(tick_indexes)]
    figure.update_layout(
        title=dict(text=f"{ticker}  /  15 MIN  /  {target_band}" +
                   (f"  /  CORTE {cutoff_seconds} S" if cutoff_seconds in (30, 60) else ""),
                   font=dict(family="Georgia, serif", size=17)),
        height=355, margin=dict(l=12, r=12, t=52, b=20),
        paper_bgcolor="#f7f7f7", plot_bgcolor="#f7f7f7", font=dict(color="#333333", size=11),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)),
        xaxis=dict(tickmode="array", tickvals=ticks.bar_index,
                   ticktext=[f"{r.session[0]} {r.time}" for r in ticks.itertuples()],
                   rangeslider=dict(visible=False), gridcolor="#e5e5e5",
                   range=[-1, len(series) + 1]),
        yaxis=dict(gridcolor="#e5e5e5"),
    )
    return figure
