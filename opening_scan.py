"""Diagnóstico temporal de gaps y reversión hacia Bandas de Bollinger."""

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from indicators import BOLLINGER_DEVIATIONS, BOLLINGER_WINDOW, bollinger_bands
from scanner_settings import (DEFAULT_THRESHOLDS, SCORE_SCALES, SCORE_WEIGHTS,
                              MAX_5MIN_BAND_SLOPE_PCT_PER_MIN, MAX_UPPER_ANGLE_DEG,
                              SCORE_CUTOFF_WIDTH_RATIO_GRACE,
                              SCORE_CUTOFF_WIDTH_PENALTY_PER_RATIO,
                              SCORE_CUTOFF_WIDTH_PENALTY_MAX,
                              ScannerThresholds)


SETUP_COLUMNS = [
    "ticker", "gap_pct", "gap_direction", "target_band", "expected_reversion_direction",
    "opening_price", "prior_close", "prior_bb_upper", "prior_bb_lower",
    "cutoff_bb_upper", "cutoff_bb_lower", "cutoff_target_band",
    "cutoff_target_band_reaction_pct", "prior_distance_to_band_pct",
    "target_angle_10s_deg", "angle_10s_event_span_seconds",
    "target_angle_cutoff_deg", "angle_cutoff_event_span_seconds", "cutoff_seconds",
    "prior_bb_width_pct", "prior_target_band_slope_pct", "prior_bb_width_change",
    "prior_bb_width_growth_pct", "distance_to_band_pct", "z_open",
    "five_min_band_open", "five_min_band_end", "five_min_band_slope_pct_per_min",
    "upper_angle_0930_0931", "upper_angle_0930_0935",
    "target_angle_deg", "target_angle_1m_deg", "target_angle_5m_deg",
    "target_angle_0930_0935", "target_angle_0935_0940",
    "target_angle_0940_0945", "target_angle_score",
    "score_gap_component", "score_distance_component",
    "score_compression_component", "score_prior_width_growth_component",
    "score_target_angle_component", "score_first_minute_component",
    "score_post_open_expansion_component", "score_cutoff_expansion_component",
    "first_minute_open", "first_minute_high", "first_minute_low", "first_minute_close",
    "first_minute_return_pct", "first_minute_body_pct", "first_minute_reversal_pct",
    "post_open_bb_width_pct", "post_open_bb_width_change",
    "post_open_bb_expansion_pct", "post_open_target_band_slope_pct",
    "distance_to_band_after_15m", "score_open", "score_1m", "score_5m",
    "score_15m", "score",
    "candidate_at_open", "candidate", "confirmed", "failed_conditions",
    "pending_conditions", "data_status", "signal_stage", "confirmation_time",
    "confirmation_bar_index", "touched_target_band", "touch_time",
    "bars_until_touch", "minutes_until_touch", "target_band_price_at_touch",
    "touch_status", "reversion_to_touch_band_pct", "band_at_or_beyond_open_at_touch",
]
EVENT_COLUMNS = ["ticker", "bar_index", "time", "price", "kind", "target_band",
                 "touched_target_band", "touch_time", "bars_until_touch",
                 "minutes_until_touch", "target_band_price_at_touch"]
HARD_OPEN_CONDITIONS = ("gap", "band_width", "band_slope", "band_growth", "distance")
UPPER_ANGLE_COLUMNS = [
    "date", "ticker", "previous_close", "opening_price", "gap_pct",
    "prior_upper_band", "distance_above_upper_pct", "upper_band_0930",
    "upper_band_0935", "upper_angle_0930_0935", "upper_band_0940",
    "upper_angle_0935_0940", "upper_band_0945", "upper_angle_0940_0945",
    "candidate", "failed_conditions",
]
MINUTE_BAND_COLUMNS = [
    "date", "ticker", "time", "upper_band_provisional", "lower_band_provisional",
    "upper_angle_deg", "lower_angle_deg",
]


def candle_close_time(start: str) -> str:
    return (datetime.strptime(start, "%H:%M") + timedelta(minutes=15)).strftime("%H:%M")


def _linear_slope(values: pd.Series | np.ndarray, scale: float = 1.0) -> float:
    points = np.asarray(values, dtype=float)
    if len(points) < 2 or not np.isfinite(points).all() or not np.isfinite(scale) or scale == 0:
        return np.nan
    x = np.arange(len(points), dtype=float)
    return float(np.polyfit(x, points, 1)[0] / scale * 100)


def _width(upper: float, lower: float, middle: float) -> float:
    return (upper - lower) / middle * 100 if middle > 0 else np.nan


def _distance(opening: float, band: float, sign: int) -> float:
    if opening <= 0 or band <= 0:
        return np.nan
    return (opening / band - 1) * 100 if sign == 1 else (band / opening - 1) * 100


def _passes(value: float, limit: float | None, minimum: bool = False) -> bool | None:
    if limit is None:
        return None
    if not np.isfinite(value):
        return False
    return bool(value >= limit if minimum else value <= limit)


def normalized_band_angle_deg(band_previous: float, band_actual: float,
                              delta_minutes: int) -> float:
    """Ángulo reproducible: cambio porcentual escalado a cinco minutos."""
    if (delta_minutes <= 0 or not np.isfinite(band_previous) or band_previous <= 0
            or not np.isfinite(band_actual)):
        return np.nan
    band_change_pct = (band_actual / band_previous - 1) * 100
    normalized_change_5m = band_change_pct * (5 / delta_minutes)
    return math.degrees(math.atan(normalized_change_5m))


def _provisional_bands(prior: pd.DataFrame, current_close: float) -> tuple[float, float]:
    """BB(20,2) de 15m con el cierre conocido de la vela 09:30 aún abierta."""
    closes = np.append(prior.close.astype(float).tail(BOLLINGER_WINDOW - 1), current_close)
    middle = float(closes.mean())
    deviation = float(closes.std(ddof=0))
    return (middle + BOLLINGER_DEVIATIONS * deviation,
            middle - BOLLINGER_DEVIATIONS * deviation)


def _target_angle_quality(angle: float) -> float:
    if angle <= 0:
        return 1.0
    if angle <= MAX_UPPER_ANGLE_DEG:
        return 1 - 0.6 * angle / MAX_UPPER_ANGLE_DEG
    if angle >= 70.0:
        return 0.0
    return 0.4 * (70.0 - angle) / (70.0 - MAX_UPPER_ANGLE_DEG)


def _scored_angle(metrics: dict, stage: str) -> float:
    if stage == "open":
        angle = metrics.get("target_angle_cutoff_deg", np.nan)
        return angle if np.isfinite(angle) else metrics.get("target_angle_10s_deg", np.nan)
    if stage == "1m":
        return metrics["target_angle_1m_deg"]
    angles = [metrics["target_angle_5m_deg"]]
    if stage == "15m":
        angles.extend((metrics["target_angle_0935_0940"],
                       metrics["target_angle_0940_0945"]))
    observed = [angle for angle in angles if np.isfinite(angle)]
    return max(observed) if observed else np.nan


def upper_angle_openings(candles: pd.DataFrame, minute_bars: pd.DataFrame,
                         date: str, max_upper_angle_deg: float = MAX_UPPER_ANGLE_DEG,
                         include_final_15m: bool = True
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aperturas fuera de BB(20,2) y evolución provisional cada minuto.

    Cada cierre de 1m sustituye únicamente al cierre aún desconocido de la
    primera vela de 15m. La fila 09:45 usa la vela de 15m ya cerrada.
    """
    setups = []
    minute_series = []
    if candles.empty:
        return (pd.DataFrame(columns=UPPER_ANGLE_COLUMNS),
                pd.DataFrame(columns=MINUTE_BAND_COLUMNS))
    for ticker, raw in candles.groupby("ticker", sort=True):
        bars = _ordered_bars(raw)
        prior = bars.loc[bars.session == "PREVIA"]
        current = bars.loc[bars.session == "ACTUAL"]
        if len(prior) < BOLLINGER_WINDOW or current.empty:
            continue
        prior_bands = bollinger_bands(prior)
        previous_close = float(prior.iloc[-1].close)
        prior_upper = float(prior_bands.iloc[-1].bb_upper)
        prior_lower = float(prior_bands.iloc[-1].bb_lower)
        ticker_minutes = minute_bars.loc[minute_bars.ticker == ticker].sort_values("time")
        first_minute = ticker_minutes.loc[ticker_minutes.time.astype(str) == "09:30"]
        opening = (float(first_minute.iloc[0].open)
                   if not first_minute.empty and "open" in first_minute else
                   float(current.iloc[0].open))
        gap_pct = (opening / previous_close - 1) * 100
        distance = (opening / prior_upper - 1) * 100
        base_closes = prior.close.astype(float).tail(BOLLINGER_WINDOW - 1).to_numpy()
        stamps = {"09:30": (prior_upper, prior_lower)}
        observations = [("09:30", prior_upper, prior_lower)]
        for bar in ticker_minutes.itertuples():
            start = str(bar.time)
            if not "09:30" <= start <= "09:43":
                continue
            end = (datetime.strptime(start, "%H:%M") + timedelta(minutes=1)).strftime("%H:%M")
            closes = np.append(base_closes, float(bar.close))
            middle = float(closes.mean())
            deviation = float(closes.std(ddof=0))
            upper = middle + BOLLINGER_DEVIATIONS * deviation
            lower = middle - BOLLINGER_DEVIATIONS * deviation
            stamps[end] = (upper, lower)
            observations.append((end, upper, lower))
        # Sólo el cierre definitivo de la primera vela de 15m da las 09:45.
        first = current.loc[current.time.astype(str) == "09:30"]
        if include_final_15m and not first.empty:
            closes = np.append(base_closes, float(first.iloc[0].close))
            middle = float(closes.mean())
            deviation = float(closes.std(ddof=0))
            stamps["09:45"] = (middle + BOLLINGER_DEVIATIONS * deviation,
                               middle - BOLLINGER_DEVIATIONS * deviation)
            observations.append(("09:45", *stamps["09:45"]))
        previous_time = None
        previous_upper = previous_lower = np.nan
        for time, upper, lower in sorted(observations):
            delta = (int(time[:2]) * 60 + int(time[3:]) -
                     int(previous_time[:2]) * 60 - int(previous_time[3:])) if previous_time else 0
            minute_series.append({
                "date": date, "ticker": ticker, "time": time,
                "upper_band_provisional": upper, "lower_band_provisional": lower,
                "upper_angle_deg": normalized_band_angle_deg(previous_upper, upper, delta),
                "lower_angle_deg": normalized_band_angle_deg(previous_lower, lower, delta),
            })
            previous_time, previous_upper, previous_lower = time, upper, lower
        def interval_angle(start: str, end: str) -> float:
            return (normalized_band_angle_deg(stamps[start][0], stamps[end][0], 5)
                    if start in stamps and end in stamps else np.nan)
        first_angle = interval_angle("09:30", "09:35")
        failures = []
        if gap_pct <= 0:
            failures.append("gap_not_up")
        if opening <= prior_upper:
            failures.append("opening_not_above_upper")
        if not np.isfinite(first_angle):
            failures.append("missing_upper_angle_0930_0935")
        elif first_angle > max_upper_angle_deg:
            failures.append("upper_angle_0930_0935")
        setups.append({
            "date": date, "ticker": ticker, "previous_close": previous_close,
            "opening_price": opening, "gap_pct": gap_pct,
            "prior_upper_band": prior_upper, "distance_above_upper_pct": distance,
            "upper_band_0930": prior_upper,
            "upper_band_0935": stamps.get("09:35", (np.nan,))[0],
            "upper_angle_0930_0935": first_angle,
            "upper_band_0940": stamps.get("09:40", (np.nan,))[0],
            "upper_angle_0935_0940": interval_angle("09:35", "09:40"),
            "upper_band_0945": stamps.get("09:45", (np.nan,))[0],
            "upper_angle_0940_0945": interval_angle("09:40", "09:45"),
            "candidate": not failures, "failed_conditions": ",".join(failures),
        })
    result = pd.DataFrame.from_records(setups, columns=UPPER_ANGLE_COLUMNS)
    if not result.empty:
        result = result.sort_values(["candidate", "distance_above_upper_pct", "ticker"],
                                    ascending=[False, False, True]).reset_index(drop=True)
    return result, pd.DataFrame.from_records(minute_series, columns=MINUTE_BAND_COLUMNS)


def _score(metrics: dict, stage: str, breakdown: dict | None = None) -> float:
    """Parecido geométrico 0–100; las escalas son exploratorias, no calibradas."""
    if not np.isfinite(metrics["distance_to_band_pct"]) or metrics["distance_to_band_pct"] <= 0:
        if breakdown is not None:
            angle = _scored_angle(metrics, stage)
            breakdown["target_angle_score"] = (_target_angle_quality(angle)
                                                 if np.isfinite(angle) else np.nan)
            for column in ("score_gap_component", "score_distance_component",
                           "score_compression_component", "score_prior_width_growth_component",
                           "score_target_angle_component", "score_first_minute_component",
                           "score_post_open_expansion_component",
                           "score_cutoff_expansion_component"):
                breakdown[column] = 0.0
        return 0.0
    values = {
        "gap": min(abs(metrics["gap_pct"]) / SCORE_SCALES["gap_pct"], 1.0),
        "distance": min(metrics["distance_to_band_pct"] / SCORE_SCALES["distance_pct"], 1.0),
        "compression": 1 / (1 + max(metrics["prior_bb_width_pct"], 0) /
                            SCORE_SCALES["bb_width_pct"]),
        "prior_width_growth": 1 / (1 + max(metrics["prior_bb_width_change"], 0) /
                                   SCORE_SCALES["prior_width_growth_pp_per_bar"]),
    }
    angle = _scored_angle(metrics, stage)
    if np.isfinite(angle):
        values["target_angle"] = _target_angle_quality(angle)
    if stage in ("1m", "5m", "15m") and np.isfinite(metrics["first_minute_reversal_pct"]):
        values["first_minute_reversal"] = min(
            max(metrics["first_minute_reversal_pct"], 0) /
            SCORE_SCALES["first_minute_reversal_pct"], 1.0)
    if stage == "15m" and np.isfinite(metrics["post_open_bb_expansion_pct"]):
        values["post_open_expansion"] = 1 / (
            1 + max(metrics["post_open_bb_expansion_pct"], 0) /
            SCORE_SCALES["post_open_expansion_pct"])
    available = {key: value for key, value in values.items() if np.isfinite(value)}
    total = sum(SCORE_WEIGHTS[key] for key in available)
    cutoff_penalty = 0.0
    if stage == "open":
        upper = metrics.get("cutoff_bb_upper", np.nan)
        lower = metrics.get("cutoff_bb_lower", np.nan)
        prior_width = metrics.get("prior_bb_width_pct", np.nan)
        if (np.isfinite(upper) and np.isfinite(lower) and
                np.isfinite(prior_width) and prior_width > 0):
            cutoff_width = _width(upper, lower, (upper + lower) / 2)
            if np.isfinite(cutoff_width):
                width_ratio = cutoff_width / prior_width
                cutoff_penalty = min(SCORE_CUTOFF_WIDTH_PENALTY_MAX,
                                     SCORE_CUTOFF_WIDTH_PENALTY_PER_RATIO *
                                     max(width_ratio - SCORE_CUTOFF_WIDTH_RATIO_GRACE, 0))
    if breakdown is not None:
        breakdown["target_angle_score"] = available.get("target_angle", np.nan)
        breakdown["score_cutoff_expansion_component"] = -cutoff_penalty
        for component, key in (
                ("score_gap_component", "gap"),
                ("score_distance_component", "distance"),
                ("score_compression_component", "compression"),
                ("score_prior_width_growth_component", "prior_width_growth"),
                ("score_target_angle_component", "target_angle"),
                ("score_first_minute_component", "first_minute_reversal"),
                ("score_post_open_expansion_component", "post_open_expansion")):
            breakdown[component] = (100 * SCORE_WEIGHTS[key] * available[key] / total
                                    if key in available else 0.0)
    return round(max(0.0, 100 * sum(SCORE_WEIGHTS[key] * value
                                    for key, value in available.items()) / total
                     - cutoff_penalty), 1)


def _ordered_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Orden cronológico explícito; rechaza índices que mezclen sesiones."""
    bars = raw.sort_values("bar_index").copy()
    if bars.bar_index.duplicated().any():
        raise ValueError("bar_index duplicado dentro de un ticker")
    if bars.session.isin(["PREVIA", "ACTUAL"]).all():
        seen_actual = (bars.session == "ACTUAL").cummax()
        if ((bars.session == "PREVIA") & seen_actual).any():
            raise ValueError("Velas PREVIA después de ACTUAL")
    else:
        raise ValueError("Sesión distinta de PREVIA/ACTUAL")
    for session in ("PREVIA", "ACTUAL"):
        times = bars.loc[bars.session == session, "time"].astype(str)
        if not times.is_monotonic_increasing:
            raise ValueError(f"Horarios {session} fuera de orden")
    return bars


def opening_setups(candles: pd.DataFrame, first_minutes: pd.DataFrame | None = None,
                   mode: str = "DIAGNOSTIC",
                   thresholds: ScannerThresholds = DEFAULT_THRESHOLDS,
                   opening_prices: dict[str, float] | None = None,
                   early_minutes: pd.DataFrame | None = None,
                   five_minutes: pd.DataFrame | None = None,
                   cutoff_seconds: int | None = None,
                   opening_events: dict[str, list[tuple[datetime, float]]] | None = None
                   ) -> pd.DataFrame:
    """Una fila por ticker; métricas posteriores aparecen sólo cuando hay vela cerrada.

    candidate_at_open usa exclusivamente velas PREVIA y precio de apertura.
    candidate puede incorporar el límite posterior tras las 09:45. La vela de
    1 minuto es característica/confirmación independiente, no descarte duro.
    """
    if mode not in ("DIAGNOSTIC", "FILTERED"):
        raise ValueError("mode debe ser DIAGNOSTIC o FILTERED")
    if first_minutes is None:
        first_minutes = pd.DataFrame(columns=["ticker", "open", "high", "low", "close"])
    if early_minutes is None:
        early_minutes = pd.DataFrame(columns=["ticker", "time", "open", "high", "low", "close"])
    use_1m_for_five = five_minutes is None
    if five_minutes is None:
        five_minutes = pd.DataFrame(columns=["ticker", "time", "open", "high", "low", "close"])
    opening_prices = opening_prices or {}
    opening_events = opening_events or {}
    if candles.empty:
        return pd.DataFrame(columns=SETUP_COLUMNS)
    found = []
    for ticker, raw in candles.groupby("ticker", sort=True):
        bars = _ordered_bars(raw)
        prior = bars.loc[bars.session == "PREVIA"]
        current = bars.loc[bars.session == "ACTUAL"]
        minute_rows = first_minutes.loc[first_minutes.ticker == ticker]
        minute = minute_rows.iloc[0] if not minute_rows.empty else None
        first = current.iloc[0] if not current.empty else None
        record = dict.fromkeys(SETUP_COLUMNS, np.nan)
        record.update(ticker=ticker, candidate=False, candidate_at_open=False,
                      confirmed=-1, touched_target_band=False,
                      confirmation_time="", confirmation_bar_index=np.nan,
                      failed_conditions="", pending_conditions="", signal_stage="PRE_OPEN",
                      touch_status="PENDING")
        if len(prior) < 24:
            record.update(data_status="INSUFFICIENT_PRIOR_BARS",
                          failed_conditions="insufficient_prior_bars")
            found.append(record)
            continue
        bands = bollinger_bands(prior)
        tail = bands.tail(5)
        last = bands.iloc[-1]
        upper, lower, middle = map(float, (last.bb_upper, last.bb_lower, last.bb_middle))
        widths = (tail.bb_upper - tail.bb_lower) / tail.bb_middle * 100
        width_change = _linear_slope(widths, 100.0)  # anchos ya están en porcentaje: puntos por vela
        width_growth = (_linear_slope(widths, float(widths.iloc[0]))
                        if widths.iloc[0] > 0 else np.nan)
        record.update(prior_close=float(last.close), prior_bb_upper=upper,
                      prior_bb_lower=lower, prior_bb_width_pct=_width(upper, lower, middle),
                      prior_bb_width_change=width_change,
                      prior_bb_width_growth_pct=width_growth)
        if minute is None and first is None and ticker not in opening_prices:
            record.update(data_status="MISSING_OPEN", failed_conditions="missing_open")
            found.append(record)
            continue
        opening = (float(opening_prices[ticker]) if ticker in opening_prices else
                   float(minute.open) if minute is not None else float(first.open))
        gap = (opening / float(last.close) - 1) * 100
        sign = 1 if gap >= 0 else -1
        target = "UPPER" if sign == 1 else "LOWER"
        band = upper if sign == 1 else lower
        target_values = tail.bb_upper if sign == 1 else tail.bb_lower
        slope = _linear_slope(target_values, float(target_values.iloc[-1]))
        distance = _distance(opening, band, sign)
        subminute_cutoff = (cutoff_seconds in (10, 30, 60, 90) and ticker in opening_prices
                            and minute is None and first is None)
        if subminute_cutoff:
            cutoff_upper, cutoff_lower = _provisional_bands(prior, opening)
            cutoff_band = cutoff_upper if sign == 1 else cutoff_lower
            prior_distance = distance
            distance = _distance(opening, cutoff_band, sign)
            record.update(cutoff_bb_upper=cutoff_upper, cutoff_bb_lower=cutoff_lower,
                          cutoff_target_band=cutoff_band,
                          cutoff_target_band_reaction_pct=sign * (cutoff_band / band - 1) * 100,
                          prior_distance_to_band_pct=prior_distance,
                          cutoff_seconds=cutoff_seconds)
            events = sorted(opening_events.get(ticker, []), key=lambda event: event[0])
            if len(events) >= 2:
                elapsed = (events[-1][0] - events[0][0]).total_seconds()
                if elapsed > 0:
                    first_upper, first_lower = _provisional_bands(prior, events[0][1])
                    last_upper, last_lower = _provisional_bands(prior, events[-1][1])
                    first_band = first_upper if sign == 1 else first_lower
                    last_band = last_upper if sign == 1 else last_lower
                    angle = sign * normalized_band_angle_deg(
                        first_band, last_band, elapsed / 60)
                    record["target_angle_cutoff_deg"] = angle
                    record["angle_cutoff_event_span_seconds"] = elapsed
                    if cutoff_seconds == 10:
                        record["target_angle_10s_deg"] = angle
                        record["angle_10s_event_span_seconds"] = elapsed
        deviation = float(last.bb_upper - last.bb_middle) / BOLLINGER_DEVIATIONS
        z_open = (opening - middle) / deviation if deviation > 0 else np.nan
        record.update(gap_pct=gap, gap_direction="UP" if sign == 1 else "DOWN",
                      target_band=target, expected_reversion_direction="DOWN" if sign == 1 else "UP",
                      opening_price=opening, prior_target_band_slope_pct=slope,
                      distance_to_band_pct=distance, z_open=z_open,
                      data_status="OPEN", signal_stage="09:30", direction_sign=sign)
        checks = {
            "gap": _passes(abs(gap), thresholds.minimum_gap_pct, minimum=True),
            "band_width": _passes(record["prior_bb_width_pct"],
                                  thresholds.maximum_prior_bb_width_pct),
            "band_slope": _passes(sign * slope, thresholds.maximum_target_band_slope_pct),
            "band_growth": _passes(width_change, thresholds.maximum_prior_bb_width_growth),
            "distance": _passes(distance, thresholds.minimum_distance_to_band_pct,
                                minimum=True),
        }
        if subminute_cutoff:
            # La pendiente previa no sustituye una medición dentro de la apertura.
            checks.pop("band_slope")
            angle_key = "target_angle_10s" if cutoff_seconds == 10 else "target_angle_cutoff"
            checks[angle_key] = (_passes(record["target_angle_cutoff_deg"],
                                         MAX_UPPER_ANGLE_DEG)
                                 if pd.notna(record["target_angle_cutoff_deg"]) else None)
            hard_open_conditions = ("gap", "band_width", "band_growth", "distance",
                                    angle_key)
        else:
            hard_open_conditions = HARD_OPEN_CONDITIONS
        record["candidate_at_open"] = bool(distance > 0 and all(
            checks[key] is not False for key in hard_open_conditions))
        if minute is not None:
            minute_open, minute_close = float(minute.open), float(minute.close)
            minute_return = (minute_close / minute_open - 1) * 100
            reversal = -sign * minute_return
            record.update(first_minute_open=minute_open, first_minute_high=float(minute.high),
                          first_minute_low=float(minute.low), first_minute_close=minute_close,
                          first_minute_return_pct=minute_return,
                          first_minute_body_pct=abs(minute_return),
                          first_minute_reversal_pct=reversal,
                          confirmation_time="09:31", signal_stage="09:31")
            checks["first_minute_confirmation"] = _passes(
                reversal, thresholds.minimum_first_minute_reversal_pct, minimum=True)
            remaining_distance = _distance(minute_close, band, sign)
            record["confirmed"] = int(
                reversal > 0 and remaining_distance > 0
                and (thresholds.minimum_distance_to_band_pct is None or
                     remaining_distance >= thresholds.minimum_distance_to_band_pct)
                and checks["first_minute_confirmation"] is not False)
            upper_1m, lower_1m = _provisional_bands(prior, minute_close)
            angle_1m = normalized_band_angle_deg(band,
                                                 upper_1m if sign == 1 else lower_1m, 1)
            record["target_angle_1m_deg"] = sign * angle_1m
            if sign == 1:
                record["upper_angle_0930_0931"] = angle_1m
        else:
            checks["first_minute_confirmation"] = None
        early = early_minutes.loc[early_minutes.ticker == ticker].sort_values("time")
        five = five_minutes.loc[five_minutes.ticker == ticker].sort_values("time")
        first_five = five.loc[five.time.astype(str) == "09:30"]
        first_five_close = float(first_five.iloc[0].close) if len(first_five) == 1 else np.nan
        if not np.isfinite(first_five_close) and use_1m_for_five:
            first_five_1m = early.loc[early.time.isin(["09:31", "09:32", "09:33", "09:34"])]
            if minute is not None and first_five_1m.time.astype(str).tolist() == [
                    "09:31", "09:32", "09:33", "09:34"]:
                first_five_close = float(first_five_1m.iloc[-1].close)
        band_0940 = np.nan
        if np.isfinite(first_five_close):
            upper_0935, lower_0935 = _provisional_bands(prior, first_five_close)
            target_0935 = upper_0935 if sign == 1 else lower_0935
            angle_0935 = sign * normalized_band_angle_deg(band, target_0935, 5)
            record.update(target_angle_5m_deg=angle_0935,
                          target_angle_0930_0935=angle_0935,
                          signal_stage="09:35")
            if sign == 1:
                record.update(five_min_band_open=upper, five_min_band_end=upper_0935,
                              five_min_band_slope_pct_per_min=(upper_0935 / upper - 1) * 100 / 5,
                              upper_angle_0930_0935=angle_0935)
            second_five = five.loc[five.time.astype(str) == "09:35"]
            second_close = float(second_five.iloc[0].close) if len(second_five) == 1 else np.nan
            if not np.isfinite(second_close) and use_1m_for_five:
                second_1m = early.loc[early.time.astype(str) == "09:39"]
                if len(second_1m) == 1:
                    second_close = float(second_1m.iloc[0].close)
            if np.isfinite(second_close):
                upper_0940, lower_0940 = _provisional_bands(prior, second_close)
                band_0940 = upper_0940 if sign == 1 else lower_0940
                record["target_angle_0935_0940"] = sign * normalized_band_angle_deg(
                    target_0935, band_0940, 5)
                record["signal_stage"] = "09:40"
        record["target_angle_deg"] = (record["target_angle_5m_deg"]
                                      if np.isfinite(record["target_angle_5m_deg"])
                                      else record["target_angle_1m_deg"])
        if first is not None:
            # La primera vela ACTUAL sólo entra después de cerrar a las 09:45.
            post = bollinger_bands(pd.concat([prior, current.iloc[[0]]])).iloc[-1]
            post_upper, post_lower, post_middle = map(
                float, (post.bb_upper, post.bb_lower, post.bb_middle))
            post_width = _width(post_upper, post_lower, post_middle)
            post_band = post_upper if sign == 1 else post_lower
            if np.isfinite(band_0940):
                record["target_angle_0940_0945"] = sign * normalized_band_angle_deg(
                    band_0940, post_band, 5)
            record.update(post_open_bb_width_pct=post_width,
                          post_open_bb_width_change=post_width - record["prior_bb_width_pct"],
                          post_open_bb_expansion_pct=(
                              (post_width / record["prior_bb_width_pct"] - 1) * 100
                              if record["prior_bb_width_pct"] > 0 else np.nan),
                          post_open_target_band_slope_pct=(post_band / band - 1) * 100,
                          distance_to_band_after_15m=_distance(float(first.close), post_band, sign),
                          signal_stage="09:45", data_status="POST_OPEN")
            checks["post_open_expansion"] = _passes(
                record["post_open_bb_expansion_pct"],
                thresholds.maximum_post_open_bb_expansion_pct)
        else:
            checks["post_open_expansion"] = None
        five_min_available = pd.notna(record["target_angle_5m_deg"])
        if sign == 1:
            checks["five_min_band_slope"] = (_passes(
                record["five_min_band_slope_pct_per_min"], MAX_5MIN_BAND_SLOPE_PCT_PER_MIN)
                if five_min_available else None)
        checks["target_angle"] = (_passes(record["target_angle_5m_deg"],
                                          MAX_UPPER_ANGLE_DEG)
                                  if five_min_available else None)
        if five_min_available:
            # Desde 09:35 el ángulo de la banda objetivo sustituye la pendiente
            # previa en ambos sentidos. candidate_at_open conserva su evaluación.
            checks.pop("band_slope")
            record["candidate"] = bool(distance > 0 and all(
                checks[key] is not False for key in
                ("gap", "band_width", "band_growth", "distance", "target_angle"))
                and (sign != 1 or checks["five_min_band_slope"] is not False)
                and checks["post_open_expansion"] is not False)
        else:
            record["candidate"] = bool(
                record["candidate_at_open"] and checks["post_open_expansion"] is not False)
            if first is not None:
                record["candidate"] = False  # falta el ángulo de 09:35
        record["failed_conditions"] = ",".join(
            key for key, passed in checks.items() if passed is False)
        record["pending_conditions"] = ",".join(
            key for key, passed in checks.items() if passed is None and
            (key in ("five_min_band_slope", "target_angle", "target_angle_10s",
                     "target_angle_cutoff") or getattr(thresholds, {
                "gap": "minimum_gap_pct", "band_width": "maximum_prior_bb_width_pct",
                "band_slope": "maximum_target_band_slope_pct",
                "band_growth": "maximum_prior_bb_width_growth",
                "distance": "minimum_distance_to_band_pct",
                "first_minute_confirmation": "minimum_first_minute_reversal_pct",
                "post_open_expansion": "maximum_post_open_bb_expansion_pct",
            }[key]) is not None))
        record["score_open"] = _score(record, "open", record)
        record["score_1m"] = _score(record, "1m", record) if minute is not None else np.nan
        record["score_5m"] = _score(record, "5m", record) if five_min_available else np.nan
        record["score_15m"] = _score(record, "15m", record) if first is not None else np.nan
        record["score"] = (record["score_15m"] if first is not None else
                           record["score_5m"] if five_min_available else
                           record["score_1m"] if minute is not None else record["score_open"])
        signal_bar = int(first.bar_index) if first is not None else int(prior.iloc[-1].bar_index) + 1
        record["confirmation_bar_index"] = signal_bar
        complete_early_window = set(early.time.astype(str)) >= {
            f"09:{minute_number:02d}" for minute_number in range(31, 45)}
        if record["confirmed"] == 1 and distance > 0:
            for m in early.itertuples():
                if not ("09:31" <= str(m.time) <= "09:44"):
                    continue
                hit = (float(m.low) <= band if sign == 1 else float(m.high) >= band)
                if hit:
                    close_time = (datetime.strptime(str(m.time), "%H:%M") +
                                  timedelta(minutes=1)).strftime("%H:%M")
                    minutes = (datetime.strptime(close_time, "%H:%M") -
                               datetime.strptime("09:31", "%H:%M")).seconds // 60
                    record.update(touched_target_band=True, touch_time=close_time,
                                  bars_until_touch=1, minutes_until_touch=minutes,
                                  target_band_price_at_touch=band,
                                  touch_status="OBSERVED_1M")
                    break
        if not current.empty and distance > 0:
            # Con confirmación 09:31 se omite todo el rango 09:30–09:45: parte
            # de ese rango transcurrió antes de la confirmación.
            minimum = signal_bar if record["confirmed"] == 1 else signal_bar - 1
            events = (reentry_events(bars, {ticker: minimum}, {ticker: target})
                      if not record["touched_target_band"] else pd.DataFrame())
            if not events.empty:
                event = events.iloc[0]
                record.update(touched_target_band=True, touch_time=event.touch_time,
                              bars_until_touch=event.bars_until_touch,
                              minutes_until_touch=event.minutes_until_touch,
                              target_band_price_at_touch=event.target_band_price_at_touch,
                              touch_status=("OBSERVED_15M" if record["confirmed"] != 1 or
                                            complete_early_window else
                                            "FIRST_OBSERVED_EARLY_UNKNOWN"))
            elif not record["touched_target_band"]:
                record["touch_status"] = (
                    "EARLY_WINDOW_UNOBSERVED" if record["confirmed"] == 1 and
                    not complete_early_window else "NOT_TOUCHED")
        elif distance <= 0:
            record["touch_status"] = "NOT_APPLICABLE_INSIDE_BAND"
        if record["touched_target_band"]:
            reversion = sign * (opening - record["target_band_price_at_touch"]) / opening * 100
            record["reversion_to_touch_band_pct"] = reversion
            record["band_at_or_beyond_open_at_touch"] = bool(reversion <= 0)
        record.pop("direction_sign")
        found.append(record)
    result = pd.DataFrame.from_records(found, columns=SETUP_COLUMNS)
    if mode == "FILTERED":
        result = result.loc[result.candidate == True]
    return result.sort_values(["score", "ticker"], ascending=[False, True],
                              na_position="last").reset_index(drop=True)


def reentry_events(candles: pd.DataFrame,
                   confirmation_bars: dict[str, int] | None = None,
                   target_bands: dict[str, str] | None = None) -> pd.DataFrame:
    """Primer rango que toca la banda conocida al inicio de esa vela.

    El evento se conoce al cierre de la vela de 15 minutos. Si la señal fue
    confirmada a las 09:31, se omite la vela 09:30–09:45 completa.
    """
    events = []
    if candles.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    for ticker, raw in candles.groupby("ticker", sort=True):
        minimum_bar = confirmation_bars.get(ticker) if confirmation_bars is not None else None
        if confirmation_bars is not None and minimum_bar is None:
            continue
        series = bollinger_bands(_ordered_bars(raw))
        current = series.loc[series.session == "ACTUAL"]
        if current.empty:
            continue
        target = (target_bands or {}).get(ticker)
        if target is None:
            first_open = float(current.iloc[0].open)
            target = "UPPER" if first_open >= float(series.loc[series.session == "PREVIA"].iloc[-1].close) else "LOWER"
        opening_bar = int(current.iloc[0].bar_index)
        for _, bar in current.iterrows():
            bar_index = int(bar.bar_index)
            if minimum_bar is not None and bar_index <= minimum_bar:
                continue
            previous = series.loc[series.bar_index < bar_index].iloc[-1]
            known_band = float(previous.bb_upper if target == "UPPER" else previous.bb_lower)
            if not np.isfinite(known_band):
                continue
            touched = (float(bar.low) <= known_band if target == "UPPER"
                       else float(bar.high) >= known_band)
            if touched:
                close_time = candle_close_time(str(bar.time))
                minutes = (datetime.strptime(close_time, "%H:%M") -
                           datetime.strptime("09:30", "%H:%M")).seconds // 60
                if minimum_bar == opening_bar:
                    minutes -= 1  # la confirmación de 1 min se conoció a las 09:31
                kind = "REINGRESO_BANDA_SUPERIOR" if target == "UPPER" else "REINGRESO_BANDA_INFERIOR"
                events.append(dict(ticker=ticker, bar_index=bar_index, time=close_time,
                                   price=known_band, kind=kind, target_band=target,
                                   touched_target_band=True, touch_time=close_time,
                                   bars_until_touch=bar_index - opening_bar + 1,
                                   minutes_until_touch=minutes,
                                   target_band_price_at_touch=known_band))
                break
    return pd.DataFrame.from_records(events, columns=EVENT_COLUMNS)
