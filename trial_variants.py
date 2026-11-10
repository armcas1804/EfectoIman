"""Reglas exploratorias para las dos vistas de Historial prueba."""

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


LIVE_COLUMNS = ("ticker", "window_seconds", "source", "first_price", "price",
                "first_time", "last_time", "event_count")

# Puntos experimentales del score de Home prueba B. No alteran la selección B.
HOME_B_MAX_PRIOR_WIDTH_PCT = 1.5
HOME_B_DESIRED_GAP_PCT = 2.0
HOME_B_DESIRED_EXTERIOR_DISTANCE_PCT = 1.0
HOME_B_PRIOR_SLOPE_FULL_PENALTY_PCT = 0.15  # % por vela de 15 minutos
HOME_B_LIVE_BAND_MOVE_FULL_PENALTY_PCT = 0.20  # cambio real durante el corte
HOME_B_PRIOR_WIDTH_POINTS_PER_PP = 15.0
HOME_B_PRIOR_WIDTH_MAX_PENALTY = 15.0
HOME_B_PRIOR_GROWTH_POINTS_PER_PP = 50.0
HOME_B_PRIOR_GROWTH_MAX_PENALTY = 5.0
HOME_B_LIVE_EXPANSION_POINTS_PER_PCT = 0.2
HOME_B_LIVE_EXPANSION_MAX_PENALTY = 10.0


def live_quote_observations(quotes: pd.DataFrame, opened_at: datetime,
                            seconds: int) -> pd.DataFrame:
    """Primera y última cotización ya recibidas para un corte de apertura."""
    needed = {"ticker", "price", "quote_event_time", "quote_received_at"}
    if quotes.empty or not needed.issubset(quotes.columns):
        return pd.DataFrame(columns=LIVE_COLUMNS)
    rows = quotes.copy()
    rows["received"] = pd.to_datetime(rows.quote_received_at, utc=True,
                                       errors="coerce").dt.tz_convert("America/New_York")
    rows["event"] = pd.to_datetime(rows.quote_event_time, utc=True,
                                    errors="coerce").dt.tz_convert("America/New_York")
    rows["price"] = pd.to_numeric(rows.price, errors="coerce")
    opened = pd.Timestamp(opened_at).tz_convert("America/New_York")
    deadline = opened + timedelta(seconds=seconds)
    rows = rows.loc[rows.received.between(opened, deadline) &
                    rows.event.ge(opened) & rows.price.gt(0)].copy()
    if rows.empty:
        return pd.DataFrame(columns=LIVE_COLUMNS)
    records = []
    for ticker, group in rows.groupby("ticker", sort=True):
        first = group.sort_values(["received", "event"]).iloc[0]
        last = group.sort_values(["event", "received"]).iloc[-1]
        records.append({
            "ticker": ticker, "window_seconds": seconds,
            "source": "yahoo_websocket_local", "first_price": float(first.price),
            "price": float(last.price), "first_time": first.event.isoformat(),
            "last_time": last.event.isoformat(), "event_count": len(group),
        })
    return pd.DataFrame.from_records(records, columns=LIVE_COLUMNS)


def read_live_cut_observations(day: str, seconds: int,
                               snapshot_dir: Path) -> pd.DataFrame:
    """Recupera un corte ya congelado; filtra el flujo por hora de recepción."""
    saved = snapshot_dir / f"home_prueba_b_datos_{seconds}s_{day}.csv"
    if saved.is_file():
        rows = pd.read_csv(saved)
        if set(LIVE_COLUMNS).issubset(rows.columns):
            matching = rows.loc[rows.window_seconds.eq(seconds), list(LIVE_COLUMNS)]
            if not matching.empty:
                return matching
    opened = datetime.combine(datetime.fromisoformat(day).date(), time(9, 30),
                              ZoneInfo("America/New_York"))
    for available_seconds in (10, 30, 60, 90):
        if available_seconds < seconds:
            continue
        flow = snapshot_dir / f"apertura_flujo_{available_seconds}s_{day}.csv"
        if flow.is_file():
            return live_quote_observations(pd.read_csv(flow), opened, seconds)
    return pd.DataFrame(columns=LIVE_COLUMNS)


def add_low_volatility(model_history: pd.DataFrame, setups_history: pd.DataFrame,
                       max_width_pct: float = 1.5, relative_quantile: float = .25,
                       lookback: int = 20, min_prior: int = 10) -> pd.DataFrame:
    """Compara el ancho BB previo con el propio símbolo en fechas anteriores.

    El ancho ya pertenece a BB(20,2) de 15 minutos. El cuantil se desplaza
    una sesión para que nunca use el ancho de la fecha evaluada ni fechas futuras.
    """
    result = model_history.copy()
    history = setups_history[["session_date", "ticker", "prior_bb_width_pct"]].copy()
    history["prior_bb_width_pct"] = pd.to_numeric(
        history.prior_bb_width_pct, errors="coerce")
    history = (history.drop_duplicates(["session_date", "ticker"])
               .sort_values(["ticker", "session_date"]))
    history["trial_width_reference_pct"] = history.groupby("ticker")[
        "prior_bb_width_pct"].transform(
            lambda values: values.shift(1).rolling(
                lookback, min_periods=min_prior).quantile(relative_quantile))
    result = result.merge(
        history[["session_date", "ticker", "trial_width_reference_pct"]],
        on=["session_date", "ticker"], how="left", validate="one_to_one")
    width = pd.to_numeric(result.prior_width_pct, errors="coerce")
    reference = result.trial_width_reference_pct
    result["trial_low_volatility"] = (width.notna() & width.le(max_width_pct) &
                                      (reference.isna() | width.le(reference)))
    result["trial_width_limit_pct"] = np.where(
        reference.notna(), np.minimum(reference, max_width_pct), max_width_pct)
    result["trial_width_status"] = np.select(
        [width.isna(), width.gt(max_width_pct),
         reference.notna() & width.gt(reference), reference.isna()],
        ["Sin ancho BB previo", "Ancho BB supera el máximo absoluto",
         "Ancho BB supera su rango bajo previo",
         "Bajo el máximo absoluto; sin 10 sesiones previas"],
        default="Baja volatilidad previa")
    result["trial_candidate_model_20s"] = result.trial_candidate_20s.fillna(False)
    result["trial_candidate_20s"] = (result.trial_candidate_model_20s &
                                      result.trial_low_volatility)
    rejected = result.trial_candidate_model_20s & ~result.trial_low_volatility
    result.loc[rejected, "trial_signal_status_20s"] = result.loc[
        rejected, "trial_width_status"]
    result.loc[rejected, "trial_failed_conditions_20s"] = result.loc[
        rejected, "trial_width_status"]
    return result


def opening_reversal_table(setups: pd.DataFrame, observations: pd.DataFrame,
                           seconds: int, min_gap_pct: float = 1.0) -> pd.DataFrame:
    """Gap del primer precio y giro hacia el cierre previo al corte solicitado."""
    base = setups[["ticker", "prior_close"]].drop_duplicates("ticker").copy()
    base = base.rename(columns={"prior_close": "previous_close"})
    cut = observations.loc[observations.window_seconds.eq(seconds), [
        "ticker", "source", "first_price", "price", "first_time", "last_time",
        "event_count"]].drop_duplicates("ticker")
    cut = cut.rename(columns={"price": "price_at_cutoff"})
    result = base.merge(cut, on="ticker", how="left", validate="one_to_one")
    previous = pd.to_numeric(result.previous_close, errors="coerce")
    first = pd.to_numeric(result.first_price, errors="coerce")
    current = pd.to_numeric(result.price_at_cutoff, errors="coerce")
    result["gap_at_open_pct"] = (first / previous - 1) * 100
    result["gap_direction"] = np.select(
        [result.gap_at_open_pct.gt(0), result.gap_at_open_pct.lt(0)],
        ["UP", "DOWN"], default="SIN GAP")
    sign = np.sign(result.gap_at_open_pct)
    result["reversal_pct"] = sign * (first - current) / first * 100
    valid = previous.gt(0) & first.gt(0) & current.gt(0)
    result["candidate"] = (valid & result.gap_at_open_pct.abs().ge(min_gap_pct) &
                           result.reversal_pct.gt(0))
    result["status"] = np.select(
        [~valid, result.gap_at_open_pct.abs().lt(min_gap_pct),
         result.reversal_pct.le(0)],
        [f"Sin operación válida a {seconds} s", "Gap insuficiente",
         "Sin giro hacia el cierre previo"], default="Gap y giro observados")
    result["window_seconds"] = seconds
    return result.sort_values(
        ["candidate", "gap_at_open_pct", "reversal_pct"],
        ascending=[False, False, False], key=lambda column: (
            column.abs() if column.name == "gap_at_open_pct" else column),
        na_position="last").reset_index(drop=True)


def score_home_trial_b(setups: pd.DataFrame, reversals: pd.DataFrame) -> pd.DataFrame:
    """Puntúa exterioridad, lateralidad y baja volatilidad sin cambiar la regla B.

    El ancho provisional usa BB(20,2) de 15 minutos calculada por el scanner
    sólo con el precio disponible en el corte. La distancia se exige tanto al
    primer precio como al corte; sin bandas o pendiente válidas no hay score.
    """
    columns = ["ticker", "prior_bb_width_pct", "prior_bb_width_change",
               "prior_bb_upper", "prior_bb_lower", "prior_target_band_slope_pct",
               "cutoff_bb_upper", "cutoff_bb_lower", "target_angle_cutoff_deg",
               "angle_cutoff_event_span_seconds"]
    bands = setups.reindex(columns=columns).drop_duplicates("ticker")
    result = reversals.drop(columns=[column for column in columns[1:]
                                     if column in reversals.columns]).merge(
        bands, on="ticker", how="left", validate="one_to_one")
    prior = pd.to_numeric(result.prior_bb_width_pct, errors="coerce")
    growth = pd.to_numeric(result.prior_bb_width_change, errors="coerce")
    upper = pd.to_numeric(result.cutoff_bb_upper, errors="coerce")
    lower = pd.to_numeric(result.cutoff_bb_lower, errors="coerce")
    prior_upper = pd.to_numeric(result.prior_bb_upper, errors="coerce")
    prior_lower = pd.to_numeric(result.prior_bb_lower, errors="coerce")
    first = pd.to_numeric(result.first_price, errors="coerce")
    current = pd.to_numeric(result.price_at_cutoff, errors="coerce")
    upward = result.gap_direction.eq("UP")
    result["distance_first_outside_bb_pct"] = np.where(
        upward, (first / prior_upper - 1) * 100,
        (prior_lower / first - 1) * 100)
    result["distance_cut_outside_bb_pct"] = np.where(
        upward, (current / upper - 1) * 100,
        (lower / current - 1) * 100)
    first_distance = result.distance_first_outside_bb_pct
    cut_distance = result.distance_cut_outside_bb_pct
    result["home_b_candidate"] = (result.candidate & first_distance.gt(0) &
                                  cut_distance.gt(0))
    result["home_b_status"] = np.select(
        [~result.candidate, first_distance.isna() | cut_distance.isna(),
         first_distance.le(0), cut_distance.le(0)],
        ["No cumple gap y giro", "Sin bandas para medir distancia",
         "Primera operación dentro de Bollinger",
         "Precio al corte dentro de Bollinger"],
        default="Gap, giro y precio fuera de Bollinger")
    middle = (upper + lower) / 2
    current_width = (upper - lower) / middle * 100
    expansion = (current_width / prior - 1) * 100
    result["cutoff_bb_width_pct"] = current_width
    result["bb_width_expansion_pct"] = expansion
    # El ángulo del scanner está normalizado a cinco minutos. Deshacemos esa
    # normalización para medir el movimiento real de banda en este corte.
    angle = pd.to_numeric(result.target_angle_cutoff_deg, errors="coerce")
    span = pd.to_numeric(result.angle_cutoff_event_span_seconds, errors="coerce")
    result["live_band_move_pct"] = np.tan(np.deg2rad(angle)) * span / 300
    prior_slope = pd.to_numeric(result.prior_target_band_slope_pct,
                                errors="coerce")
    result["penalty_gap"] = (
        (1 - result.gap_at_open_pct.abs() / HOME_B_DESIRED_GAP_PCT)
        .clip(lower=0, upper=1) * 10)
    result["penalty_exterior_distance"] = (
        (1 - cut_distance / HOME_B_DESIRED_EXTERIOR_DISTANCE_PCT)
        .clip(lower=0, upper=1) * 30)
    result["penalty_prior_lateral"] = (
        prior_slope.abs() / HOME_B_PRIOR_SLOPE_FULL_PENALTY_PCT
        * 20).clip(upper=20)
    result["penalty_live_lateral"] = (
        result.live_band_move_pct.abs() /
        HOME_B_LIVE_BAND_MOVE_FULL_PENALTY_PCT * 20).clip(upper=20)
    result["penalty_prior_width"] = (
        (prior - HOME_B_MAX_PRIOR_WIDTH_PCT).clip(lower=0)
        * HOME_B_PRIOR_WIDTH_POINTS_PER_PP).clip(upper=HOME_B_PRIOR_WIDTH_MAX_PENALTY)
    result["penalty_prior_growth"] = (
        growth.clip(lower=0) * HOME_B_PRIOR_GROWTH_POINTS_PER_PP
    ).clip(upper=HOME_B_PRIOR_GROWTH_MAX_PENALTY)
    result["penalty_live_expansion"] = (
        expansion.clip(lower=0) * HOME_B_LIVE_EXPANSION_POINTS_PER_PCT
    ).clip(upper=HOME_B_LIVE_EXPANSION_MAX_PENALTY)
    valid_bands = (prior.gt(0) & growth.notna() & prior_upper.gt(prior_lower) &
                   upper.gt(lower) & middle.gt(0) & prior_slope.notna() &
                   span.gt(0) & np.isfinite(current_width) &
                   np.isfinite(expansion) & np.isfinite(result.live_band_move_pct))
    total_penalty = result[["penalty_gap", "penalty_exterior_distance",
                            "penalty_prior_lateral",
                            "penalty_live_lateral", "penalty_prior_width",
                            "penalty_prior_growth", "penalty_live_expansion"]].sum(axis=1)
    result["score_home_b"] = np.where(
        result.home_b_candidate & valid_bands,
        (100 - total_penalty).clip(lower=0).round(1), np.nan)
    return result
