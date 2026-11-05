"""Reglas exploratorias para las dos vistas de Historial prueba."""

import numpy as np
import pandas as pd


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
