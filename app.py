"""Operativa diaria e historial diagnóstico local de Efecto Imán."""

import io
import importlib
import math
import time
from pathlib import Path
from html import escape
import zipfile
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import real_data
from config import DATA_DIR, HISTORY_SESSION_LIMIT, TICKERS
from daily_chart import build_daily_chart
from home_chart import build_home_chart
from daily_store import (export_history, initialize_daily_store,
                         load_daily_session, save_daily_session, session_dates)
from outcomes import build_outcomes
from opening_live import CaptureProgress, DISPLAY_WINDOWS, OpeningCapture, next_capture_day
from market_refresh import MarketRefresh
from manual_review import (LABELS, build_early_chart, early_features,
                           early_trajectory,
                           early_observations, load_reviews, save_review)
from signal_audit import first_availability
from scanner_settings import DEFAULT_THRESHOLDS
from subminute_history import ten_second_coverage


NY = ZoneInfo("America/New_York")
HERMOSILLO = ZoneInfo("America/Hermosillo")
MAX_CHARTS = 4
CAPTURE_CACHE_VERSION = 7
OPENING_WINDOW_COLUMNS = ["ticker", "score_open", "gap_pct", "gap_direction",
                          "opening_price", "cutoff_bb_upper", "cutoff_bb_lower",
                          "distance_to_band_pct", "target_angle_cutoff_deg",
                          "score_cutoff_expansion_component",
                          "angle_cutoff_event_span_seconds", "candidate_at_open",
                          "failed_conditions", "pending_conditions"]
DISPLAY_COLUMNS = [
    "ticker", "score", "score_open", "score_1m", "score_5m", "score_15m",
    "gap_pct", "gap_direction", "opening_price", "prior_close",
    "target_band", "distance_to_band_pct", "prior_distance_to_band_pct",
    "cutoff_bb_upper", "cutoff_bb_lower", "cutoff_target_band",
    "cutoff_target_band_reaction_pct", "target_angle_10s_deg",
    "angle_10s_event_span_seconds", "z_open", "prior_bb_width_pct",
    "target_angle_cutoff_deg", "angle_cutoff_event_span_seconds", "cutoff_seconds",
    "prior_target_band_slope_pct", "prior_bb_width_change", "prior_bb_width_growth_pct",
    "five_min_band_open", "five_min_band_end", "five_min_band_slope_pct_per_min",
    "upper_angle_0930_0931", "upper_angle_0930_0935",
    "target_angle_deg", "target_angle_1m_deg", "target_angle_5m_deg",
    "target_angle_0930_0935", "target_angle_0935_0940",
    "target_angle_0940_0945", "target_angle_score",
    "score_gap_component", "score_distance_component",
    "score_compression_component", "score_prior_width_growth_component",
    "score_target_angle_component", "score_first_minute_component",
    "score_post_open_expansion_component", "score_cutoff_expansion_component",
    "first_minute_return_pct", "first_minute_body_pct", "first_minute_reversal_pct",
    "post_open_bb_width_pct", "post_open_bb_width_change", "post_open_bb_expansion_pct",
    "post_open_target_band_slope_pct", "distance_to_band_after_15m",
    "touched_target_band", "touch_time", "minutes_until_touch", "touch_status",
    "reversion_to_touch_band_pct", "band_at_or_beyond_open_at_touch",
    "candidate_at_open", "candidate", "confirmed",
    "failed_conditions", "pending_conditions", "data_status",
]
st.set_page_config(page_title="Efecto Imán", layout="wide")


def style() -> None:
    st.markdown("""<style>
    html, body, [class*="css"], [data-testid="stApp"] {font-family: Georgia, 'Times New Roman', serif;}
    [data-testid="stAppViewContainer"] {background:#efefed;color:#242424;}
    [data-testid="stSidebar"] {background:#dededb;border-right:1px solid #aaaaa5;}
    [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
      background:#f7f7f5;border:1px solid #bdbdb7;border-radius:2px;padding:8px;}
    h1,h2,h3 {font-family:Georgia,'Times New Roman',serif;font-weight:400;letter-spacing:.02em;}
    .eyebrow {font:12px Arial,sans-serif;letter-spacing:.15em;color:#676762;text-transform:uppercase;}
    .rule {border-bottom:1px solid #aaa9a3;margin:10px 0 20px 0;}
    .alert {border:1px solid #ac2525;background:#f2dddd;color:#8e2020;padding:9px 12px;}
    .clock-card {border:1px solid #bdbdb7;background:#f7f7f5;padding:10px 14px;}
    .clock-card.clock-open {border-color:#4d765b;background:#dcebe0;color:#23452d;}
    .clock-label {display:block;font:11px Arial,sans-serif;letter-spacing:.13em;text-transform:uppercase;}
    .clock-time {display:block;font:28px Georgia,serif;line-height:1.25;font-variant-numeric:tabular-nums;}
    .clock-note {display:block;font:11px Arial,sans-serif;min-height:15px;}
    .capture-banner {border:1px solid #8d815d;background:#f5f0df;padding:10px 14px;
                     margin:9px 0;font:13px Arial,sans-serif;color:#49412c;}
    .capture-banner.ready {border-color:#4d765b;background:#e2ece4;color:#23452d;}
    .capture-banner.error {border-color:#ac2525;background:#f2dddd;color:#8e2020;}
    .capture-banner.active::before {content:'';display:inline-block;width:8px;height:8px;
      border-radius:50%;background:#a77416;margin-right:9px;animation:capture-pulse 1s infinite;}
    @keyframes capture-pulse {50% {opacity:.2;}}
    div.stButton > button {border:1px solid #777;background:#f7f7f5;color:#242424;border-radius:2px;}
    </style>""", unsafe_allow_html=True)


def chart_context(value: str) -> None:
    if st.session_state.get("chart_context") != value:
        st.session_state.chart_context = value
        st.session_state.selected_charts = []


@st.cache_resource
def opening_capture(day: str, scheduler_version: int) -> OpeningCapture:
    capture = OpeningCapture(date.fromisoformat(day), TICKERS)
    capture.start()
    return capture


@st.cache_resource
def market_refresh(day: str, refresh_version: int) -> MarketRefresh:
    return MarketRefresh(day, TICKERS)


def capture_message(now: datetime, capture: OpeningCapture,
                    progress: CaptureProgress) -> tuple[str, str]:
    """Mensaje visible basado en el estado observado, no sólo en el reloj."""
    total = len(capture.tickers)
    prep_hmo = capture.prepare_at.astimezone(HERMOSILLO).strftime("%H:%M")
    open_hmo = capture.opened_at.astimezone(HERMOSILLO).strftime("%H:%M:%S")
    cut_hmo = capture.cutoff.astimezone(HERMOSILLO).strftime("%H:%M:%S")
    thirty = capture.opened_at.replace(second=30)
    minute = capture.opened_at.replace(minute=31, second=0)
    minute_and_half = capture.opened_at.replace(minute=31, second=30)
    ready_windows = progress.window_ready_at or {}
    window_counts = progress.window_counts or {}
    window_candidates = progress.window_candidates or {}
    if now.weekday() >= 5:
        return "", "Hoy no hay apertura programada. Abre Operativa diaria antes de las 09:25 ET del próximo día hábil."
    if 90 in ready_windows:
        ready = ready_windows[90].astimezone(NY)
        if not window_counts.get(90, 0):
            return "error", "Corte de 1 minuto y medio completado sin cotizaciones válidas."
        return "ready", (f"Lista de 1 minuto y medio lista a las {ready:%H:%M:%S} ET: "
                         f"{window_counts.get(90, 0)}/{total} cotizaciones; "
                         f"{window_candidates.get(90, 0)} candidatos. "
                         "Se guardaron por separado los cortes de 30, 60 y 90 s.")
    if 60 in ready_windows:
        ready = ready_windows[60].astimezone(NY)
        if not window_counts.get(60, 0):
            return "error", "Corte de 1 minuto completado sin cotizaciones válidas."
        remaining = max(0, math.ceil((minute_and_half - now).total_seconds()))
        next_status = (f"Actualización de 1 minuto y medio en {remaining} s."
                       if remaining else "Calculando la actualización de 1 minuto y medio.")
        return "active", (f"Lista de 1 minuto lista a las {ready:%H:%M:%S} ET: "
                          f"{window_counts.get(60, 0)}/{total} cotizaciones; "
                          f"{window_candidates.get(60, 0)} candidatos. " + next_status)
    if 30 in ready_windows:
        ready = ready_windows[30].astimezone(NY)
        remaining = max(0, math.ceil((minute - now).total_seconds()))
        next_status = (f"Actualización de 1 minuto en {remaining} s." if remaining else
                       "Calculando la actualización de 1 minuto.")
        return "active", (f"Lista de 30 s lista a las {ready:%H:%M:%S} ET: "
                          f"{window_counts.get(30, 0)}/{total} cotizaciones; "
                          f"{window_candidates.get(30, 0)} candidatos. " +
                          next_status)
    if progress.ready_at is not None and now < thirty:
        if not progress.frozen_count:
            return "error", (f"Corte de {cut_hmo} HMO / 09:30:10 ET completado: 0/{total} "
                             "cotizaciones recibidas. No hay diagnóstico temprano.")
        ready = progress.ready_at.astimezone(NY)
        return "active", (f"Corte de 10 s conservado para auditoría. "
                          f"Preparando la lista visible de 30 s; "
                          f"cotizaciones iniciales: {progress.frozen_count}/{total}.")
    if progress.status.startswith(("Corte de diez segundos no capturado", "No se pudo calcular")):
        return "error", f"El corte automático no produjo diagnóstico: {progress.status}."
    if progress.prior_error:
        return "error", "No se prepararon las bandas previas; el corte de diez segundos no podrá clasificarse."
    if now < capture.prepare_at:
        return "", (f"Programado: preparación {prep_hmo} HMO / 09:25 ET; "
                    f"captura {open_hmo}–{cut_hmo} HMO.")
    if now < capture.opened_at:
        if progress.prior_count and progress.stream_connected:
            return "ready", (f"Preparación lista: bandas previas de {progress.prior_count}/{total} "
                             "símbolos y flujo conectado. Esperando la apertura.")
        stream = "conectado" if progress.stream_connected else "conectando"
        return "active", (f"Preparando apertura: bandas previas de {progress.prior_count}/{total} "
                          f"símbolos; flujo {stream}.")
    if now < thirty:
        remaining = max(0, math.ceil((thirty - now).total_seconds()))
        stream = "flujo conectado" if progress.stream_connected else "reconectando flujo"
        return "active", (f"Capturando apertura: {progress.quote_count}/{total} símbolos "
                          f"con cotización; faltan {remaining} s para la primera lista. {stream}.")
    if now < minute:
        return "active", ("Corte de 30 s alcanzado. Calculando gap y Bollinger "
                          "provisional; el flujo sigue hasta 09:31 ET.")
    if progress.status == "Los cortes de apertura no se capturaron en este proceso":
        return "error", "La aplicación se inició después de los cortes de apertura."
    if progress.prior_error:
        return "error", "No hay bandas previas; no se pudieron calcular los cortes de apertura."
    if now < minute_and_half:
        return "active", ("Corte de 1 minuto alcanzado. Calculando gap y Bollinger "
                          "provisional; el flujo sigue hasta 09:31:30 ET.")
    return "active", "Corte de 1 minuto y medio alcanzado. Calculando la tercera lista."


@st.fragment(run_every="1s", parallel=True)
def market_clock(page: str) -> None:
    now_ny = datetime.now(NY)
    now_hmo = datetime.now(HERMOSILLO)
    opening_minute = now_ny.weekday() < 5 and (now_ny.hour, now_ny.minute) == (9, 30)
    left, right = st.columns(2)
    left.markdown(
        f"<div class='clock-card{' clock-open' if opening_minute else ''}'>"
        f"<span class='clock-label'>Nueva York</span>"
        f"<span class='clock-time'>{now_ny:%H:%M:%S}</span>"
        f"<span class='clock-note'>{'Mercado abierto · primer minuto' if opening_minute else '&nbsp;'}</span>"
        "</div>", unsafe_allow_html=True)
    right.markdown(
        f"<div class='clock-card'><span class='clock-label'>Hermosillo</span>"
        f"<span class='clock-time'>{now_hmo:%H:%M:%S}</span>"
        "<span class='clock-note'>&nbsp;</span></div>", unsafe_allow_html=True)
    capture = opening_capture(now_ny.date().isoformat(), CAPTURE_CACHE_VERSION)
    scheduled_day = next_capture_day(now_ny)
    if scheduled_day != now_ny.date():
        opening_capture(scheduled_day.isoformat(), CAPTURE_CACHE_VERSION)
    capture.poll()
    banner_class, message = capture_message(now_ny, capture, capture.progress())
    if now_ny.weekday() >= 5:
        message += f" Próximo corte programado: {scheduled_day:%d/%m} a las 09:30:10 ET."
    st.markdown(f"<div class='capture-banner {banner_class}' role='status'>{escape(message)}</div>",
                unsafe_allow_html=True)
    if page == "Operativa diaria":
        refresher = market_refresh(now_ny.date().isoformat(), 2)
        if (now_ny.weekday() < 5 and
                (9, 31) <= (now_ny.hour, now_ny.minute) <= (16, 15)):
            refresher.start()
        refresh_status, _, refresh_version = refresher.snapshot()
        _, _, _, version = capture.snapshot()
        if version and st.session_state.get("opening_capture_version") != (
                now_ny.date().isoformat(), version):
            st.session_state.opening_capture_version = (now_ny.date().isoformat(), version)
            st.rerun()
        if refresh_version and st.session_state.get("market_refresh_version") != (
                now_ny.date().isoformat(), refresh_version):
            st.session_state.market_refresh_version = (now_ny.date().isoformat(), refresh_version)
            st.rerun()


def refresh_real(days: int = 1) -> tuple[int, int, tuple[str, ...], int]:
    provider = importlib.reload(real_data)
    today = datetime.now(NY).date().isoformat()
    bars, missing_15m = provider.download_bars(period="59d" if days > 1 else "10d", tickers=TICKERS)
    try:
        minute_bars, missing_1m = provider.download_bars(
            period="7d" if days > 1 else "1d", tickers=TICKERS, interval="1m")
    except Exception:
        minute_bars, missing_1m = {}, TICKERS
    five_minute_bars, missing_5m = (provider.download_bars(period="59d", tickers=TICKERS,
                                                  interval="5m") if days > 1 else ({}, ()))
    missing = tuple(f"15m {ticker}" for ticker in missing_15m) + tuple(
        f"1m {ticker}" for ticker in missing_1m) + tuple(
        f"5m {ticker}" for ticker in missing_5m)
    st.session_state.market_missing_tickers = missing
    available = sorted(set(provider.available_days(bars)) | set(provider.available_days(minute_bars)))
    dates = available[-days:] if days > 1 else ([today] if today in available else [])
    saved = 0
    for day in dates:
        setups, candles = provider.scan_real_day(bars, day, minute_bars, five_minute_bars)
        if not candles.empty:
            save_daily_session(day, setups, candles, replace_existing=days > 1)
            saved += 1
    st.session_state.real_last_tick = time.monotonic()
    return saved, len(dates), missing, len(bars)


def setup_list(setups: pd.DataFrame, key: str,
               availability: dict[str, tuple[str, str]] | None = None,
               preliminary: bool = False, trial: bool = False,
               trial_seconds: int = 10) -> None:
    if setups.empty:
        st.info("No hay candidatos con los límites exploratorios actuales.")
        return
    for row in setups.itertuples():
        cols = st.columns([1.5, 1, .6], vertical_alignment="center")
        cols[0].write(f"**{row.ticker}** · {row.gap_direction}")
        shown_score = row.score_open if preliminary else row.score
        shown_distance = row.trial_distance_10s_pct if trial else row.distance_to_band_pct
        score_name = (f"modelo a {trial_seconds} s" if trial else
                      "apertura" if preliminary else "actual")
        cols[0].caption(f"Score {score_name} {shown_score:.1f} · "
                        f"banda {row.target_band} · "
                        f"distancia {shown_distance:+.2f}%")
        if availability and row.ticker in availability:
            recorded, source = availability[row.ticker]
            cols[0].caption(f"Primer registro local {recorded[11:19]} ET · {source}")
        if preliminary and pd.notna(row.touched_target_band) and bool(row.touched_target_band):
            cols[0].caption("Ya tocó la banda; alerta preliminar vencida")
        if preliminary and pd.notna(row.target_angle_cutoff_deg):
            cols[0].caption(f"Ángulo dentro del corte {row.target_angle_cutoff_deg:+.1f}° · "
                            f"{row.angle_cutoff_event_span_seconds:.0f} s entre precios")
        if trial:
            angles = ([row.trial_angle_10s_deg]
                      if pd.notna(row.trial_angle_10s_deg) else [])
        else:
            angles = [value for value in (row.target_angle_5m_deg,
                      row.target_angle_0935_0940, row.target_angle_0940_0945)
                      if pd.notna(value)]
        if angles:
            detail = (f"Ángulo dentro de {trial_seconds} s {angles[0]:+.1f}°" if trial
                      else f"Ángulo inicial {angles[0]:+.1f}°")
            if len(angles) > 1:
                detail += f" · máximo 15m {max(angles):+.1f}°"
            if not trial and pd.notna(row.post_open_bb_expansion_pct):
                detail += f" · ancho {row.post_open_bb_expansion_pct:+.1f}%"
            cols[0].caption(detail)
        shown_gap = row.trial_gap_10s_pct if trial else row.gap_pct
        cols[1].write(f"{shown_gap:+.2f}%")
        if cols[2].button("+", key=f"{key}_{row.ticker}", help=f"Mostrar u ocultar {row.ticker}"):
            chosen = st.session_state.selected_charts
            if row.ticker in chosen:
                chosen.remove(row.ticker)
            elif len(chosen) < MAX_CHARTS:
                chosen.append(row.ticker)
            else:
                st.toast(f"Máximo {MAX_CHARTS} gráficas simultáneas")


def charts(setups: pd.DataFrame, candles: pd.DataFrame) -> None:
    chosen = [ticker for ticker in st.session_state.selected_charts if ticker in set(setups.ticker)]
    if not chosen:
        st.caption("Selecciona + en un símbolo para abrir su gráfica. Puedes comparar hasta cuatro.")
        return
    if candles.empty:
        st.info("La lista está guardada, pero faltan las velas previas para dibujar la gráfica.")
        return
    for ticker in chosen:
        setup = setups.loc[setups.ticker == ticker].iloc[0]
        marker_bar = (int(setup.confirmation_bar_index)
                      if setup.confirmed == 1 and pd.notna(setup.confirmation_bar_index) else -1)
        minute_close = (float(setup.first_minute_close)
                        if pd.notna(setup.first_minute_close) else float("nan"))
        cutoff_seconds = (int(setup.cutoff_seconds)
                          if pd.notna(setup.cutoff_seconds) else None)
        st.plotly_chart(build_daily_chart(
            candles, ticker, float(setup.gap_pct), marker_bar, minute_close,
            str(setup.target_band), setup.touch_time,
            float(setup.target_band_price_at_touch)
            if pd.notna(setup.target_band_price_at_touch) else float("nan"),
            cutoff_seconds=cutoff_seconds,
            cutoff_price=float(setup.opening_price)
            if pd.notna(setup.opening_price) else float("nan"),
            cutoff_upper=float(setup.cutoff_bb_upper)
            if pd.notna(setup.cutoff_bb_upper) else float("nan"),
            cutoff_lower=float(setup.cutoff_bb_lower)
            if pd.notna(setup.cutoff_bb_lower) else float("nan"),
            focus_opening=cutoff_seconds == 20),
            width="stretch", config={"displayModeBar": False})


def diagnostic_table(setups: pd.DataFrame, trial: bool = False,
                     trial_seconds: int = 10) -> None:
    if setups.empty:
        st.info("Aún no hay observaciones de esta sesión.")
        return
    columns = (['ticker', 'score', 'trial_signal_status',
                'trial_failed_conditions_10s', 'trial_candidate',
                'manual_label', 'trial_gap_10s_pct', 'trial_distance_10s_pct',
                'trial_angle_10s_deg', 'trial_score_actual_10s',
                'trial_score_final_original', 'early_data_status'] +
               (['model_training_example'] if trial_seconds == 20 else [])
               if trial else DISPLAY_COLUMNS)
    st.dataframe(setups.reindex(columns=columns), hide_index=True, width="stretch",
                 column_config={
                     "score": st.column_config.NumberColumn(
                         f"Score modelo {trial_seconds} s" if trial else "Score actual",
                         format="%.1f"),
                     "trial_score_actual_10s": f"Score heurístico {trial_seconds} s",
                     "trial_model_score_raw_10s": f"Índice bruto modelo {trial_seconds} s",
                     "trial_score_final_original": "Score posterior (referencia)",
                     "trial_score_difference_10s": f"Diferencia a {trial_seconds} s",
                     "trial_signal_status": f"Estado a {trial_seconds} s",
                     "trial_failed_conditions_10s": (
                         "Motivo sin señal 20 s" if trial_seconds == 20
                         else "Por qué no entra al scanner"),
                     "trial_gap_10s_pct": f"Gap a {trial_seconds} s",
                     "trial_distance_10s_pct": f"Distancia a {trial_seconds} s",
                     "trial_angle_10s_deg": f"Ángulo a {trial_seconds} s",
                     "trial_candidate": "Señal del modelo" if trial_seconds == 20 else
                                        "Candidato estricto",
                     "model_training_example": "Juicio usado para entrenar",
                     "score_open": st.column_config.NumberColumn("Score apertura preliminar",
                                                                   format="%.1f"),
                     "score_5m": st.column_config.NumberColumn("Score con ángulo 09:35",
                                                                 format="%.1f"),
                     "failed_conditions": "Condiciones fallidas",
                     "pending_conditions": "Pendientes",
                 })
    if trial:
        st.caption("La puntuación de esta tabla usa sólo precios y bandas disponibles "
                   f"hasta {trial_seconds} s. El score posterior se muestra como "
                   "referencia y no interviene en la selección del modelo.")
        with st.expander("Comparación técnica y todas las métricas"):
            technical = ['ticker', 'trial_model_score_raw_10s',
                         'trial_score_difference_10s'] + [
                             column for column in DISPLAY_COLUMNS if column != 'ticker']
            st.dataframe(setups.reindex(columns=technical), hide_index=True,
                         width="stretch")
            st.caption("El índice bruto del clasificador también existe para precios "
                       "dentro de Bollinger; no es una señal ni una probabilidad "
                       f"calibrada. El score heurístico de {trial_seconds} s vale cero "
                       "cuando el precio está dentro de la banda. Las métricas de 1, "
                       "5 y 15 minutos describen lo ocurrido después.")
    else:
        st.caption("En cada corte subminuto la distancia usa Bollinger provisional de 15 min; "
               "el ángulo dentro del corte sólo se valora si llegaron dos precios con horas distintas. "
               "A las 09:35 se valora el ángulo inicial; "
               "desde las 09:45 se valora el mayor ángulo de los tres tramos de cinco minutos "
               "y la expansión de Bollinger. Los valores vacíos aún no estaban disponibles.")


def session_view(setups: pd.DataFrame, candles: pd.DataFrame, key: str,
                 preliminary: bool = False,
                 availability: dict[str, tuple[str, str]] | None = None,
                 trial: bool = False, trial_score_list: bool = False,
                 trial_seconds: int = 10) -> None:
    observed = setups.loc[setups.opening_price.notna()] if not setups.empty else setups
    candidates = observed.loc[observed.candidate_at_open == True] if preliminary else (
        observed.loc[observed.candidate == True])
    if preliminary and not candidates.empty:
        candidates = candidates.sort_values("score_open", ascending=False)
    metrics = st.columns(4)
    metrics[0].metric("Símbolos con apertura", len(observed))
    metrics[1].metric("Señales para revisar" if trial_score_list else
                      "Preliminares" if preliminary else "Candidatos", len(candidates))
    metrics[2].metric("Toques de banda", int(observed.touched_target_band.fillna(False).sum())
                      if not observed.empty else 0)
    metrics[3].metric("Banda aún separada al toque", int(
        ((observed.touched_target_band == True) &
         (observed.reversion_to_touch_band_pct > 0)).sum()) if not observed.empty else 0)
    st.subheader("Diagnóstico de apertura")
    diagnostic_table(setups, trial, trial_seconds)
    st.caption("Un toque puede ocurrir porque la banda se desplaza hasta la apertura. "
               "La columna «band_at_or_beyond_open_at_touch» separa esos casos; "
               "un toque no demuestra rentabilidad.")
    st.subheader("Señales para revisar y gráficas" if trial_score_list else
                 "Candidatos y gráficas")
    if preliminary:
        st.caption("Hasta las 09:35 ET se muestra la lista preliminar de apertura, ordenada "
                   "por score de apertura. El ángulo posterior aún no confirma estos casos; "
                   "una entrada puede haber tocado la banda antes de que llegue una actualización.")
    left, right = st.columns([.95, 1.75], gap="medium")
    with left:
        setup_list(candidates, key, availability, preliminary, trial, trial_seconds)
    with right:
        charts(setups, candles)
    touched = candidates.loc[candidates.touched_target_band == True]
    if not touched.empty:
        st.markdown("<div class='alert'>Banda objetivo tocada: " +
                    ", ".join(touched.ticker) +
                    ". Observación histórica, no orden de salida.</div>",
                    unsafe_allow_html=True)


@st.fragment(run_every="30s")
def daily_panel() -> None:
    now = datetime.now(NY)
    day = now.date().isoformat()
    chart_context(f"daily:{day}")
    market_window = now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) <= (16, 15)
    first_minute_open = market_window and (now.hour, now.minute) == (9, 30)
    capture = opening_capture(day, CAPTURE_CACHE_VERSION)
    capture.poll()
    cut_setups, cut_candles, capture_status, _ = capture.snapshot()
    windows = {}
    for seconds in DISPLAY_WINDOWS:
        window_setups, window_candles, ready_at = capture.snapshot_at(seconds)
        if window_setups is None:
            path = DATA_DIR / "snapshots" / f"diagnostico_apertura_{seconds}s_{day}.csv"
            if path.exists():
                window_setups = pd.read_csv(path)
                window_candles = cut_candles
        if window_setups is not None:
            windows[seconds] = (window_setups, window_candles, ready_at)
    refresher = market_refresh(day, 2)
    if st.button("Actualizar datos de mercado", width="stretch", disabled=first_minute_open):
        refresher.start(force=True)
    if market_window and not first_minute_open:
        refresher.start()
    refresh_status, refresh_missing, _ = refresher.snapshot()
    setups, candles, updated = load_daily_session(day)
    st.markdown(f"<span class='eyebrow'>Operativa diaria · {now:%d/%m/%Y} · "
                "DATOS PÚBLICOS</span><div class='rule'></div>", unsafe_allow_html=True)
    cols = st.columns(2)
    cols[0].metric("Universo configurado", len(TICKERS))
    cols[1].metric("Última actualización", updated[11:19] if updated else "Pendiente")
    if capture_status == "Los cortes de apertura no se capturaron en este proceso":
        st.warning("La aplicación comenzó después de los cortes. Se muestran los archivos "
                   "guardados, si existen, y las velas obtenidas después.")
    else:
        st.caption(capture_status)
    st.caption(refresh_status)
    if refresh_missing:
        with st.expander(f"Velas no disponibles del proveedor ({len(refresh_missing)} consultas)"):
            st.write(", ".join(refresh_missing))
    if windows:
        st.subheader("Listas congeladas de apertura")
        for seconds in DISPLAY_WINDOWS:
            if seconds not in windows:
                continue
            window_setups, _, ready_at = windows[seconds]
            chosen = window_setups.loc[window_setups.candidate_at_open == True].sort_values(
                "score_open", ascending=False)
            label = {30: "30 segundos", 60: "1 minuto",
                     90: "1 minuto y medio"}[seconds]
            with st.expander(f"{label}: {len(chosen)} candidatos", expanded=False):
                cut_time = {30: "09:30:30", 60: "09:31:00",
                            90: "09:31:30"}[seconds]
                ready_text = (f"cálculo terminado a las {ready_at:%H:%M:%S} ET"
                              if ready_at else "cargado del archivo local")
                st.caption(f"Corte {cut_time} ET · {ready_text}")
                st.dataframe(chosen.reindex(columns=OPENING_WINDOW_COLUMNS),
                             hide_index=True, width="stretch")
                st.download_button(f"Descargar candidatos de {label}",
                                   chosen.to_csv(index=False),
                                   file_name=f"candidatos_apertura_{seconds}s_{day}.csv",
                                   mime="text/csv", key=f"download_open_{seconds}_{day}")
    else:
        st.caption("Las listas de 30 segundos, 1 minuto y 1 minuto y medio aparecerán cuando "
                   "se congelen sus respectivos cortes.")
    if cut_setups is not None:
        with st.expander("Corte de 10 segundos conservado para auditoría"):
            st.dataframe(cut_setups.reindex(columns=OPENING_WINDOW_COLUMNS),
                         hide_index=True, width="stretch")
    early = market_window and (now.hour, now.minute) < (9, 35)
    active_window = max(windows) if windows else None
    if early and active_window is not None:
        setups, prior_candles, _ = windows[active_window]
        candles = (prior_candles if prior_candles is not None else
                   candles.loc[candles.session == "PREVIA"])
        st.caption(f"Lista activa: corte de {active_window} segundos. BB(20,2) "
                   "de 15 minutos provisional calculada sólo con precios recibidos "
                   "hasta ese corte. El ángulo queda pendiente si no hubo dos horas "
                   "de mercado distintas.")
    elif early and now < capture.opened_at.replace(minute=31, second=0):
        st.info("Esperando el corte de 30 segundos y sus bandas previas. "
                "La lista de 10 segundos no se usa como lista activa.")
        return
    elif early:
        st.warning("No hay lista guardada de 30, 60 ni 90 s; se muestran "
                   "las velas posteriores disponibles.")
    if not updated and not windows:
        st.info("Los cortes de 30 segundos, 1 minuto y 1 minuto y medio aún no están disponibles. "
                "Mantén la aplicación abierta antes de las 09:25 ET.")
        return
    if not market_window:
        st.caption("Mercado fuera de horario. Se muestra la sesión local guardada.")
    availability = first_availability(day)
    st.caption("El primer registro local indica cuándo terminó un cálculo desde que se activó "
               "esta versión; no reconstruye señales anteriores ni confirma cuándo se vio la pantalla.")
    session_view(setups, candles, f"daily_{day}", early, availability)


def home_cut_changed() -> None:
    """Una elección anterior pausa el avance automático hasta volver al último corte."""
    available = st.session_state.get("home_available_cuts", ())
    if available:
        st.session_state.home_follow_latest = st.session_state.home_cut == max(available)


@st.cache_data(ttl=30, show_spinner=False)
def home_minute_data(ticker: str, day: str) -> pd.DataFrame:
    """Minutos recientes de una sola empresa para la vela 15m en formación."""
    try:
        bars, _ = real_data.download_bars(period="1d", tickers=(ticker,), interval="1m")
        return bars.get(ticker, pd.DataFrame())
    except Exception:
        return pd.DataFrame()


@st.fragment(run_every="10s")
def home_panel() -> None:
    now = datetime.now(NY)
    day = now.date().isoformat()
    capture = opening_capture(day, CAPTURE_CACHE_VERSION)
    capture.poll()
    _, prior_candles, capture_status, _ = capture.snapshot()
    windows = {}
    for seconds in DISPLAY_WINDOWS:
        setups, candles, ready_at = capture.snapshot_at(seconds)
        if setups is None:
            path = DATA_DIR / "snapshots" / f"diagnostico_apertura_{seconds}s_{day}.csv"
            if path.exists():
                setups = pd.read_csv(path)
                candles = prior_candles
        if setups is not None:
            windows[seconds] = (setups, candles, ready_at)

    st.markdown("<span class='eyebrow'>Home · apertura de mercado</span>"
                "<div class='rule'></div>", unsafe_allow_html=True)
    st.write("Explora las listas de apertura y sigue cada empresa en una gráfica de "
             "15 minutos con Bandas de Bollinger.")
    st.caption("Los resultados de cada corte quedan congelados. La gráfica incorpora "
               "velas nuevas cuando el proveedor las publica; el score del corte no cambia.")
    labels = {30: "30 segundos", 60: "1 minuto", 90: "1 minuto y medio"}
    cut_cards = st.columns(3)
    for seconds, card in zip(DISPLAY_WINDOWS, cut_cards):
        with card:
            st.metric(labels[seconds], "Disponible" if seconds in windows else "Pendiente")
    if not windows:
        st.info("Aún no hay listas de apertura para hoy. La primera se genera a las "
                "09:30:30 de Nueva York si el proceso estaba activo y recibió cotizaciones.")
        st.caption(capture_status)
        return

    available = tuple(windows)
    latest = max(available)
    if st.session_state.get("home_day") != day:
        st.session_state.home_day = day
        st.session_state.home_follow_latest = True
        st.session_state.home_cut = latest
    st.session_state.home_available_cuts = available
    if (st.session_state.get("home_follow_latest", True) or
            st.session_state.get("home_cut") not in available):
        st.session_state.home_cut = latest
        st.session_state.home_follow_latest = True
    st.radio("Lista de apertura", available, format_func=lambda value: labels[value],
             horizontal=True, key="home_cut", on_change=home_cut_changed)
    seconds = st.session_state.home_cut
    if st.session_state.home_follow_latest:
        st.caption("Siguiendo automáticamente el corte más reciente. Puedes volver a uno anterior.")
    else:
        st.caption("Viendo un corte anterior. Elige el más reciente para retomar el avance automático.")

    setups, frozen_candles, ready_at = windows[seconds]
    observed = setups.loc[setups.opening_price.notna()].copy()
    candidates = observed.loc[observed.candidate_at_open == True].sort_values(
        ["score_open", "ticker"], ascending=[False, True])
    st.markdown(f"### Lista de {labels[seconds]}")
    cut_clock = {30: "09:30:30", 60: "09:31:00", 90: "09:31:30"}[seconds]
    st.caption(f"{day} · corte {cut_clock} ET · "
               f"{len(observed)} símbolos con precio · {len(candidates)} candidatos"
               + (f" · calculado {ready_at:%H:%M:%S} ET" if ready_at else ""))
    if candidates.empty:
        st.info("Este corte no produjo candidatos. Revisa los diagnósticos completos abajo.")
    else:
        st.dataframe(candidates.reindex(columns=[
            "ticker", "score_open", "gap_pct", "gap_direction",
            "distance_to_band_pct", "target_angle_cutoff_deg", "opening_price"]),
            hide_index=True, width="stretch", column_config={
                "ticker": "Empresa", "score_open": "Score del corte",
                "gap_pct": st.column_config.NumberColumn("Gap %", format="%+.2f"),
                "gap_direction": "Dirección",
                "distance_to_band_pct": st.column_config.NumberColumn(
                    "Fuera de Bollinger %", format="%+.2f"),
                "target_angle_cutoff_deg": st.column_config.NumberColumn(
                    "Ángulo del corte", format="%+.1f°"),
                "opening_price": st.column_config.NumberColumn("Precio del corte", format="%.2f"),
            })
    with st.expander(f"Ver diagnóstico completo ({len(observed)} símbolos)"):
        st.dataframe(observed.reindex(columns=OPENING_WINDOW_COLUMNS),
                     hide_index=True, width="stretch")

    choices = candidates.ticker.tolist()
    if not choices:
        return
    selected = st.selectbox("Ver gráfica de una empresa", choices, index=None,
                            placeholder="Selecciona una empresa", key=f"home_company_{day}_{seconds}")
    if selected is None:
        return
    row = candidates.loc[candidates.ticker == selected].iloc[0]
    band_values = st.columns(3)
    band_values[0].metric(f"Precio · {cut_clock} ET", f"{row.opening_price:.3f}")
    band_values[1].metric("Bollinger superior provisional", f"{row.cutoff_bb_upper:.3f}")
    band_values[2].metric("Bollinger inferior provisional", f"{row.cutoff_bb_lower:.3f}")
    refresher = market_refresh(day, 2)
    market_window = (now.weekday() < 5 and (9, 31) <= (now.hour, now.minute) <= (16, 15))
    if market_window:
        refresher.start()
    _, session_candles, updated = load_daily_session(day)
    chart_candles = session_candles
    if (chart_candles.empty or selected not in set(chart_candles.ticker)) and frozen_candles is not None:
        chart_candles = frozen_candles
    if chart_candles.empty or selected not in set(chart_candles.ticker):
        st.info("Faltan velas previas de 15 minutos para dibujar esta empresa.")
        return
    minutes = home_minute_data(selected, day) if market_window else pd.DataFrame()
    st.plotly_chart(build_home_chart(
        chart_candles, selected, seconds, float(row.opening_price),
        float(row.cutoff_bb_upper), float(row.cutoff_bb_lower), float(row.gap_pct),
        minutes, now),
        width="stretch", config={"displaylogo": False,
                                 "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    st.caption("Puntos claros: Bollinger provisional calculado con el precio recibido "
               f"hasta {cut_clock} ET. Las velas son de 15 minutos; el punto del corte "
               "no representa una vela completa. La vela tenue en formación, cuando hay "
               "minutos completos disponibles, también usa BB(20,2) de 15 minutos. "
               + (f"Última descarga de velas: {updated}." if updated else
                  "Todavía no hay velas posteriores guardadas."))


def historical_panel() -> None:
    st.subheader("Historial")
    coverage = ten_second_coverage()
    captured_days = int((coverage.ten_second_quotes > 0).sum())
    historical_days = int((coverage.historical_10s_quotes > 0).sum())
    st.caption(f"Apertura de 10 segundos, últimas {len(coverage)} sesiones: "
               f"{captured_days} con cotizaciones recibidas en vivo; "
               f"{historical_days} con operaciones históricas SIP. "
               "Las operaciones históricas no indican cuándo llegaron al equipo.")
    dates = session_dates()
    if st.button("Obtener histórico reciente de 15 min"):
        with st.spinner("Descargando histórico reciente..."):
            try:
                saved, available, missing, scanned = refresh_real(HISTORY_SESSION_LIMIT)
                st.success(f"Se guardaron {saved} sesiones de {available} fechas disponibles; "
                           f"{scanned} de {len(TICKERS)} símbolos devolvieron velas.")
                if missing:
                    st.warning("Sin velas del proveedor: " + ", ".join(missing))
            except Exception as exc:
                st.error(f"No se pudo descargar: {exc}")
        dates = session_dates()
    if not dates:
        st.info("Aún no hay sesiones guardadas para esta versión del scanner.")
        return
    selected_date = st.selectbox("Fecha de sesión", dates)
    chart_context(f"history:{selected_date}")
    setups, candles, _ = load_daily_session(selected_date)
    st.caption(f"{selected_date} · {len(setups)} símbolos analizados")
    session_view(setups, candles, f"hist_{selected_date}")
    st.divider()
    st.subheader("Exportación histórica")
    setups_all, candles_all = export_history(HISTORY_SESSION_LIMIT)
    st.caption(f"{len(setups_all)} registros diagnósticos en hasta "
               f"{HISTORY_SESSION_LIMIT} sesiones. "
               "Las velas de 1 minuto solo están disponibles para fechas recientes.")
    if not setups_all.empty:
        outcomes = build_outcomes(setups_all, candles_all)
        with io.BytesIO() as buffer:
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("diagnosticos.csv", setups_all.to_csv(index=False))
                archive.writestr("velas_15m.csv", candles_all.to_csv(index=False))
                archive.writestr("resultados_descriptivos.csv", outcomes.to_csv(index=False))
                archive.writestr("cobertura_apertura_10s.csv", coverage.to_csv(index=False))
            st.download_button("Exportar diagnósticos y velas (ZIP/CSV)", buffer.getvalue(),
                               file_name="efecto_iman_diagnostico.zip",
                               mime="application/zip")


def trial_session_setups(setups: pd.DataFrame, day_rows: pd.DataFrame,
                         score_list: bool = False,
                         model_20s: bool = False,
                         max_signals: int | None = None,
                         min_gap_pct: float = 0.0) -> pd.DataFrame:
    """Misma estructura de Historial; sólo sustituye score y selección de prueba."""
    trial = day_rows.set_index("ticker")
    result = setups.copy()
    result["trial_score_final_original"] = result.score
    if model_20s:
        mapping = (
            ("score", "model_score_20s"),
            ("trial_model_score_raw_10s", "model_index_raw_20s"),
            ("trial_score_actual_10s", "existing_score_at_20s"),
            ("trial_signal_status", "trial_signal_status_20s"),
            ("trial_failed_conditions_10s", "trial_failed_conditions_20s"),
            ("trial_gap_10s_pct", "gap_pct"),
            ("trial_distance_10s_pct", "distance_outside_target_pct"),
            ("trial_angle_10s_deg", "target_angle_deg_5m_equiv"),
            ("manual_label", "manual_label"),
            ("early_data_status", "early_data_status"),
            ("trial_candidate", "trial_candidate_20s"),
            ("model_training_example", "model_training_example"),
        )
        for destination, source in mapping:
            result[destination] = (result.ticker.map(trial[source])
                                   if source in trial.columns else pd.NA)
        result["score"] = pd.to_numeric(result.score, errors="coerce")
        result["trial_score_difference_10s"] = (
            result.score - pd.to_numeric(result.trial_score_actual_10s,
                                         errors="coerce")).round(1)
        result["candidate"] = (result.trial_candidate.eq(True) &
                               pd.to_numeric(result.trial_gap_10s_pct,
                                             errors="coerce").abs().ge(min_gap_pct))
        if max_signals is not None:
            ranked = result.loc[result.candidate].sort_values(
                ["score", "ticker"], ascending=[False, True]).head(max_signals)
            result["candidate"] = result.index.isin(ranked.index)
        for destination, source in (("opening_price", "price"),
                                    ("gap_pct", "gap_pct"),
                                    ("cutoff_bb_upper", "upper_provisional"),
                                    ("cutoff_bb_lower", "lower_provisional")):
            if source in trial.columns:
                observed = result.ticker.map(trial[source])
                result[destination] = observed.combine_first(result[destination])
        result.loc[result.early_data_status.eq("Disponible"), "cutoff_seconds"] = 20
        return result.sort_values(["score", "ticker"], ascending=[False, True],
                                  na_position="last").reset_index(drop=True)
    for destination, source in (
            ("score", "experimental_score_10s"),
            ("trial_model_score_raw_10s", "model_score_without_filters_10s"),
            ("trial_score_actual_10s", "existing_score_at_cutoff"),
            ("trial_score_difference_10s", "score_difference_10s"),
            ("trial_signal_status", "trial_signal_status"),
            ("trial_failed_conditions_10s", "trial_failed_conditions_10s"),
            ("trial_gap_10s_pct", "gap_pct"),
            ("trial_distance_10s_pct", "distance_outside_target_pct"),
            ("trial_angle_10s_deg", "target_angle_deg_5m_equiv"),
            ("manual_label", "manual_label"),
            ("early_data_status", "early_data_status"),
            ("trial_candidate", "trial_candidate")):
        result[destination] = (result.ticker.map(trial[source])
                               if source in trial.columns else pd.NA)
    result.loc[result.trial_distance_10s_pct.le(0), "trial_score_actual_10s"] = pd.NA
    result["candidate"] = (result.score.ge(70) if score_list else
                           result.trial_candidate.eq(True))
    return result.sort_values(["score", "trial_model_score_raw_10s", "ticker"],
                              ascending=[False, False, True],
                              na_position="last").reset_index(drop=True)


def trial_history_panel() -> None:
    st.subheader("Historial prueba")
    st.caption("Diagnóstico de apertura, señales para revisar y gráficas ordenadas "
               "por el modelo experimental del corte de 20 segundos.")
    st.info("La banda es BB(20,2) de velas de 15 minutos, calculada "
            "provisionalmente con operaciones disponibles hasta 09:30:20 ET. "
            "El modelo de esta vista se ajustó retrospectivamente con todos los "
            "juicios guardados: los scores de fechas ya etiquetadas pueden ser "
            "de entrenamiento. La evaluación por fechas está más abajo.")
    snapshot_dir = DATA_DIR / "snapshots"
    files = sorted(snapshot_dir.glob("manual_label_history_trial_*.csv"))
    model_files = sorted(snapshot_dir.glob("cutoff_model_history_20s_*.csv"))
    if st.button("Recalcular Historial prueba con mis etiquetas"):
        with st.spinner("Recalculando las puntuaciones de 20 s y la comparación histórica..."):
            try:
                from evaluate_manual_labels import main as evaluate_manual_labels
                from study_cutoff_models import main as study_cutoffs
                evaluate_manual_labels()
                study_cutoffs()
                st.success("Comparación actualizada.")
            except ValueError as exc:
                st.error(f"Todavía no se puede entrenar el modelo: {exc}")
        files = sorted(snapshot_dir.glob("manual_label_history_trial_*.csv"))
        model_files = sorted(snapshot_dir.glob("cutoff_model_history_20s_*.csv"))
    if not files or not model_files:
        st.info("Pulsa «Recalcular Historial prueba con mis etiquetas» para "
                "crear la vista de 20 segundos.")
        return
    history = pd.read_csv(files[-1])
    model_history = pd.read_csv(model_files[-1])
    model_metadata_path = model_files[-1].with_suffix(".json")
    if model_metadata_path.exists():
        import json
        model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
        st.caption(f"El índice 70 corresponde al umbral bruto "
                   f"{model_metadata['threshold_raw']:.3f} del clasificador. "
                   f"El modelo se ajustó con {model_metadata['training_cases']} "
                   "juicios y el umbral se eligió con fechas anteriores dentro "
                   "del estudio. El índice no es una "
                   "probabilidad de éxito.")
    labeled_comparison = history.loc[
        history.manual_label.isin(("Excelente", "Buena", "Mala")) &
        history.early_data_status.eq("Disponible")]
    with st.expander("Comparación anterior con el experimento de 10 segundos"):
        if labeled_comparison.empty:
            st.write("Todavía no hay juicios con un corte real de 10 segundos.")
        else:
            positive = labeled_comparison.manual_label.isin(("Excelente", "Buena"))
            comparison = []
            for description, column, timing in (
                    ("Original a 10 s", "existing_score_at_cutoff", "09:30:10"),
                    ("Prueba a 10 s", "experimental_score_10s", "09:30:10"),
                    ("Original posterior", "existing_final_score", "Tras la primera vela de 15 min")):
                selected = labeled_comparison[column].ge(70)
                comparison.append({
                    "Score": description, "Momento": timing,
                    "Buenas/Excelentes ≥70": int((positive & selected).sum()),
                    "Malas ≥70": int((~positive & selected).sum()),
                    "Total ≥70 con juicio": int(selected.sum()),
                })
            st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
            unlabeled_high = int((history.manual_label.eq("Pendiente") &
                                  history.experimental_score_10s.ge(70)).sum())
            st.caption(f"Hay {unlabeled_high} scores de prueba ≥70 sin juicio visual. "
                       "Los casos ya juzgados se usaron para reajustar la prueba, así "
                       "que esta tabla es descriptiva, no una validación independiente. "
                       "El score original posterior incorpora datos que todavía no "
                       "existían a los 10 segundos.")
    dates = sorted(model_history.session_date.unique(), reverse=True)
    selected_day = st.selectbox("Fecha de sesión", dates, key="trial_date")
    chart_context(f"trial:{selected_day}")
    day_rows = model_history.loc[model_history.session_date.eq(selected_day)]
    setups, candles, _ = load_daily_session(selected_day)
    min_gap = st.slider("Gap mínimo de la lista (%)", 0.0, 5.0,
                        float(DEFAULT_THRESHOLDS.minimum_gap_pct or 0.0),
                        step=0.25, key="trial20_min_gap")
    total_signals = int(day_rows.trial_candidate_20s.fillna(False).sum())
    signal_count = int((day_rows.trial_candidate_20s.fillna(False) &
                        day_rows.gap_abs_pct.abs().ge(min_gap)).sum())
    if signal_count:
        max_signals = st.slider("Señales visibles en la lista", 1, signal_count,
                                min(8, signal_count), key=f"trial20_limit_{selected_day}")
    else:
        max_signals = 0
    trial_setups = trial_session_setups(
        setups, day_rows, score_list=True, model_20s=True,
        max_signals=max_signals, min_gap_pct=min_gap)
    available = int(day_rows.early_data_status.eq("Disponible").sum())
    st.caption(f"{selected_day} · {len(trial_setups)} símbolos analizados · "
               f"{available} con operación real antes de 20 s. "
               "La lista exige precio fuera de Bollinger al corte, dirección "
               "coherente con la apertura e índice experimental ≥70. "
               f"Hay {total_signals} señales según el modelo, {signal_count} "
               f"con gap de al menos {min_gap:.2f}%; se muestran "
               f"las {max_signals} mejor puntuadas en la lista. "
               "Las velas posteriores sirven para revisar visualmente el caso.")
    if total_signals > 15:
        st.warning("El modelo marcó muchas aperturas de esta sesión. La lista "
                   "ordenada sirve para revisar y etiquetar; este recuento todavía "
                   "no demuestra que funcione sobre todo el universo.")
    session_view(trial_setups, candles, f"trial_{selected_day}", trial=True,
                 trial_score_list=True, trial_seconds=20)
    st.divider()
    stamp = files[-1].stem.removeprefix("manual_label_history_trial_")
    summary_file = snapshot_dir / f"manual_label_daily_summary_{stamp}.csv"
    report_file = snapshot_dir / f"manual_label_validation_{stamp}.json"
    with st.expander("Evaluación de las 50 sesiones y comparación de scores"):
        if not summary_file.exists():
            st.info("Recalcula la comparación para obtener el resumen de 50 sesiones.")
            return
        summary = pd.read_csv(summary_file)
        scored = history.loc[history.early_data_status.eq("Disponible")]
        up = int(scored.score_difference_10s.gt(0).sum())
        high = int(scored.experimental_score_10s.ge(70).sum())
        raw_high = int(scored.model_score_without_filters_10s.ge(70).sum())
        st.write(f"En {len(summary)} sesiones se observaron {len(scored)} cortes de 10 s. "
                 f"El modelo antes de reglas superó 70 en {raw_high}; entre las "
                 f"aperturas con gap suficiente y precio fuera de banda, {high} "
                 f"tuvieron score de prueba ≥70. De ellas, "
                 f"{int(scored.trial_candidate.sum())} cumplieron también todos los "
                 "filtros y quedaron como candidatos. "
                 f"El score de prueba fue mayor que el score anterior en {up} casos. "
                 "Las empresas sin juicio visual no se cuentan como aciertos.")
        st.caption("Las etiquetas proceden de casos escogidos para revisión, así que el "
                   "modelo puede asignar scores altos a empresas fuera de ese grupo. "
                   "La prueba en fechas posteriores sólo evalúa los casos etiquetados.")
        if {"manual_good_candidates", "manual_bad_candidates",
                "unreviewed_candidates"}.issubset(summary.columns):
            st.caption(f"De los candidatos de prueba en estas sesiones: "
                       f"{int(summary.manual_good_candidates.sum())} están marcados "
                       f"Excelente/Buena, {int(summary.manual_bad_candidates.sum())} "
                       f"Mala y {int(summary.unreviewed_candidates.sum())} siguen "
                       "Pendiente. Los juicios usados para entrenar no son validación "
                       "independiente.")
        if report_file.exists():
            import json
            result = json.loads(report_file.read_text(encoding="utf-8"))
            validation = result.get("windows", {}).get("10", {})
            outcome = validation.get("model_ge_70", {})
            st.caption(f"Comparación anterior (otro modelo) en fechas posteriores: "
                       f"{outcome.get('tp_buenas', 0)} buenas detectadas, "
                       f"{outcome.get('fn_buenas', 0)} omitidas, "
                       f"{outcome.get('tn_malas', 0)} malas rechazadas y "
                       f"{outcome.get('fp_malas', 0)} aceptadas. "
                       "El nuevo score IA se evalúa en el panel siguiente.")
            gated = validation.get("trial_candidate_with_existing_filters", {})
            if gated:
                st.caption(f"Al aplicar también los filtros actuales de candidato "
                           f"a esas fechas: {gated.get('tp_buenas', 0)} buenas y "
                           f"{gated.get('fp_malas', 0)} malas habrían entrado en la "
                           "lista. Sólo se conoce el resultado de casos etiquetados.")
        st.dataframe(summary, hide_index=True, width="stretch")
        st.download_button("Descargar resumen (CSV)", summary.to_csv(index=False),
                           file_name="historial_prueba_resumen.csv", mime="text/csv")
    with st.expander("Estudio anterior: 10 a 60 segundos", expanded=False):
        st.caption("Este estudio anterior puede tener archivos desactualizados. "
                   "Aprende de tus etiquetas Excelente, Buena y Mala. Usa gap, distancia, "
                   "Bandas de Bollinger y su reacción al corte; no aplica el límite de "
                   "25° ni incorpora el score heurístico como variable.")
        if st.button("Actualizar estudio de inteligencia artificial"):
            with st.spinner("Entrenando con fechas anteriores y evaluando fechas posteriores..."):
                from study_early_ai import main as run_early_ai_study
                run_early_ai_study()
            st.success("Estudio actualizado.")
        ai_reports = sorted(snapshot_dir.glob("early_ai_study_*.json"))
        if ai_reports:
            import json
            ai = json.loads(ai_reports[-1].read_text(encoding="utf-8"))
            coverage = []
            for second, result in ai["windows"].items():
                coverage.append({"Corte": f"{second} s", "Etiquetadas": result["cases"],
                                 "Fechas": result["days"],
                                 "Prueba posterior": result["test_cases"],
                                 "¿Validable?": "Sí" if result["status"] ==
                                 "validated_on_later_dates" else "Aún no"})
            st.dataframe(pd.DataFrame(coverage), hide_index=True, width="stretch")
            evaluated = ai["windows"].get("10", {})
            if evaluated.get("status") == "validated_on_later_dates":
                rows = []
                for name, result in evaluated["models"].items():
                    outcome = result["test"]
                    rows.append({"Método": name,
                                 "Buenas detectadas": outcome["good_detected"],
                                 "Malas rechazadas": outcome["bad_rejected"],
                                 "Excelentes detectadas": outcome["excellent_detected"],
                                 "Exactitud equilibrada": outcome["balanced_accuracy"]})
                st.caption("Prueba reservada por fecha: " +
                           " a ".join(evaluated["test_dates"]) +
                           ". El modelo se selecciona por resultados de entrenamiento.")
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                three = evaluated.get("three_label_exploratory", {})
                if three:
                    st.caption(f"Al intentar distinguir Excelente de Buena y Mala por "
                               f"separado, acertó {three['test_correct_excellent']}/"
                               f"{three['test_total_excellent']} Excelentes; "
                               f"F1 promedio de las tres clases: {three['test_macro_f1']}. "
                               "Esta parte aún es exploratoria.")
                with st.expander("Qué aportan el gap, el ángulo y otras variables"):
                    ablations = evaluated.get("ablations_exploratory", {})
                    ablation_rows = [{"Variables": key.replace("_", " "),
                                      "Buenas detectadas": value["test"]["good_detected"],
                                      "Malas rechazadas": value["test"]["bad_rejected"],
                                      "Exactitud equilibrada":
                                      value["test"]["balanced_accuracy"]}
                                     for key, value in ablations.items()]
                    if ablation_rows:
                        st.dataframe(pd.DataFrame(ablation_rows),
                                     hide_index=True, width="stretch")
                    st.caption("Estas comparaciones exploran si las variables aportan "
                               "separación; una importancia de árbol no establece una "
                               "regla universal ni demuestra causalidad.")
                    grid = evaluated.get("observed_gap_angle_grid_training", [])
                    if grid:
                        st.write("Casos observados por tamaño de gap y ángulo")
                        st.dataframe(pd.DataFrame(grid), hide_index=True,
                                     width="stretch")
                        st.caption("Cada casilla muestra cuántas etiquetas había en "
                                   "ese rango. Las casillas pequeñas no permiten "
                                   "deducir un límite general.")
                    pairs = evaluated.get("similar_gap_opposite_labels_test", [])
                    if pairs:
                        st.write("Pares con gaps parecidos y etiquetas distintas")
                        st.dataframe(pd.DataFrame([{
                            "Buena/Excelente": f"{pair['good']['date']} {pair['good']['ticker']}",
                            "Gap buena %": pair["good"]["gap_pct"],
                            "Ángulo buena °": pair["good"]["angle_deg"],
                            "Reacción banda buena %": pair["good"]["band_reaction_pct"],
                            "Modelo buena": pair["good"]["model_score"],
                            "Mala": f"{pair['bad']['date']} {pair['bad']['ticker']}",
                            "Gap mala %": pair["bad"]["gap_pct"],
                            "Ángulo mala °": pair["bad"]["angle_deg"],
                            "Reacción banda mala %": pair["bad"]["band_reaction_pct"],
                            "Modelo mala": pair["bad"]["model_score"],
                        } for pair in pairs]), hide_index=True, width="stretch")
            audit = ai.get("universe_audit", {})
            audit_file = ai_reports[-1].with_name(
                ai_reports[-1].name.replace("early_ai_study_", "early_ai_universe_")
                .replace(".json", ".csv"))
            if audit and audit_file.exists():
                with st.expander("Auditoría del score IA sobre todo el universo"):
                    st.warning(f"El modelo daría ≥70 a {audit['ai_score_ge70']}/"
                               f"{audit['observed_cuts']} cortes de 10 s. De ellos, "
                               f"{audit['unlabeled_ai_score_ge70']} no tienen juicio "
                               "visual. Por esa cobertura no sustituye todavía el "
                               "score principal de Historial prueba.")
                    audit_rows = pd.read_csv(audit_file)
                    day_audit = audit_rows.loc[
                        audit_rows.session_date.eq(selected_day) &
                        audit_rows.early_data_status.eq("Disponible")].sort_values(
                            "experimental_score_10s", ascending=False)
                    if not day_audit.empty:
                        st.write(f"Puntuaciones IA del {selected_day} (1 a 100)")
                        st.dataframe(day_audit.reindex(columns=[
                            "ticker", "experimental_score_10s", "manual_label",
                            "gap_pct", "distance_outside_prior_target_pct",
                            "target_angle_deg_5m_equiv", "trial_signal_status"]),
                            hide_index=True, width="stretch")
                    st.download_button("Descargar auditoría IA completa (CSV)",
                                       audit_file.read_bytes(),
                                       file_name=audit_file.name, mime="text/csv")
            prediction_file = ai_reports[-1].with_name(
                ai_reports[-1].name.replace("early_ai_study_", "early_ai_predictions_")
                .replace(".json", ".csv"))
            if prediction_file.exists():
                heldout = pd.read_csv(prediction_file)
                selected_cases = heldout.loc[heldout.session_date.eq(selected_day)]
                if not selected_cases.empty:
                    st.write(f"Predicciones independientes del {selected_day}")
                    shown = selected_cases.reindex(columns=[
                        "ticker", "label", "gap_pct",
                        "distance_outside_target_pct", "target_angle_deg_5m_equiv",
                        "ExtraTrees_score_raw_0_100", "ExtraTrees_selected",
                        "RandomForest_score_raw_0_100", "RandomForest_selected"])
                    st.dataframe(shown, hide_index=True, width="stretch")
                    st.caption("Estos porcentajes son índices del modelo, no "
                               "probabilidades calibradas. Sólo se muestran fechas "
                               "reservadas que el modelo no vio al entrenar.")
                st.download_button("Descargar predicciones reservadas (CSV)",
                                   prediction_file.read_bytes(),
                                   file_name=prediction_file.name, mime="text/csv")
            st.caption("Las operaciones históricas SIP indican hora de mercado, no "
                       "garantizan que los datos hubieran llegado al programa en ese "
                       "segundo. Se necesitan más cortes históricos y etiquetas "
                       "representativas antes de usar esto en la operativa.")
        else:
            st.caption("Pulsa «Actualizar estudio» para comparar los cortes disponibles.")
    with st.expander("Comparación anterior de scores y filtros actuales"):
        if st.button("Repetir comparación con mis juicios"):
            with st.spinner("Comparando modelos por fechas..."):
                from study_10s_scoring import main as run_score_study
                run_score_study()
            st.success("Estudio actualizado.")
        studies = sorted(snapshot_dir.glob("ten_second_scoring_study_*.json"))
        if studies:
            import json
            study = json.loads(studies[-1].read_text(encoding="utf-8"))
            st.caption(f"{study['labels_with_10s']} juicios con corte real; "
                       f"{study['labels_with_signal_at_10s']} tenían gap suficiente y "
                       "precio fuera de Bollinger a los 10 s. Fechas posteriores de "
                       f"prueba: {study['test_dates'][0]} a {study['test_dates'][1]}.")
            rows = []
            baseline = study["baseline_score_heuristico_10s_ge70"]
            rows.append({"Método": "Heurístico anterior", "Buenas detectadas":
                         baseline["tp_buenas"], "Malas rechazadas": baseline["tn_malas"],
                         "Malas aceptadas": baseline["fp_malas"],
                         "Exactitud equilibrada": baseline["exactitud_equilibrada"]})
            for name, comparison in study["models"].items():
                if "later_dates" not in comparison:
                    continue
                metrics = comparison["later_dates"]
                rows.append({"Método": name, "Buenas detectadas": metrics["tp_buenas"],
                             "Malas rechazadas": metrics["tn_malas"],
                             "Malas aceptadas": metrics["fp_malas"],
                             "Exactitud equilibrada": metrics["exactitud_equilibrada"],
                             "Excelentes detectadas": comparison["excellent_detected_later"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            tree_result = study["models"].get("Árboles aleatorios simples", {})
            if "feature_effects_exploratory" in tree_result:
                feature_names = {
                    "gap_abs_pct": "Tamaño del gap",
                    "distance_outside_target_pct": "Distancia fuera de banda",
                    "prior_width_pct": "Ancho previo de Bollinger",
                    "prior_width_change_pp_per_bar": "Cambio previo del ancho",
                    "target_angle_deg_5m_equiv": "Ángulo provisional a 10 s",
                    "existing_score_at_cutoff": "Score heurístico a 10 s",
                }
                effects = pd.DataFrame([
                    {"Variable": feature_names.get(name, name),
                     "Importancia exploratoria": value}
                    for name, value in tree_result["feature_effects_exploratory"].items()
                ]).sort_values("Importancia exploratoria", ascending=False)
                st.dataframe(effects, hide_index=True, width="stretch")
                st.caption("La importancia indica cuánto usaron los árboles cada "
                           "variable; no indica si subirla mejora el caso ni es un "
                           "peso aprobado para el scanner.")
            angle_counts = study.get("angle_above_limit_among_signals_by_label", {})
            if angle_counts:
                good_over = sum(angle_counts.get(label, {}).get("above", 0)
                                for label in ("Excelente", "Buena"))
                good_total = sum(angle_counts.get(label, {}).get("total", 0)
                                 for label in ("Excelente", "Buena"))
                bad_over = angle_counts.get("Mala", {}).get("above", 0)
                bad_total = angle_counts.get("Mala", {}).get("total", 0)
                st.caption(f"Ángulo provisional superior al límite de 25°: "
                           f"{good_over}/{good_total} juicios Excelente/Buena y "
                           f"{bad_over}/{bad_total} Mala. A los 10 s este ángulo "
                           "puede ser muy inestable; el filtro estricto se informa "
                           "por separado.")
            st.caption("Estos resultados corresponden sólo a los casos con juicio humano. "
                       "Los juicios no son una muestra aleatoria del universo y el estudio "
                       "no cambia el score ni la operativa. El bosque de esta tabla se "
                       "entrenó sólo con señales, por lo que difiere del modelo que "
                       "genera el Historial prueba.")
        else:
            st.caption("Pulsa «Repetir comparación con mis juicios» para crear el estudio.")
    with st.expander("Modelo experimental por corte: 10, 20 y 30 segundos"):
        st.caption("Se entrena con fechas anteriores, sin datos posteriores al corte. "
                   "Sólo se evalúan casos etiquetados que seguían fuera de Bollinger. "
                   "El índice del modelo no es una probabilidad calibrada.")
        if st.button("Actualizar comparación temporal por cortes"):
            with st.spinner("Evaluando modelos y fechas posteriores..."):
                from study_cutoff_models import main as run_cutoff_study
                run_cutoff_study()
            st.success("Comparación actualizada.")
        cutoff_reports = sorted(snapshot_dir.glob("cutoff_model_study_eligible_*.json"))
        if cutoff_reports:
            import json
            study = json.loads(cutoff_reports[-1].read_text(encoding="utf-8"))
            rows = []
            for second in (10, 20, 30):
                result = study[str(second)]
                chosen = result["selected_by_inner_dates"]
                metrics = chosen["test_counts"]
                original = result["score_actual_ge70"]
                rows.append({"Corte": f"{second} s", "Casos posteriores":
                             result["test_cases"], "Buenas IA": metrics["good_detected"],
                             "Malas rechazadas IA": metrics["bad_rejected"],
                             "Malas aceptadas IA": metrics["bad_accepted"],
                             "Buenas score actual": original["good_detected"],
                             "Malas rechazadas actual": original["bad_rejected"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            rolling = study["rolling_20s"]
            model_counts = rolling["model_total"]
            original_counts = rolling["original_ge70_total"]
            st.caption(f"Al avanzar fecha por fecha a 20 s ({rolling['cases']} casos con "
                       f"juicio): IA detectó {model_counts['good_detected']} buenas y "
                       f"rechazó {model_counts['bad_rejected']} malas; score actual "
                       f"detectó {original_counts['good_detected']} buenas y rechazó "
                       f"{original_counts['bad_rejected']} malas.")
            interval = rolling.get("date_bootstrap_gain_ci95")
            if interval:
                st.caption(f"Mejora de exactitud equilibrada: "
                           f"{rolling['paired_gain_balanced_accuracy']:+.3f}; "
                           f"intervalo exploratorio por fechas: "
                           f"{interval[0]:+.3f} a {interval[1]:+.3f}.")
            selected_file = cutoff_reports[-1].with_name(
                cutoff_reports[-1].name.replace("cutoff_model_study_eligible_",
                                                "cutoff_model_predictions_20s_")
                .replace(".json", ".csv"))
            if selected_file.exists():
                predicted = pd.read_csv(selected_file)
                day = predicted.loc[predicted.session_date.eq(selected_day)]
                if not day.empty:
                    st.write(f"Juicios disponibles del {selected_day} a 20 s")
                    st.dataframe(day.rename(columns={
                        "ticker": "Símbolo", "label": "Juicio visual",
                        "target_band": "Dirección",
                        "gap_abs_pct": "Gap %",
                        "distance_outside_target_pct": "Distancia exterior %",
                        "target_angle_deg_5m_equiv": "Ángulo 20 s °",
                        "provisional_width_ratio": "Ancho BB / previo",
                        "existing_score_at_cutoff": "Score actual 20 s",
                        "model_score": "Índice IA 20 s",
                        "selected": "Seleccionada IA"}).reindex(columns=[
                            "Símbolo", "Juicio visual", "Dirección", "Gap %",
                            "Distancia exterior %", "Ángulo 20 s °",
                            "Ancho BB / previo", "Score actual 20 s", "Índice IA 20 s",
                            "Seleccionada IA"]),
                            hide_index=True, width="stretch")
            st.caption("Los juicios se eligieron manualmente y estas fechas ya se "
                       "examinaron durante el ajuste; falta una prueba futura "
                       "independiente antes de cambiar el score operativo.")


@st.cache_data(ttl=60)
def review_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    return export_history(HISTORY_SESSION_LIMIT)


@st.cache_data(ttl=60)
def review_observations(day: str) -> pd.DataFrame:
    return early_observations(day)


def review_export(reviews: pd.DataFrame, setups: pd.DataFrame,
                  candles: pd.DataFrame) -> pd.DataFrame:
    """Una fila por caso etiquetado y corte real; sin métricas del futuro como variables."""
    records = []
    details = setups.set_index(["session_date", "ticker"])
    for review in reviews.itertuples():
        key = (review.session_date, review.ticker)
        if key not in details.index:
            continue
        observations = review_observations(review.session_date)
        rows = early_features(review.session_date, review.ticker,
                              candles.loc[candles.session_date == review.session_date],
                              observations)
        audit = details.loc[key]
        if rows.empty:
            records.append({"session_date": review.session_date, "ticker": review.ticker,
                            "window_seconds": pd.NA, "source": "SIN_CORTE",
                            "manual_label": review.label,
                            "label_updated_at": review.updated_at,
                            "audit_final_score": audit.score,
                            "audit_final_candidate": audit.candidate})
        for feature in rows.to_dict("records"):
            records.append({**feature, "manual_label": review.label,
                            "label_updated_at": review.updated_at,
                            "audit_final_score": audit.score,
                            "audit_final_candidate": audit.candidate})
    return pd.DataFrame.from_records(records)


def trial_review_cases(setups: pd.DataFrame, trial_history: pd.DataFrame,
                       model_20s: bool = False) -> pd.DataFrame:
    """Une los casos de Historial prueba con las velas usadas en la revisión."""
    score_column = "model_score_20s" if model_20s else "experimental_score_10s"
    actual_column = "existing_score_at_20s" if model_20s else "existing_score_at_cutoff"
    candidate_column = "trial_candidate_20s" if model_20s else "trial_candidate"
    status_column = "trial_signal_status_20s" if model_20s else "trial_signal_status"
    columns = ["session_date", "ticker", score_column,
               actual_column, candidate_column, "early_data_status",
               "gap_pct", "target_band", "distance_outside_target_pct",
               "distance_outside_prior_target_pct", "target_angle_deg_5m_equiv",
               status_column]
    trial = trial_history.reindex(columns=columns).rename(columns={
        score_column: "review_score_trial",
        actual_column: "review_score_actual_10s",
        candidate_column: "review_trial_candidate",
        "gap_pct": "review_gap_10s_pct",
        "target_band": "review_target_band",
        "distance_outside_target_pct": "review_distance_10s_pct",
        "distance_outside_prior_target_pct": "review_prior_distance_10s_pct",
        "target_angle_deg_5m_equiv": "review_angle_10s_deg",
        status_column: "review_signal_status",
    })
    result = setups.merge(trial, on=["session_date", "ticker"], how="inner",
                          validate="one_to_one")
    result["review_trial_candidate"] = result.review_trial_candidate.eq(True)
    result["review_score_high"] = result.review_score_trial.ge(70)
    return result


def balanced_ai_review_sample(setups: pd.DataFrame,
                              labeled: dict[tuple[str, str], str]) -> pd.DataFrame:
    """Hasta doce aperturas exteriores pendientes por día, de todo el rango IA."""
    eligible = setups.loc[
        setups.early_data_status.eq("Disponible") &
        pd.to_numeric(setups.review_prior_distance_10s_pct, errors="coerce").gt(0) &
        setups.gap_direction.map({"UP": "UPPER", "DOWN": "LOWER"}).eq(
            setups.review_target_band) &
        pd.to_numeric(setups.review_score_trial, errors="coerce").notna()].copy()
    eligible = eligible.loc[[
        (row.session_date, row.ticker) not in labeled for row in eligible.itertuples()]]
    if eligible.empty:
        return eligible
    eligible["score_rank"] = eligible.groupby("session_date").review_score_trial.rank(
        pct=True, method="first")
    eligible["score_group"] = pd.cut(eligible.score_rank, [0, 1/3, 2/3, 1],
                                     labels=("bajo", "medio", "alto"),
                                     include_lowest=True)
    sample = pd.concat([
        rows.sample(n=min(4, len(rows)), random_state=42)
        for _, rows in eligible.groupby(["session_date", "score_group"],
                                        observed=True)
    ])
    return sample.drop(columns=["score_rank", "score_group"]).sort_values(
        ["session_date", "review_score_trial"], ascending=[False, False])


def change_review_page(delta: int, pages: int) -> None:
    st.session_state.manual_review_page = max(
        1, min(pages, st.session_state.manual_review_page + delta))


def review_page_controls(pages: int, position: str) -> None:
    page = st.session_state.manual_review_page
    _, previous, current, following, _ = st.columns([2, .6, 1.2, .6, 2])
    previous.button("←", key=f"review_prev_{position}", help="Página anterior",
                    disabled=page <= 1, use_container_width=True,
                    on_click=change_review_page, args=(-1, pages))
    current.markdown(f"<div style='text-align:center;padding-top:9px'>"
                     f"Página {page} de {pages}</div>", unsafe_allow_html=True)
    following.button("→", key=f"review_next_{position}", help="Página siguiente",
                     disabled=page >= pages, use_container_width=True,
                     on_click=change_review_page, args=(1, pages))


def manual_review_panel() -> None:
    st.subheader("Revisión manual de aperturas")
    st.info("Elige un juicio visual debajo de cada gráfica: se guarda automáticamente "
            "en SQLite al seleccionarlo. Puedes cambiar de página sin pulsar Guardar. "
            "El CSV es una descarga opcional para analizar tus etiquetas.")
    saved_notice = st.session_state.pop("review_saved_notice", None)
    if saved_notice:
        st.success(saved_notice)
    st.caption("La gráfica de apertura muestra la última hora previa y las primeras "
               "dos horas de la sesión. Los cortes tempranos sólo muestran datos "
               "disponibles hasta ese segundo. Los juicios no modifican el scanner operativo.")
    setups, candles = review_history()
    if setups.empty:
        st.info("No hay sesiones históricas locales. Descárgalas desde Historial.")
        return
    source = st.radio("Revisar casos de", ("Historial prueba", "Auditoría IA",
                                          "Historial actual"),
                      horizontal=True)
    trial_mode = source != "Historial actual"
    ai_mode = source == "Auditoría IA"
    review_cutoff_seconds = 10 if ai_mode else 20
    recalc_notice = st.session_state.pop("review_recalc_notice", None)
    if recalc_notice:
        st.success(recalc_notice)
    if trial_mode:
        pattern = ("early_ai_universe_*.csv" if ai_mode else
                   "cutoff_model_history_20s_*.csv")
        files = sorted((DATA_DIR / "snapshots").glob(pattern))
        if not files:
            st.info("Primero genera el estudio en Historial prueba.")
            return
        trial_history = pd.read_csv(files[-1])
        setups = trial_review_cases(setups, trial_history, model_20s=not ai_mode)
        st.caption(("Se muestran los scores IA de todo el universo. La muestra "
                    "equilibrada invita a etiquetar casos altos, medios y bajos "
                    "que estaban fuera de la banda previa. " if ai_mode else
                    "Se muestran las puntuaciones del modelo de 20 segundos de "
                    "Historial prueba. Etiqueta también casos con índice alto que "
                    "visualmente hayan fallado. ") +
                   f"El score procede del corte real de {review_cutoff_seconds} s; la gráfica completa "
                   "sirve para juzgar el resultado. Marca también «Mala» cuando una "
                   "selección de prueba no cumplió lo que buscabas. "
                   f"Última sesión disponible en esta prueba: "
                   f"{trial_history.session_date.max()}.")
    dates = sorted(setups.session_date.unique(), reverse=True)[:HISTORY_SESSION_LIMIT]
    if not dates:
        st.info("No hay sesiones de Historial prueba que coincidan con las velas locales.")
        return
    setups = setups.loc[setups.session_date.isin(dates)].copy()
    reviews = load_reviews()
    reviews = reviews.loc[reviews.session_date.isin(dates)]
    labeled = {(row.session_date, row.ticker): row.label for row in reviews.itertuples()}
    coverage = {day: set(review_observations(day).window_seconds.unique()) for day in dates}
    metrics = st.columns(4)
    metrics[0].metric("Sesiones", len(dates))
    metrics[1].metric(("IA ≥70 (sin validar)" if ai_mode else "Señales con score ≥70")
                      if trial_mode else "Aperturas fuera de banda",
                      int(setups.review_score_high.sum()) if trial_mode else
                      int((setups.distance_to_band_pct > 0).sum()))
    metrics[2].metric("Etiquetadas", len(reviews))
    metrics[3].metric("Historial con 20/30/60 s",
                      sum(bool({20, 30, 60} & coverage[d]) for d in dates))
    if trial_mode:
        if st.button("Recalcular auditoría IA" if ai_mode else
                     "Recalcular Historial prueba con mis juicios guardados"):
            with st.spinner("Actualizando la auditoría..." if ai_mode else
                            "Actualizando el score de prueba y su evaluación..."):
                try:
                    if ai_mode:
                        from study_early_ai import main as recalculate
                        recalculate(audit_only=True)
                    else:
                        from study_cutoff_models import main as recalculate
                        recalculate()
                    st.session_state.review_recalc_notice = (
                        "Auditoría actualizada con tus juicios. Para actualizar también "
                        "la evaluación por fechas, ejecuta el estudio completo de IA."
                        if ai_mode else
                        "Resultados actualizados. La evaluación reservada por fechas "
                        "permanece separada de la vista exploratoria.")
                    st.rerun()
                except ValueError as exc:
                    st.error(f"No se pudo recalcular: {exc}")
        st.caption("Los juicios se guardan al seleccionar una opción. Recalcula cuando "
                   "termines. La evaluación reserva fechas posteriores, pero la vista "
                   "exploratoria se entrena después con todos los juicios disponibles.")
    with st.expander("Variables tempranas y score heurístico anterior"):
        st.write(f"A los {review_cutoff_seconds} segundos disponemos de operación inicial y última operación "
                 "del corte histórico, gap frente al cierre previo, BB(20,2) de 15 min "
                 "provisional, distancia exterior, ancho y pendiente previos. "
                 "El ángulo dentro del corte compara la primera y última operación "
                 "con tiempos distintos y se normaliza a cinco minutos.")
        st.write("Los archivos históricos guardan cortes de 10, 20 y 30 segundos, "
                 "pero no reconstruyen todos los puntos entre cortes. No se usan velas posteriores "
                 "para recrearlos. La sesión completa sirve para etiquetar, pero sus "
                 "métricas posteriores se excluyen de las variables tempranas.")
        from scanner_settings import SCORE_WEIGHTS
        st.dataframe(pd.DataFrame([{"factor": key, "peso_relativo": value}
                                   for key, value in SCORE_WEIGHTS.items()]),
                     hide_index=True, width="stretch")
        st.caption("El score heurístico no es una probabilidad de éxito. El clasificador "
                   "experimental usa esos datos y juicios visuales; su evaluación por "
                   "fecha se guarda aparte y aún no valida todas las empresas.")
    filters = st.columns([1.7, 1.2, 1.1, .8])
    pool_options = (("Muestra equilibrada sin etiqueta", "Candidatos IA fuera de banda",
                     "Todos con corte de 10 s") if ai_mode else
                    ("Señales con score prueba ≥70", "Señales modelo válidas",
                     "Descartados por la prueba",
                     "Original 20 s ≥70 sin juicio", "Original 20 s <70 sin juicio",
                     "Todos con corte de 20 s") if trial_mode else
                    ("Fuera de banda y gap ≥1%", "Todas las aperturas fuera de banda",
                     "Candidatos del scanner", "Todos"))
    pool_name = filters[0].selectbox("Casos", pool_options)
    candidate_days = (sorted(setups.loc[setups.review_score_high,
                                         "session_date"].unique(), reverse=True)
                      if trial_mode else [])
    preferred_day = candidate_days[0] if candidate_days else dates[0]
    date_choice = filters[1].selectbox("Sesión", (*dates, "Todas"),
                                       index=dates.index(preferred_day))
    label_choice = filters[2].selectbox("Etiqueta", ("Todas", "Pendiente", *LABELS))
    page_size = filters[3].selectbox("Gráficas/página", (12, 24, 50))
    only_high_score = (st.checkbox("Sólo score final ≥ 70") if not trial_mode else False)
    if pool_name == "Muestra equilibrada sin etiqueta":
        pool = balanced_ai_review_sample(setups, labeled)
    elif pool_name == "Original 20 s ≥70 sin juicio":
        pool = setups.loc[setups.early_data_status.eq("Disponible") &
                          setups.review_score_actual_10s.ge(70)].copy()
        pool = pool.loc[[
            (row.session_date, row.ticker) not in labeled for row in pool.itertuples()]]
    elif pool_name == "Original 20 s <70 sin juicio":
        pool = setups.loc[setups.early_data_status.eq("Disponible") &
                          setups.review_distance_10s_pct.gt(0) &
                          setups.review_score_actual_10s.lt(70)].copy()
        pool = pool.loc[[
            (row.session_date, row.ticker) not in labeled for row in pool.itertuples()]]
    elif pool_name == "Candidatos IA fuera de banda":
        pool = setups.loc[setups.review_trial_candidate].copy()
    elif pool_name == "Señales con score prueba ≥70":
        pool = setups.loc[setups.review_score_high].copy()
    elif pool_name == "Señales modelo válidas":
        pool = setups.loc[setups.review_trial_candidate].copy()
    elif pool_name == "Descartados por la prueba":
        pool = setups.loc[setups.early_data_status.eq("Disponible") &
                          ~setups.review_trial_candidate].copy()
    elif pool_name in ("Todos con corte de 10 s", "Todos con corte de 20 s"):
        pool = setups.loc[setups.early_data_status.eq("Disponible")].copy()
    elif pool_name == "Fuera de banda y gap ≥1%":
        pool = setups.loc[(setups.distance_to_band_pct > 0) &
                          (setups.gap_pct.abs() >= 1)].copy()
    elif pool_name == "Todas las aperturas fuera de banda":
        pool = setups.loc[setups.distance_to_band_pct > 0].copy()
    elif pool_name == "Candidatos del scanner":
        pool = setups.loc[setups.candidate_at_open == True].copy()
    else:
        pool = setups.copy()
    if date_choice != "Todas":
        pool = pool.loc[pool.session_date == date_choice]
    if only_high_score:
        pool = pool.loc[pd.to_numeric(pool.score, errors="coerce") >= 70]
    pool["manual_label"] = [labeled.get((row.session_date, row.ticker), "Pendiente")
                            for row in pool.itertuples()]
    if label_choice != "Todas":
        pool = pool.loc[pool.manual_label == label_choice]
    original_pool = pool_name.startswith(("Original 10 s", "Original 20 s"))
    pool = pool.sort_values(["session_date", "review_score_actual_10s"
                             if original_pool else "review_score_trial" if trial_mode else
                             "distance_to_band_pct"], ascending=[False, False],
                            na_position="last")
    st.caption(f"{len(pool)} casos en este filtro · "
               f"{int(pool.manual_label.eq('Mala').sum())} marcados Mala · "
               f"{int(pool.manual_label.eq('Pendiente').sum())} pendientes. "
               "«Pendiente» no significa Mala.")
    with st.expander("Descargar etiquetas y variables tempranas (opcional)"):
        st.caption("Tus juicios ya están guardados. Genera el CSV sólo si quieres una "
                   "copia para análisis; después de cambiar etiquetas, vuelve a generarlo.")
        if reviews.empty:
            st.caption("Etiqueta al menos un caso para crear el CSV.")
        else:
            if st.button("Generar CSV actualizado"):
                st.session_state.review_csv = review_export(reviews, setups, candles).to_csv(index=False)
            if "review_csv" in st.session_state:
                st.download_button("Descargar CSV", st.session_state.review_csv,
                                   file_name="efecto_iman_revision_manual.csv", mime="text/csv")
    if pool.empty:
        return
    view = st.radio("Vista de la gráfica", ("Apertura (15 min)", "10 s", "20 s",
                                           "30 s", "60 s"),
                    horizontal=True)
    pages = math.ceil(len(pool) / page_size)
    st.session_state.manual_review_page = max(
        1, min(pages, st.session_state.get("manual_review_page", 1)))
    review_page_controls(pages, "top")
    page = st.session_state.manual_review_page
    visible = pool.iloc[(page - 1) * page_size:page * page_size]
    reviewed_on_page = sum(labeled.get((row.session_date, row.ticker)) in LABELS
                           for row in visible.itertuples())
    st.caption(f"Página {page} de {pages} · {reviewed_on_page}/{len(visible)} casos "
               "con juicio guardado en esta página")
    for row in visible.itertuples():
        day, ticker = row.session_date, row.ticker
        sample_candles = candles.loc[(candles.session_date == day) &
                                     (candles.ticker == ticker)]
        observations = review_observations(day)
        features = early_features(day, ticker, sample_candles, observations)
        with st.container(border=True):
            header = st.columns([1.6, 1, 1, 1])
            header[0].markdown(f"**{day} · {ticker} · {row.gap_direction}**")
            if trial_mode:
                header[1].caption(f"Gap {review_cutoff_seconds} s "
                                  f"{row.review_gap_10s_pct:+.2f}%")
                header[2].caption(f"Distancia {review_cutoff_seconds} s "
                                  f"{row.review_distance_10s_pct:+.2f}%")
                trial_score = (f"{row.review_score_trial:.1f}"
                               if pd.notna(row.review_score_trial) else "sin señal")
                if original_pool:
                    header[3].caption(f"Score original {review_cutoff_seconds} s "
                                      f"{row.review_score_actual_10s:.1f}")
                else:
                    header[3].caption(f"{'Score IA' if ai_mode else 'Score prueba'} "
                                      f"{review_cutoff_seconds} s {trial_score}")
                angle = (f"{row.review_angle_10s_deg:+.1f}°"
                         if pd.notna(row.review_angle_10s_deg) else "no medible")
                heuristic = (f"{row.review_score_actual_10s:.1f}"
                             if pd.notna(row.review_score_actual_10s) and
                             row.review_distance_10s_pct > 0 else "sin señal")
                st.caption(f"Ángulo {review_cutoff_seconds} s {angle} · "
                           f"Score heurístico {review_cutoff_seconds} s "
                           f"{heuristic} · Score posterior {row.score:.1f} · "
                           f"{row.review_signal_status}")
            else:
                header[1].caption(f"Gap {row.gap_pct:+.2f}%")
                header[2].caption(f"Distancia {row.distance_to_band_pct:+.2f}%")
                header[3].caption(f"Score final {row.score:.1f}")
            if view == "Apertura (15 min)":
                if sample_candles.empty:
                    st.info("No hay velas de 15 minutos para este caso.")
                else:
                    figure = build_daily_chart(sample_candles, ticker, float(row.gap_pct),
                                               -1, float("nan"), str(row.target_band),
                                               focus_opening=True)
                    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False},
                                    key=f"review_chart_{day}_{ticker}")
            else:
                seconds = int(view.split()[0])
                trajectory = early_trajectory(day, ticker)
                figure = build_early_chart(day, ticker, sample_candles, features,
                                           seconds, trajectory)
                if figure is None:
                    st.info(f"No existe un corte real de {seconds} s para {day} · {ticker}.")
                else:
                    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False},
                                    key=f"review_chart_{day}_{ticker}_{seconds}")
                    chosen = features.loc[features.window_seconds == seconds].iloc[0]
                    angle = (f"{chosen.target_angle_deg_5m_equiv:+.1f}°"
                             if pd.notna(chosen.target_angle_deg_5m_equiv) else "no medible")
                    st.caption(f"Fuente: {chosen.source} · precio {chosen.price:.3f} · "
                               f"BB {chosen.target_band.lower()} provisional "
                               f"{chosen.target_provisional:.3f} · gap {chosen.gap_pct:+.2f}% · "
                               f"distancia exterior {chosen.distance_outside_target_pct:+.2f}% · "
                               f"ángulo {angle}")
            previous = labeled.get((day, ticker), "Pendiente")
            choices = (*LABELS, "Pendiente")
            selected = st.radio("Juicio visual", choices, index=choices.index(previous),
                                horizontal=True, key=f"review_label_{day}_{ticker}")
            st.caption("Juicio guardado automáticamente" if previous != "Pendiente"
                       else "Sin juicio guardado")
            if selected != previous:
                save_review(day, ticker, None if selected == "Pendiente" else selected)
                labeled[(day, ticker)] = selected
                st.session_state.pop("review_csv", None)
                st.session_state.review_saved_notice = (
                    f"{day} · {ticker}: juicio guardado como {selected}")
                st.rerun()
    review_page_controls(pages, "bottom")


def main() -> None:
    initialize_daily_store()
    if "selected_charts" not in st.session_state:
        st.session_state.selected_charts = []
    style()
    st.sidebar.markdown("## Efecto Imán")
    page = st.sidebar.radio("NAVEGACIÓN", ("Home", "Operativa diaria", "Historial",
                                            "Historial prueba", "Revisión manual"),
                            label_visibility="collapsed")
    st.sidebar.caption("Gaps alcistas y bajistas · reversión hacia Bollinger")
    st.sidebar.caption("Análisis local · Sin ejecución de órdenes")
    st.title("Efecto Imán")
    st.caption("Estudio diagnóstico · Bandas de Bollinger de 15 minutos")
    market_clock(page)
    if page == "Home":
        home_panel()
    elif page == "Operativa diaria":
        daily_panel()
    elif page == "Historial":
        historical_panel()
    elif page == "Historial prueba":
        trial_history_panel()
    else:
        manual_review_panel()


if __name__ == "__main__":
    main()
