"""Operativa diaria e historial diagnóstico local de Efecto Imán."""

import io
import importlib
import math
import time
from html import escape
import zipfile
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import real_data
from config import DATA_DIR, TICKERS
from daily_chart import build_daily_chart
from daily_store import (export_history, initialize_daily_store,
                         load_daily_session, save_daily_session, session_dates)
from outcomes import build_outcomes
from opening_live import CaptureProgress, OpeningCapture, next_capture_day
from market_refresh import MarketRefresh
from manual_review import (LABELS, build_early_chart, early_features,
                           early_trajectory,
                           early_observations, load_reviews, save_review)
from signal_audit import first_availability
from subminute_history import ten_second_coverage


NY = ZoneInfo("America/New_York")
HERMOSILLO = ZoneInfo("America/Hermosillo")
MAX_CHARTS = 4
CAPTURE_CACHE_VERSION = 5
OPENING_WINDOW_COLUMNS = ["ticker", "score_open", "gap_pct", "gap_direction",
                          "opening_price", "cutoff_bb_upper", "cutoff_bb_lower",
                          "distance_to_band_pct", "target_angle_cutoff_deg",
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
    "score_post_open_expansion_component",
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
    ready_windows = progress.window_ready_at or {}
    window_counts = progress.window_counts or {}
    window_candidates = progress.window_candidates or {}
    if now.weekday() >= 5:
        return "", "Hoy no hay apertura programada. Abre Operativa diaria antes de las 09:25 ET del próximo día hábil."
    if 60 in ready_windows:
        ready = ready_windows[60].astimezone(NY)
        if not window_counts.get(60, 0):
            return "error", "Corte de 1 minuto completado sin cotizaciones válidas."
        return "ready", (f"Lista de 1 minuto lista a las {ready:%H:%M:%S} ET: "
                         f"{window_counts.get(60, 0)}/{total} cotizaciones; "
                         f"{window_candidates.get(60, 0)} candidatos. "
                         "Se guardaron por separado los cortes de 30 s y 1 min.")
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
    return "active", "Corte de 1 minuto alcanzado. Calculando y guardando las dos listas."


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
               preliminary: bool = False) -> None:
    if setups.empty:
        st.info("No hay candidatos con los límites exploratorios actuales.")
        return
    for row in setups.itertuples():
        cols = st.columns([1.5, 1, .6], vertical_alignment="center")
        cols[0].write(f"**{row.ticker}** · {row.gap_direction}")
        shown_score = row.score_open if preliminary else row.score
        cols[0].caption(f"Score {'apertura' if preliminary else 'actual'} {shown_score:.1f} · "
                        f"banda {row.target_band} · "
                        f"distancia {row.distance_to_band_pct:+.2f}%")
        if availability and row.ticker in availability:
            recorded, source = availability[row.ticker]
            cols[0].caption(f"Primer registro local {recorded[11:19]} ET · {source}")
        if preliminary and pd.notna(row.touched_target_band) and bool(row.touched_target_band):
            cols[0].caption("Ya tocó la banda; alerta preliminar vencida")
        if preliminary and pd.notna(row.target_angle_cutoff_deg):
            cols[0].caption(f"Ángulo dentro del corte {row.target_angle_cutoff_deg:+.1f}° · "
                            f"{row.angle_cutoff_event_span_seconds:.0f} s entre precios")
        angles = [value for value in (row.target_angle_5m_deg,
                  row.target_angle_0935_0940, row.target_angle_0940_0945)
                  if pd.notna(value)]
        if angles:
            detail = f"Ángulo inicial {angles[0]:+.1f}°"
            if len(angles) > 1:
                detail += f" · máximo 15m {max(angles):+.1f}°"
            if pd.notna(row.post_open_bb_expansion_pct):
                detail += f" · ancho {row.post_open_bb_expansion_pct:+.1f}%"
            cols[0].caption(detail)
        cols[1].write(f"{row.gap_pct:+.2f}%")
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
            if pd.notna(setup.cutoff_bb_lower) else float("nan")),
            width="stretch", config={"displayModeBar": False})


def diagnostic_table(setups: pd.DataFrame) -> None:
    if setups.empty:
        st.info("Aún no hay observaciones de esta sesión.")
        return
    st.dataframe(setups.reindex(columns=DISPLAY_COLUMNS), hide_index=True, width="stretch",
                 column_config={
                     "score": st.column_config.NumberColumn("Score actual", format="%.1f"),
                     "score_open": st.column_config.NumberColumn("Score apertura preliminar",
                                                                   format="%.1f"),
                     "score_5m": st.column_config.NumberColumn("Score con ángulo 09:35",
                                                                 format="%.1f"),
                     "failed_conditions": "Condiciones fallidas",
                     "pending_conditions": "Pendientes",
                 })
    st.caption("En cada corte subminuto la distancia usa Bollinger provisional de 15 min; "
               "el ángulo dentro del corte sólo se valora si llegaron dos precios con horas distintas. "
               "A las 09:35 se valora el ángulo inicial; "
               "desde las 09:45 se valora el mayor ángulo de los tres tramos de cinco minutos "
               "y la expansión de Bollinger. Los valores vacíos aún no estaban disponibles.")


def session_view(setups: pd.DataFrame, candles: pd.DataFrame, key: str,
                 preliminary: bool = False,
                 availability: dict[str, tuple[str, str]] | None = None) -> None:
    observed = setups.loc[setups.opening_price.notna()] if not setups.empty else setups
    candidates = observed.loc[observed.candidate_at_open == True] if preliminary else (
        observed.loc[observed.candidate == True])
    if preliminary and not candidates.empty:
        candidates = candidates.sort_values("score_open", ascending=False)
    metrics = st.columns(4)
    metrics[0].metric("Símbolos con apertura", len(observed))
    metrics[1].metric("Preliminares" if preliminary else "Candidatos", len(candidates))
    metrics[2].metric("Toques de banda", int(observed.touched_target_band.fillna(False).sum())
                      if not observed.empty else 0)
    metrics[3].metric("Banda aún separada al toque", int(
        ((observed.touched_target_band == True) &
         (observed.reversion_to_touch_band_pct > 0)).sum()) if not observed.empty else 0)
    st.subheader("Diagnóstico de apertura")
    diagnostic_table(setups)
    st.caption("Un toque puede ocurrir porque la banda se desplaza hasta la apertura. "
               "La columna «band_at_or_beyond_open_at_touch» separa esos casos; "
               "un toque no demuestra rentabilidad.")
    st.subheader("Candidatos y gráficas")
    if preliminary:
        st.caption("Hasta las 09:35 ET se muestra la lista preliminar de apertura, ordenada "
                   "por score de apertura. El ángulo posterior aún no confirma estos casos; "
                   "una entrada puede haber tocado la banda antes de que llegue una actualización.")
    left, right = st.columns([.95, 1.75], gap="medium")
    with left:
        setup_list(candidates, key, availability, preliminary)
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
    for seconds in (30, 60):
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
        for seconds in (30, 60):
            if seconds not in windows:
                continue
            window_setups, _, ready_at = windows[seconds]
            chosen = window_setups.loc[window_setups.candidate_at_open == True].sort_values(
                "score_open", ascending=False)
            label = "30 segundos" if seconds == 30 else "1 minuto"
            with st.expander(f"{label}: {len(chosen)} candidatos", expanded=False):
                cut_time = "09:30:30" if seconds == 30 else "09:31:00"
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
        st.caption("Las listas de 30 segundos y 1 minuto aparecerán cuando "
                   "se congelen sus respectivos cortes.")
    if cut_setups is not None:
        with st.expander("Corte de 10 segundos conservado para auditoría"):
            st.dataframe(cut_setups.reindex(columns=OPENING_WINDOW_COLUMNS),
                         hide_index=True, width="stretch")
    early = market_window and (now.hour, now.minute) < (9, 35)
    active_window = (60 if 60 in windows else 30 if 30 in windows else None)
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
        st.warning("No hay lista guardada de 30 s ni de 1 min; se muestran "
                   "las velas posteriores disponibles.")
    if not updated and not windows:
        st.info("Los cortes de 30 segundos y 1 minuto aún no están disponibles. "
                "Mantén la aplicación abierta antes de las 09:25 ET.")
        return
    if not market_window:
        st.caption("Mercado fuera de horario. Se muestra la sesión local guardada.")
    availability = first_availability(day)
    st.caption("El primer registro local indica cuándo terminó un cálculo desde que se activó "
               "esta versión; no reconstruye señales anteriores ni confirma cuándo se vio la pantalla.")
    session_view(setups, candles, f"daily_{day}", early, availability)


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
                saved, available, missing, scanned = refresh_real(50)
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
    setups_all, candles_all = export_history(50)
    st.caption(f"{len(setups_all)} registros diagnósticos en hasta 50 sesiones. "
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


@st.cache_data(ttl=60)
def review_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    return export_history(50)


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


def change_review_page(delta: int, pages: int) -> None:
    st.session_state.manual_review_page = max(
        1, min(pages, st.session_state.manual_review_page + delta))


def manual_review_panel() -> None:
    st.subheader("Revisión manual de aperturas")
    st.info("Elige un juicio visual debajo de cada gráfica: se guarda automáticamente "
            "en SQLite al seleccionarlo. Puedes cambiar de página sin pulsar Guardar. "
            "El CSV es una descarga opcional para analizar tus etiquetas.")
    saved_notice = st.session_state.pop("review_saved_notice", None)
    if saved_notice:
        st.success(saved_notice)
    st.caption("Los cortes tempranos sólo muestran precios y bandas disponibles hasta ese "
               "segundo. Las etiquetas no modifican el scanner ni su score.")
    setups, candles = review_history()
    if setups.empty:
        st.info("No hay sesiones históricas locales. Descárgalas desde Historial.")
        return
    dates = sorted(setups.session_date.unique(), reverse=True)[:50]
    setups = setups.loc[setups.session_date.isin(dates)].copy()
    reviews = load_reviews()
    reviews = reviews.loc[reviews.session_date.isin(dates)]
    labeled = {(row.session_date, row.ticker): row.label for row in reviews.itertuples()}
    coverage = {day: set(review_observations(day).window_seconds.unique()) for day in dates}
    metrics = st.columns(4)
    metrics[0].metric("Sesiones", len(dates))
    metrics[1].metric("Aperturas fuera de banda", int((setups.distance_to_band_pct > 0).sum()))
    metrics[2].metric("Etiquetadas", len(reviews))
    metrics[3].metric("Historial con 20/30/60 s",
                      sum(bool({20, 30, 60} & coverage[d]) for d in dates))
    with st.expander("Datos para un futuro modelo y pesos actuales"):
        st.write("A los 10 segundos disponemos de operación inicial y última operación "
                 "del corte histórico, gap frente al cierre previo, BB(20,2) de 15 min "
                 "provisional, distancia exterior, ancho y pendiente previos. "
                 "El ángulo dentro del corte compara la primera y última operación "
                 "con tiempos distintos y se normaliza a cinco minutos.")
        st.write("Los archivos históricos actuales no guardan la trayectoria completa "
                 "de 10 segundos ni cortes de 20/30 segundos. No se usan velas posteriores "
                 "para recrearlos. La sesión completa sirve para etiquetar, pero sus "
                 "métricas posteriores se excluyen de las variables tempranas.")
        from scanner_settings import SCORE_WEIGHTS
        st.dataframe(pd.DataFrame([{"factor": key, "peso_relativo": value}
                                   for key, value in SCORE_WEIGHTS.items()]),
                     hide_index=True, width="stretch")
        st.caption("El score actual es una escala heurística, no una probabilidad de éxito. "
                   "Después de reunir etiquetas Excelente, Buena y Mala se podrá entrenar un "
                   "clasificador supervisado y validarlo separando las fechas en el tiempo.")
    filters = st.columns([1.7, 1.2, 1.1, .8])
    pool_name = filters[0].selectbox("Casos", ("Fuera de banda y gap ≥1%",
                                               "Todas las aperturas fuera de banda",
                                               "Candidatos del scanner", "Todos"))
    date_choice = filters[1].selectbox("Sesión", ("Todas", *dates))
    label_choice = filters[2].selectbox("Etiqueta", ("Todas", "Pendiente", *LABELS))
    page_size = filters[3].selectbox("Gráficas/página", (12, 24, 50))
    only_high_score = st.checkbox("Sólo score final ≥ 70")
    if pool_name == "Fuera de banda y gap ≥1%":
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
    pool = pool.sort_values(["session_date", "distance_to_band_pct"],
                            ascending=[False, False], na_position="last")
    st.caption(f"{len(pool)} casos en este filtro. «Pendiente» no significa Mala."
               + ("" if only_high_score else
                  " Los casos con score bajo siguen visibles si abrieron fuera de banda."))
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
    view = st.radio("Vista de la gráfica", ("Sesión completa", "10 s", "20 s",
                                           "30 s", "60 s"),
                    horizontal=True)
    pages = math.ceil(len(pool) / page_size)
    st.session_state.manual_review_page = max(
        1, min(pages, st.session_state.get("manual_review_page", 1)))
    page = st.number_input("Página", min_value=1, max_value=pages,
                           key="manual_review_page")
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
            header[1].caption(f"Gap {row.gap_pct:+.2f}%")
            header[2].caption(f"Distancia {row.distance_to_band_pct:+.2f}%")
            header[3].caption(f"Score final {row.score:.1f}")
            if view == "Sesión completa":
                if sample_candles.empty:
                    st.info("No hay velas de 15 minutos para este caso.")
                else:
                    figure = build_daily_chart(sample_candles, ticker, float(row.gap_pct),
                                               -1, float("nan"), str(row.target_band))
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
            selected = st.selectbox("Juicio visual", ("Pendiente", *LABELS),
                                    index=("Pendiente", *LABELS).index(previous),
                                    key=f"review_label_{day}_{ticker}")
            st.caption("Juicio guardado automáticamente" if previous != "Pendiente"
                       else "Sin juicio guardado")
            if selected != previous:
                save_review(day, ticker, None if selected == "Pendiente" else selected)
                labeled[(day, ticker)] = selected
                st.session_state.pop("review_csv", None)
                st.session_state.review_saved_notice = (
                    f"{day} · {ticker}: juicio guardado como {selected}")
                st.rerun()
    navigation = st.columns(2)
    navigation[0].button("Página anterior", disabled=page <= 1,
                         on_click=change_review_page, args=(-1, pages))
    navigation[1].button("Página siguiente", disabled=page >= pages,
                         on_click=change_review_page, args=(1, pages))


def main() -> None:
    initialize_daily_store()
    if "selected_charts" not in st.session_state:
        st.session_state.selected_charts = []
    style()
    st.sidebar.markdown("## Efecto Imán")
    page = st.sidebar.radio("NAVEGACIÓN", ("Operativa diaria", "Historial",
                                            "Revisión manual"),
                            label_visibility="collapsed")
    st.sidebar.caption("Gaps alcistas y bajistas · reversión hacia Bollinger")
    st.sidebar.caption("Análisis local · Sin ejecución de órdenes")
    st.title("Efecto Imán")
    st.caption("Estudio diagnóstico · Bandas de Bollinger de 15 minutos")
    market_clock(page)
    if page == "Operativa diaria":
        daily_panel()
    elif page == "Historial":
        historical_panel()
    else:
        manual_review_panel()


if __name__ == "__main__":
    main()
