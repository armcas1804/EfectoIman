"""Parámetros exploratorios del escáner; None desactiva cada umbral."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScannerThresholds:
    minimum_gap_pct: float | None = 1.0
    maximum_prior_bb_width_pct: float | None = 1.5
    maximum_target_band_slope_pct: float | None = 0.03
    maximum_prior_bb_width_growth: float | None = 0.03  # puntos porcentuales por vela
    minimum_distance_to_band_pct: float | None = 0.50
    minimum_first_minute_reversal_pct: float | None = 0.10
    maximum_post_open_bb_expansion_pct: float | None = None


DEFAULT_THRESHOLDS = ScannerThresholds()

# Pendiente provisional de la banda superior entre 09:30 y 09:35 ET.
MAX_5MIN_BAND_SLOPE_PCT_PER_MIN = 0.15

# Ángulo normalizado a cinco minutos de la banda superior provisional.
MAX_UPPER_ANGLE_DEG = 25.0

# Peso relativo del parecido visual. No equivale a probabilidad ni retorno esperado.
SCORE_WEIGHTS = {
    "gap": 1.0,
    "distance": 2.0,
    "compression": 1.0,
    "prior_width_growth": 1.0,
    "target_slope": 0.0,
    "target_angle": 5.0,
    "first_minute_reversal": 0.25,
    "post_open_expansion": 0.75,
}

SCORE_SCALES = {
    "gap_pct": 3.0,
    "distance_pct": 2.0,
    "bb_width_pct": 2.0,
    "prior_width_growth_pp_per_bar": 0.05,
    "target_slope_pct_per_bar": 0.05,
    "first_minute_reversal_pct": 0.25,
    "post_open_expansion_pct": 25.0,
}

# Ajuste exploratorio del score subminuto: sólo penaliza expansión muy marcada.
SCORE_CUTOFF_WIDTH_RATIO_GRACE = 2.0
SCORE_CUTOFF_WIDTH_PENALTY_PER_RATIO = 10.0
SCORE_CUTOFF_WIDTH_PENALTY_MAX = 30.0
