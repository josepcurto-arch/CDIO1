"""
ShoreTrend - funcions i configuració compartides per les tres parts
(S10 preparació + anàlisi, S12 predicció) de l'erosió de Castelldefels.
"""
import json
import os

import matplotlib
# Backend no interactiu: totes les figures es guarden a output/
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------------------
# RUTES
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

HISTORY_CSV = os.path.join(DATA_DIR, 'shoreline_distances_2017_2024.csv')
OBSERVED_CSV = os.path.join(DATA_DIR, 'shoreline_distances_2025.csv')

TIMESERIES_CSV = os.path.join(OUTPUT_DIR, 'part1', 'shoreline_timeseries.csv')
OBSERVED_TS_CSV = os.path.join(OUTPUT_DIR, 'part1', 'shoreline_observed_2025.csv')
METADATA_JSON = os.path.join(OUTPUT_DIR, 'part1', 'metadata.json')
BEST_MODEL_JSON = os.path.join(OUTPUT_DIR, 'part2', 'best_model.json')

# ------------------------------------------------------------------------------
# PARÀMETRES
# ------------------------------------------------------------------------------
HISTORY_END = '2024-12-31'
FORECAST_START = '2025-01-01'
FORECAST_END = '2025-06-30'

GLORIA_DATE = '2020-02-01'   # primer mes després del temporal Gloria (gener 2020)
N_FREQ = 2                   # nombre de freqüències FFT que es conserven
Z_95 = 1.96

STEPS_PER_YEAR = {'W': 52, '2W': 26, 'MS': 12}
MODEL_NAMES = {
    'linear': 'a) tendència lineal + FFT',
    'step': 'b) tendència lineal + salt (Heaviside) + FFT',
    'sigmoid': 'c) tendència lineal + sigmoide + FFT',
}


# ------------------------------------------------------------------------------
# UTILITATS
# ------------------------------------------------------------------------------
def part_dir(part):
    path = os.path.join(OUTPUT_DIR, part)
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig, part, name):
    path = os.path.join(part_dir(part), name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> figura: {os.path.relpath(path, BASE_DIR)}")


def load_raw(path):
    """Carrega un CSV amb columnes date, transect_id, distance_m (la resta s'ignora)."""
    df = pd.read_csv(path, usecols=['date', 'transect_id', 'distance_m'])
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_series(path):
    df = pd.read_csv(path, parse_dates=['date'])
    return df.set_index('date')['distance_m']


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2),
        'n': int(len(y_true)),
    }


# ------------------------------------------------------------------------------
# MODEL PARAMÈTRIC: TENDÈNCIA (lineal / salt / sigmoide) + CICLES (FFT)
# ------------------------------------------------------------------------------
def _sigmoid_trend(t, a, b, c, k, t0):
    # clip evita overflow d'exp quan k és gran (transició gairebé instantània)
    return a + b * t + c / (1 + np.exp(np.clip(-k * (t - t0), -500, 500)))


class ParametricModel:
    """y(t) = tendència(t) + Σ cicles FFT(t). t = índex de mostra (0, 1, 2, ...)."""

    def __init__(self, kind, n_freq=N_FREQ):
        if kind not in MODEL_NAMES:
            raise ValueError(f"Model desconegut: {kind}")
        self.kind = kind
        self.n_freq = n_freq

    def fit(self, ts, jump_date=GLORIA_DATE):
        y = ts.values.astype(float)
        self.n = len(y)
        self.start = ts.index[0]
        self.freq = ts.index.freq or pd.infer_freq(ts.index)
        t = np.arange(self.n)
        self.t_jump = int(ts.index.searchsorted(pd.Timestamp(jump_date)))

        # 1. Tendència
        if self.kind == 'sigmoid':
            p0 = [y[:self.t_jump].mean(), 0.0, y[self.t_jump:].mean() - y[:self.t_jump].mean(), 1.0, self.t_jump]
            lower = [-np.inf, -np.inf, -np.inf, 0.05, self.t_jump - 12]
            upper = [np.inf, np.inf, np.inf, 20.0, self.t_jump + 12]
            self.params, _ = curve_fit(_sigmoid_trend, t, y, p0=p0, bounds=(lower, upper), maxfev=20000)
        else:
            self.reg = LinearRegression().fit(self._design(t), y)
        trend = self.trend(t)

        # 2. Cicles: FFT del residu sense tendència, ens quedem amb les N freqüències més fortes
        detrended = y - trend
        spectrum = rfft(detrended)
        freqs = rfftfreq(self.n)
        magnitude = np.abs(spectrum)
        magnitude[0] = 0  # la component contínua ja és a la tendència
        top = np.argsort(magnitude)[::-1][:self.n_freq]
        self.fft_freqs = freqs[top]
        nyquist = np.isclose(self.fft_freqs, 0.5)
        self.fft_amps = np.abs(spectrum[top]) / self.n * np.where(nyquist, 1, 2)
        self.fft_phases = np.angle(spectrum[top])
        self.spectrum = (freqs, np.abs(spectrum))

        self.fitted = self.predict(t)
        self.residuals = y - self.fitted
        self.sigma = float(np.std(self.residuals))
        return self

    def _design(self, t):
        t = np.asarray(t, dtype=float)
        if self.kind == 'linear':
            return t.reshape(-1, 1)
        return np.column_stack([t, (t >= self.t_jump).astype(float)])

    def trend(self, t):
        t = np.asarray(t, dtype=float)
        if self.kind == 'sigmoid':
            return _sigmoid_trend(t, *self.params)
        return self.reg.predict(self._design(t))

    def cyclical(self, t):
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        for f, a, p in zip(self.fft_freqs, self.fft_amps, self.fft_phases):
            out += a * np.cos(2 * np.pi * f * t + p)
        return out

    def predict(self, t):
        return self.trend(t) + self.cyclical(t)

    def forecast(self, n_future):
        """Prediu els n_future passos següents al final de la sèrie d'entrenament."""
        t_future = np.arange(self.n, self.n + n_future)
        index = pd.date_range(self.start, periods=self.n + n_future, freq=self.freq)[self.n:]
        return pd.Series(self.predict(t_future), index=index)

    def confidence_interval(self, n_future, z=Z_95):
        """Interval del 95% basat en els residus, creixent amb l'horitzó (com a s11_1)."""
        h = np.arange(1, n_future + 1)
        return z * self.sigma * np.sqrt(1 + h / self.n)

    @property
    def n_params(self):
        base = {'linear': 2, 'step': 3, 'sigmoid': 5}[self.kind]
        return base + 3 * self.n_freq  # freqüència, amplitud i fase per cicle

    def aic(self):
        rss = float(np.sum(self.residuals ** 2))
        return self.n * np.log(rss / self.n) + 2 * self.n_params

    def describe(self, steps_per_year):
        lines = []
        if self.kind == 'sigmoid':
            a, b, c, k, t0 = self.params
            t0_date = pd.date_range(self.start, periods=int(round(t0)) + 1, freq=self.freq)[-1]
            lines.append(f"  Pendent: {b * steps_per_year:+.3f} m/any")
            lines.append(f"  Canvi sigmoide (β): {c:+.2f} m, k={k:.2f}, t0≈{t0_date.date()}")
        else:
            lines.append(f"  Pendent: {self.reg.coef_[0] * steps_per_year:+.3f} m/any")
            if self.kind == 'step':
                lines.append(f"  Salt Gloria: {self.reg.coef_[1]:+.2f} m")
        for f, a in zip(self.fft_freqs, self.fft_amps):
            lines.append(f"  Cicle: període {1 / f / steps_per_year * 12:5.1f} mesos, amplitud {a:.2f} m")
        return '\n'.join(lines)
