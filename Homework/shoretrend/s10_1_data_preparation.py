"""
ShoreTrend - Part 1 (S10): preparació de dades.
Transforma el CSV en brut (date, transect_id, distance_m) en una sèrie temporal
neta i mostrejada regularment de la distància mitjana de la línia de costa.

Ús:
    python Homework/shoretrend/s10_1_data_preparation.py
    python Homework/shoretrend/s10_1_data_preparation.py -i data/altre.csv -o data/obs_2025.csv
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as c

PART = 'part1'

# Paràmetres de neteja (justificats al report)
N_TRANSECTS = 100           # mida de la regió espacial (transectes contigus)
MAX_NAN_TRANSECT = 0.30     # descartar transectes amb més d'un 30% de NaN
IQR_FACTOR = 1.5            # outliers fora de [Q1 - 1.5·IQR, Q3 + 1.5·IQR]
MAX_NAN_DATE = 0.50         # descartar dates amb més d'un 50% de NaN als transectes vàlids
RESAMPLE_OPTIONS = {'W': 'setmanal', '2W': 'bisetmanal', 'MS': 'mensual'}
FREQ = 'MS'                 # freqüència escollida


def build_series(pivot, transects, bounds, max_nan_date):
    """Aplica transectes, límits IQR i filtre de dates a una taula pivot -> sèrie mitjana."""
    sub = pivot.reindex(columns=transects)
    sub = sub.where((sub >= bounds['low']) & (sub <= bounds['high']))
    nan_frac = sub.isna().mean(axis=1)
    sub = sub[nan_frac <= max_nan_date]
    return sub, sub.mean(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', default=c.HISTORY_CSV, help='CSV històric 2017-2024')
    parser.add_argument('-o', '--observed', default=c.OBSERVED_CSV,
                        help='CSV observat gener-juny 2025 (opcional; si no existeix es busquen dates 2025 al CSV històric)')
    parser.add_argument('--start-transect', type=int, default=None,
                        help='Primer transecte de la regió (per defecte: la finestra amb menys NaN)')
    parser.add_argument('--freq', default=FREQ, choices=list(RESAMPLE_OPTIONS))
    args = parser.parse_args()

    #########################################################################
    # 1-3. CARREGAR I INSPECCIONAR
    #########################################################################
    print("\n=== 1. Carregar dades ===")
    if not os.path.exists(args.input):
        raise SystemExit(f"No s'ha trobat {args.input}. Copia el CSV a {c.DATA_DIR} o passa'l amb -i.")
    raw = c.load_raw(args.input)

    print(f"\n--- head ---\n{raw.head()}")
    print(f"\n--- tail ---\n{raw.tail()}")
    print("\n--- info ---")
    raw.info()
    print(f"\nDates: {raw['date'].nunique()} ({raw['date'].min().date()} -> {raw['date'].max().date()})")
    print(f"Transectes: {raw['transect_id'].nunique()} ({raw['transect_id'].min()} -> {raw['transect_id'].max()})")

    # Si el CSV conté 2025, es separa: l'històric només fins a 2024
    history = raw[raw['date'] <= c.HISTORY_END]
    observed_raw = raw[(raw['date'] >= c.FORECAST_START) & (raw['date'] <= c.FORECAST_END)]
    if os.path.exists(args.observed):
        print(f"\nObservacions 2025 llegides de {args.observed}")
        observed_raw = c.load_raw(args.observed)
        observed_raw = observed_raw[(observed_raw['date'] >= c.FORECAST_START) & (observed_raw['date'] <= c.FORECAST_END)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(history['date'], history['distance_m'], s=1, alpha=0.05, c='C0', label='Valors per transecte')
    ax.plot(history.groupby('date')['distance_m'].mean(), c='C3', lw=1, label='Mitjana per data')
    ax.set(title='Dades en brut', xlabel='Data', ylabel='Distància (m)')
    ax.legend(); ax.grid(alpha=0.3)
    c.save_fig(fig, PART, '01_raw_data.png')

    #########################################################################
    # 4. TAULA PIVOT (date x transect_id)
    #########################################################################
    print("\n=== 4. Taula pivot ===")
    pivot = history.pivot_table(index='date', columns='transect_id', values='distance_m', aggfunc='mean').sort_index()
    print(f"Forma: {pivot.shape} (dates x transectes)")

    #########################################################################
    # 5. DADES QUE FALTEN
    #########################################################################
    print("\n=== 5. Dades que falten ===")
    nan_transect = pivot.isna().mean() * 100
    nan_date = pivot.isna().mean(axis=1) * 100
    print(f"NaN total: {pivot.isna().values.mean() * 100:.1f}%")
    print(f"NaN per transecte: mitjana {nan_transect.mean():.1f}%, mín {nan_transect.min():.1f}%, màx {nan_transect.max():.1f}%")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(nan_transect.index, nan_transect.values, lw=1)
    axes[0].set(title='% NaN per transecte', xlabel='transect_id', ylabel='% NaN')
    axes[1].hist(nan_transect.values, bins=40, edgecolor='black', alpha=0.8)
    axes[1].set(title='Distribució del % NaN per transecte', xlabel='% NaN', ylabel='Nombre de transectes')
    axes[2].imshow(pivot.isna().T.values, aspect='auto', cmap='Greys', interpolation='nearest',
                   extent=[0, len(pivot), pivot.columns.max(), pivot.columns.min()])
    ticks = np.linspace(0, len(pivot) - 1, 8).astype(int)
    axes[2].set_xticks(ticks, [pivot.index[i].strftime('%Y-%m') for i in ticks])
    axes[2].set(title='Mapa de NaN (negre = falta)', xlabel='Data (ordre de mostra)', ylabel='transect_id')
    for ax in axes[:2]:
        ax.grid(alpha=0.3)
    c.save_fig(fig, PART, '02_missing_data.png')

    nan_month = nan_date.groupby(nan_date.index.month).mean()
    print("NaN mitjà per mes de l'any (%): " + ', '.join(f"{m}:{v:.0f}" for m, v in nan_month.items()))

    #########################################################################
    # 6. SELECCIÓ DE LA REGIÓ (finestra contigua amb menys NaN)
    #########################################################################
    print("\n=== 6. Selecció de la regió ===")
    ids = nan_transect.sort_index()
    window = min(N_TRANSECTS, len(ids))
    if args.start_transect is not None:
        start_pos = int(ids.index.get_indexer([args.start_transect])[0])
        if start_pos < 0:
            raise SystemExit(f"El transecte {args.start_transect} no existeix")
    else:
        rolling = ids.rolling(window).mean().dropna()
        start_pos = ids.index.get_loc(rolling.idxmin()) - window + 1
    region = list(ids.index[start_pos:start_pos + window])
    print(f"Regió: transectes {region[0]}-{region[-1]} ({len(region)}), NaN mitjà {ids.loc[region].mean():.1f}%")

    #########################################################################
    # 7. FILTRAR TRANSECTES PROBLEMÀTICS
    #########################################################################
    print("\n=== 7. Filtrar transectes (columnes) ===")
    valid = [t for t in region if nan_transect[t] <= MAX_NAN_TRANSECT * 100]
    print(f"Transectes amb ≤{MAX_NAN_TRANSECT:.0%} NaN: {len(valid)} de {len(region)}")
    if not valid:
        raise SystemExit("Cap transecte supera el filtre de NaN; augmenta MAX_NAN_TRANSECT")

    #########################################################################
    # 8. OUTLIERS PER TRANSECTE (IQR)
    #########################################################################
    print("\n=== 8. Outliers (IQR per transecte) ===")
    sub = pivot[valid]
    q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
    iqr = q3 - q1
    bounds = pd.DataFrame({'low': q1 - IQR_FACTOR * iqr, 'high': q3 + IQR_FACTOR * iqr})
    n_before = int(sub.notna().values.sum())
    clean = sub.where((sub >= bounds['low']) & (sub <= bounds['high']))
    n_out = n_before - int(clean.notna().values.sum())
    print(f"Outliers eliminats: {n_out} de {n_before} valors ({n_out / max(n_before, 1) * 100:.1f}%)")

    #########################################################################
    # 9. FILTRAR DATES AMB POCA COBERTURA
    #########################################################################
    print("\n=== 9. Filtrar dates (files) ===")
    clean, series = build_series(pivot, valid, bounds, MAX_NAN_DATE)
    print(f"Dates conservades: {len(clean)} de {len(pivot)} (≤{MAX_NAN_DATE:.0%} NaN)")

    #########################################################################
    # 10. SÈRIE AGREGADA (mitjana entre transectes)
    #########################################################################
    print("\n=== 10. Sèrie agregada ===")
    print(series.describe())

    #########################################################################
    # 11. EXPLORAR REMOSTREIG
    #########################################################################
    print("\n=== 11. Remostreig ===")
    print(f"{'freq':>4} {'nom':>11} {'mostres':>8} {'buides':>8} {'std (m)':>8} {'autocorr1':>9}")
    fig, axes = plt.subplots(len(RESAMPLE_OPTIONS), 1, figsize=(12, 9), sharex=True)
    resampling_table = []
    for ax, (freq, name) in zip(axes, RESAMPLE_OPTIONS.items()):
        rs = series.resample(freq).mean()
        empty = rs.isna().mean() * 100
        ac1 = rs.interpolate().autocorr(lag=1)
        resampling_table.append({'freq': freq, 'name': name, 'samples': len(rs), 'empty_pct': empty,
                                 'std': rs.std(), 'autocorr_lag1': ac1})
        print(f"{freq:>4} {name:>11} {len(rs):8d} {empty:7.1f}% {rs.std():8.2f} {ac1:9.2f}")
        ax.plot(series.index, series.values, '.', ms=2, alpha=0.4, c='C0', label='Sèrie per data')
        ax.plot(rs.index, rs.values, '-o', ms=3, c='C3', label=f'{name} ({empty:.0f}% buides)')
        ax.set(ylabel='Distància (m)'); ax.legend(loc='upper right'); ax.grid(alpha=0.3)
    axes[0].set_title('Comparació de freqüències de remostreig')
    axes[-1].set_xlabel('Data')
    c.save_fig(fig, PART, '03_resampling.png')
    pd.DataFrame(resampling_table).to_csv(os.path.join(c.part_dir(PART), 'resampling_comparison.csv'), index=False)
    print(f"Freqüència escollida: {args.freq} ({RESAMPLE_OPTIONS[args.freq]})")

    #########################################################################
    # 12. INTERPOLACIÓ LINEAL
    #########################################################################
    print("\n=== 12. Interpolació ===")
    ts = series.resample(args.freq).mean()
    gaps = ts.isna()
    print(f"Intervals interpolats: {gaps.sum()} de {len(ts)}")
    run = (gaps != gaps.shift()).cumsum()
    longest = int(gaps.groupby(run).sum().max()) if gaps.any() else 0
    print(f"Buit consecutiu més llarg: {longest} intervals")
    ts_filled = ts.interpolate(method='linear', limit_direction='both')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts_filled.index, ts_filled.values, c='C0', label='Sèrie final (interpolada)')
    ax.plot(ts.index, ts.values, 'o', ms=3, c='C0', label='Valors amb observacions')
    ax.plot(ts_filled.index[gaps], ts_filled.values[gaps], 'x', c='C3', label='Valors interpolats')
    ax.set(title=f'Sèrie temporal final ({RESAMPLE_OPTIONS[args.freq]})', xlabel='Data', ylabel='Distància (m)')
    ax.legend(); ax.grid(alpha=0.3)
    c.save_fig(fig, PART, '04_final_timeseries.png')

    #########################################################################
    # 13. GUARDAR
    #########################################################################
    print("\n=== 13. Guardar ===")
    os.makedirs(os.path.dirname(c.TIMESERIES_CSV), exist_ok=True)
    out = pd.DataFrame({'distance_m': ts_filled, 'interpolated': gaps})
    out.index.name = 'date'
    out.to_csv(c.TIMESERIES_CSV)
    print(f"Sèrie temporal: {c.TIMESERIES_CSV}")

    # Observacions gener-juny 2025 amb els mateixos transectes i límits (sense interpolar)
    has_observed = not observed_raw.empty
    if has_observed:
        obs_pivot = observed_raw.pivot_table(index='date', columns='transect_id', values='distance_m', aggfunc='mean')
        missing_ids = sorted(set(valid) - set(obs_pivot.columns))
        if missing_ids:
            print(f"Avís: {len(missing_ids)} transectes de la regió no apareixen al 2025")
        _, obs_series = build_series(obs_pivot, valid, bounds, MAX_NAN_DATE)
        obs_ts = obs_series.resample(args.freq).mean()
        obs_ts = obs_ts.reindex(pd.date_range(c.FORECAST_START, c.FORECAST_END, freq=args.freq))
        obs_ts.index.name = 'date'
        obs_ts.rename('distance_m').to_csv(c.OBSERVED_TS_CSV)
        print(f"Observacions 2025: {c.OBSERVED_TS_CSV} ({obs_ts.notna().sum()} de {len(obs_ts)} intervals amb dades)")
    else:
        print("Avís: no hi ha dades de gener-juny 2025; la Part 3 no podrà validar les prediccions.")

    c.save_json({
        'input': os.path.abspath(args.input),
        'freq': args.freq,
        'steps_per_year': c.STEPS_PER_YEAR[args.freq],
        'region': [int(region[0]), int(region[-1])],
        'valid_transects': [int(t) for t in valid],
        'n_dates_kept': int(len(clean)),
        'outliers_removed': n_out,
        'interpolated_steps': int(gaps.sum()),
        'longest_gap_steps': longest,
        'has_observed_2025': has_observed,
        'thresholds': {'max_nan_transect': MAX_NAN_TRANSECT, 'iqr_factor': IQR_FACTOR, 'max_nan_date': MAX_NAN_DATE},
    }, c.METADATA_JSON)
    print(f"Metadades: {c.METADATA_JSON}")


if __name__ == '__main__':
    main()
