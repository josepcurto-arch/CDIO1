"""
ShoreTrend - Part 2 (S10): anàlisi de la sèrie temporal.
Estadística, tendència + estacionalitat amb tres models paramètrics
(lineal / salt / sigmoide + FFT), residus, velocitat i acceleració.

Ús:
    python Homework/shoretrend/s10_2_data_analysis.py
    python Homework/shoretrend/s10_2_data_analysis.py --model sigmoid   # forçar el model
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot

import common as c

PART = 'part2'
VALIDATION_YEAR = '2024'   # any reservat per comparar els models fora de mostra
ROLLING_MONTHS = 12        # finestra per suavitzar la velocitat/acceleració observades


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=list(c.MODEL_NAMES), default=None,
                        help='Model a fer servir a la Part 3 (per defecte: el de menor RMSE de validació)')
    parser.add_argument('--n-freq', type=int, default=c.N_FREQ, help='Freqüències FFT a conservar')
    args = parser.parse_args()

    #########################################################################
    # 1. CARREGAR
    #########################################################################
    print("\n=== 1. Carregar sèrie temporal ===")
    if not os.path.exists(c.TIMESERIES_CSV):
        raise SystemExit("Executa primer s10_1_data_preparation.py")
    meta = c.load_json(c.METADATA_JSON)
    spy = meta['steps_per_year']
    ts = c.load_series(c.TIMESERIES_CSV).asfreq(meta['freq'])
    print(f"{len(ts)} mostres ({meta['freq']}), {ts.index[0].date()} -> {ts.index[-1].date()}")

    #########################################################################
    # 2. VISUALITZACIÓ
    #########################################################################
    print("\n=== 2. Sèrie completa ===")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts.index, ts.values, '-o', ms=3)
    ax.axvline(pd.Timestamp(c.GLORIA_DATE), c='C3', ls='--', label='Temporal Gloria')
    ax.set(title='Distància mitjana de la línia de costa', xlabel='Data', ylabel='Distància (m)')
    ax.legend(); ax.grid(alpha=0.3)
    c.save_fig(fig, PART, '01_timeseries.png')

    #########################################################################
    # 3-4. ESTADÍSTICA DESCRIPTIVA I BOXPLOTS
    #########################################################################
    print("\n=== 3. Estadística descriptiva ===")
    stats = ts.describe()
    stats['skew'] = ts.skew()
    stats['kurtosis'] = ts.kurt()
    print(stats.round(3))
    yearly = ts.groupby(ts.index.year).agg(['mean', 'std', 'min', 'max'])
    print(f"\nPer any:\n{yearly.round(2)}")
    stats.to_csv(os.path.join(c.part_dir(PART), 'descriptive_stats.csv'), header=['value'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    df = pd.DataFrame({'d': ts.values, 'month': ts.index.month, 'year': ts.index.year})
    df.boxplot(column='d', by='month', ax=axes[0])
    axes[0].set(title='Per mes', xlabel='Mes', ylabel='Distància (m)')
    df.boxplot(column='d', by='year', ax=axes[1])
    axes[1].set(title='Per any', xlabel='Any', ylabel='Distància (m)')
    fig.suptitle('')
    c.save_fig(fig, PART, '02_boxplots.png')

    #########################################################################
    # 5-8. TENDÈNCIA + ESTACIONALITAT: TRES MODELS
    #########################################################################
    print("\n=== 5-8. Models paramètrics (tendència + FFT) ===")
    train = ts[:str(int(VALIDATION_YEAR) - 1)]
    valid = ts[VALIDATION_YEAR]
    models, rows = {}, []
    for kind, name in c.MODEL_NAMES.items():
        model = c.ParametricModel(kind, args.n_freq).fit(ts)
        models[kind] = model
        fit_metrics = c.metrics(ts, model.fitted)

        # Validació fora de mostra: ajust fins a l'any anterior i predicció del darrer any
        val_model = c.ParametricModel(kind, args.n_freq).fit(train)
        val_metrics = c.metrics(valid, val_model.forecast(len(valid)).values)

        rows.append({'model': kind, 'name': name,
                     'RMSE': fit_metrics['RMSE'], 'MAE': fit_metrics['MAE'], 'R2': fit_metrics['R2'],
                     'AIC': model.aic(), 'n_params': model.n_params,
                     f'val_RMSE_{VALIDATION_YEAR}': val_metrics['RMSE'],
                     f'val_MAE_{VALIDATION_YEAR}': val_metrics['MAE']})
        print(f"\n{name}")
        print(model.describe(spy))
        print(f"  Ajust: RMSE {fit_metrics['RMSE']:.3f} m, MAE {fit_metrics['MAE']:.3f} m, R² {fit_metrics['R2']:.3f}, AIC {model.aic():.1f}")
        print(f"  Validació {VALIDATION_YEAR}: RMSE {val_metrics['RMSE']:.3f} m, MAE {val_metrics['MAE']:.3f} m")

    comparison = pd.DataFrame(rows).set_index('model')
    comparison.to_csv(os.path.join(c.part_dir(PART), 'model_comparison.csv'))
    print(f"\n{comparison.drop(columns='name').round(3)}")

    val_col = f'val_RMSE_{VALIDATION_YEAR}'
    best = args.model or comparison[val_col].idxmin()
    reason = 'forçat per --model' if args.model else f'menor RMSE de validació {VALIDATION_YEAR}'
    print(f"\nModel escollit: {c.MODEL_NAMES[best]} ({reason})")
    c.save_json({'model': best, 'n_freq': args.n_freq, 'reason': reason,
                 'comparison': comparison.drop(columns='name').round(4).to_dict(orient='index')},
                c.BEST_MODEL_JSON)

    # Espectre (model escollit) per justificar els períodes dominants
    freqs, mag = models[best].spectrum
    fig, ax = plt.subplots(figsize=(12, 4))
    periods = np.divide(12, freqs * spy, out=np.full_like(freqs, np.nan), where=freqs > 0)
    ax.stem(periods[1:], mag[1:])
    for f in models[best].fft_freqs:
        ax.axvline(12 / (f * spy), c='C3', ls='--', alpha=0.6)
    ax.set(xscale='log', title=f'Espectre FFT del residu sense tendència ({best})', xlabel='Període (mesos)', ylabel='|FFT|')
    ax.grid(alpha=0.3)
    c.save_fig(fig, PART, '03_fft_spectrum.png')

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for ax, (kind, model) in zip(axes, models.items()):
        t = np.arange(len(ts))
        m = comparison.loc[kind]
        ax.plot(ts.index, ts.values, 'o', ms=3, alpha=0.5, c='C0', label='Original')
        ax.plot(ts.index, model.trend(t), c='C3', label='Tendència')
        ax.plot(ts.index, model.cyclical(t) + model.trend(t).mean(), c='C2', alpha=0.7, label='Cicles (desplaçats)')
        ax.plot(ts.index, model.fitted, c='C0', lw=2, label='Reconstruïda')
        ax.set(title=f"{c.MODEL_NAMES[kind]}  |  RMSE {m.RMSE:.2f} m, R² {m.R2:.3f}, val. {m[val_col]:.2f} m",
               ylabel='Distància (m)')
        ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Data')
    c.save_fig(fig, PART, '04_models_reconstruction.png')

    #########################################################################
    # 9. RESIDUS (model escollit)
    #########################################################################
    print("\n=== 9. Residus ===")
    res = pd.Series(models[best].residuals, index=ts.index)
    print(f"  Mitjana {res.mean():.3f} m, std {res.std():.3f} m, mín {res.min():.3f} m, màx {res.max():.3f} m")
    print(f"  Autocorrelació lag 1: {res.autocorr(1):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(res.index, res.values, 'o', ms=3)
    axes[0].axhline(0, c='k', lw=0.8)
    axes[0].set(title='Residus en el temps', xlabel='Data', ylabel='m')
    axes[1].hist(res.values, bins=25, edgecolor='black', alpha=0.8)
    axes[1].axvline(res.mean(), c='C3', ls='--', label=f'Mitjana {res.mean():.2f} m')
    axes[1].set(title='Distribució dels residus', xlabel='m'); axes[1].legend()
    autocorrelation_plot(res, ax=axes[2])
    axes[2].set(title='Autocorrelació dels residus')
    for ax in axes:
        ax.grid(alpha=0.3)
    c.save_fig(fig, PART, '05_residuals.png')

    #########################################################################
    # 10-11. DINÀMICA: VELOCITAT I ACCELERACIÓ
    #########################################################################
    print("\n=== 10-11. Velocitat i acceleració ===")
    dt = 12 / spy                                   # mesos entre mostres
    reconstructed = models[best].fitted
    velocity = np.gradient(reconstructed, dt)       # m/mes
    acceleration = np.gradient(velocity, dt)        # m/mes²

    # Sèrie observada suavitzada (mitjana mòbil d'un any) per veure la dinàmica sense cicles
    window = int(round(ROLLING_MONTHS / dt))
    smooth = ts.rolling(window, center=True).mean()
    v_smooth = pd.Series(np.gradient(smooth.values, dt), index=ts.index)
    a_smooth = pd.Series(np.gradient(v_smooth.values, dt), index=ts.index)

    trend_v = np.gradient(models[best].trend(np.arange(len(ts))), dt) * 12
    print(f"  Velocitat de la tendència: mitjana {trend_v.mean():+.3f} m/any")
    print(f"  Velocitat model: mitjana {velocity.mean() * 12:+.3f} m/any, rang [{velocity.min() * 12:+.2f}, {velocity.max() * 12:+.2f}] m/any")
    print(f"  Velocitat suavitzada (12 mesos): mitjana {v_smooth.mean() * 12:+.3f} m/any")

    yearly_dyn = pd.DataFrame({'v_m_year': v_smooth * 12, 'a_m_year2': a_smooth * 144}).groupby(ts.index.year).mean()
    print(f"\n  Mitjana anual de la dinàmica suavitzada:\n{yearly_dyn.round(3)}")
    yearly_dyn.to_csv(os.path.join(c.part_dir(PART), 'velocity_acceleration_yearly.csv'))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(ts.index, ts.values, 'o', ms=3, alpha=0.4, label='Sèrie')
    axes[0].plot(ts.index, reconstructed, label='Model')
    axes[0].plot(ts.index, smooth, c='k', lw=2, label=f'Mitjana mòbil {ROLLING_MONTHS} mesos')
    axes[0].set(ylabel='Distància (m)')
    axes[1].plot(ts.index, velocity * 12, alpha=0.5, label='Model')
    axes[1].plot(ts.index, v_smooth * 12, c='k', lw=2, label='Suavitzada')
    axes[1].fill_between(ts.index, 0, v_smooth * 12, where=v_smooth > 0, color='C2', alpha=0.3, label='v > 0')
    axes[1].fill_between(ts.index, 0, v_smooth * 12, where=v_smooth < 0, color='C3', alpha=0.3, label='v < 0')
    axes[1].set(ylabel='Velocitat (m/any)')
    axes[2].plot(ts.index, acceleration * 144, alpha=0.5, label='Model')
    axes[2].plot(ts.index, a_smooth * 144, c='k', lw=2, label='Suavitzada')
    axes[2].set(ylabel='Acceleració (m/any²)', xlabel='Data')
    for ax in axes:
        ax.axvline(pd.Timestamp(c.GLORIA_DATE), c='C3', ls='--', alpha=0.6)
        ax.axhline(0, c='grey', lw=0.8)
        ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
    axes[0].set_title('Dinàmica de la línia de costa (línia vermella: Gloria)')
    c.save_fig(fig, PART, '06_velocity_acceleration.png')

    dyn = pd.DataFrame({'distance_m': ts, 'reconstructed_m': reconstructed,
                        'velocity_m_month': velocity, 'acceleration_m_month2': acceleration,
                        'smooth_m': smooth, 'smooth_velocity_m_month': v_smooth, 'smooth_acceleration_m_month2': a_smooth})
    dyn.to_csv(os.path.join(c.part_dir(PART), 'dynamics.csv'))


if __name__ == '__main__':
    main()
