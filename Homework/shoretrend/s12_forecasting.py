"""
ShoreTrend - Part 3 (S11 + S12): predicció gener-juny 2025.
Model A: el millor model paramètric de la Part 2 (amb IC del 95%).
Models B: xarxes recurrents (Keras) amb les tres estratègies de la S11
(un pas, autoregressiva, one-shot) i les millores proposades (GRU, dues capes,
EarlyStopping, intervals de predicció).
Tots es comparen amb les observacions reals de 2025, amb comprovació d'overfitting.

Ús:
    python Homework/shoretrend/s12_forecasting.py
    python Homework/shoretrend/s12_forecasting.py --retrain      # tornar a entrenar les xarxes
"""
import argparse
import os

# Keras 3 necessita un backend; TensorFlow no existeix per a Python 3.14, fem servir PyTorch
os.environ.setdefault('KERAS_BACKEND', 'torch')

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Input
from keras.callbacks import EarlyStopping
from keras.layers import GRU, LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

import common as c

PART = 'part3'
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

# CONFIGURACIÓ DE LES XARXES (sèrie mensual curta: ~96 mostres)
WINDOW = 12              # mesos d'entrada
UNITS = 32
DROPOUT = 0.2            # només als models one-shot, com a s11_4
EPOCHS = 500             # màxim; EarlyStopping s'atura a la millor època
BATCH_SIZE = 8
PATIENCE = 40
VALIDATION_SPLIT = 0.2
SEED = 42
OVERFIT_GAP = 0.1        # criteri S11: R²_train - R²_test > 0.1 -> possible overfitting

# Arquitectures a entrenar: (cel·la, capes, horitzó)
ARCHITECTURES = {
    'lstm1_h1': ('lstm', 1, 1),     # base: un pas
    'lstm2_h1': ('lstm', 2, 1),     # millora: una capa LSTM extra
    'lstm1_h6': ('lstm', 1, 6),     # one-shot (multi-sortida)
    'gru1_h6': ('gru', 1, 6),       # millora: GRU one-shot
}

# Prediccions de 2025: nom -> (arquitectura, estratègia, descripció)
VARIANTS = {
    'lstm_onestep': ('lstm1_h1', 'onestep', 'LSTM un pas (amb observacions 2025)'),
    'lstm_autoreg': ('lstm1_h1', 'autoregressive', 'LSTM autoregressiu'),
    'lstm2_autoreg': ('lstm2_h1', 'autoregressive', 'LSTM 2 capes autoregressiu'),
    'lstm_oneshot': ('lstm1_h6', 'oneshot', 'LSTM one-shot'),
    'gru_oneshot': ('gru1_h6', 'oneshot', 'GRU one-shot'),
}


def create_sequences(data, window, horizon, stride=1):
    X, y = [], []
    for i in range(0, len(data) - window - horizon + 1, stride):
        X.append(data[i:i + window])
        y.append(data[i + window:i + window + horizon])
    return np.array(X), np.array(y)


def build_model(cell, layers, horizon):
    Layer = LSTM if cell == 'lstm' else GRU
    stack = [Input(shape=(WINDOW, 1))]
    for i in range(layers):
        stack.append(Layer(UNITS, return_sequences=i < layers - 1))
    if horizon > 1:
        stack.append(Dropout(DROPOUT))
    stack.append(Dense(horizon))
    model = Sequential(stack)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_or_load(name, cell, layers, horizon, train_scaled, model_dir, retrain):
    model_file = os.path.join(model_dir, f"{name}_u{UNITS}_w{WINDOW}_b{BATCH_SIZE}.keras")
    if os.path.exists(model_file) and not retrain:
        print(f"  [{name}] carregant {os.path.relpath(model_file, c.BASE_DIR)}")
        return load_model(model_file), None

    keras.utils.set_random_seed(SEED)
    X, y = create_sequences(train_scaled, WINDOW, horizon)
    y = y.squeeze(-1)
    model = build_model(cell, layers, horizon)
    # validation_split agafa les últimes mostres abans de barrejar: validació cronològica
    history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                        callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)], verbose=0)
    h = history.history
    best_epoch = int(np.argmin(h['val_loss']))
    print(f"  [{name}] {model.count_params()} paràmetres, {len(h['loss'])} èpoques, millor {best_epoch + 1}: "
          f"loss train {h['loss'][best_epoch]:.4f} | val {h['val_loss'][best_epoch]:.4f}")
    model.save(model_file)
    return model, h


def in_sample(model, scaler, train_scaled, horizon):
    """Prediccions sobre les finestres d'entrenament i validació, en metres."""
    X, y = create_sequences(train_scaled, WINDOW, horizon)
    pred = scaler.inverse_transform(model.predict(X, verbose=0).reshape(-1, 1)).reshape(len(X), horizon)
    true = scaler.inverse_transform(y.reshape(-1, 1)).reshape(len(X), horizon)
    n_val = int(np.ceil(len(X) * VALIDATION_SPLIT))
    return {
        'train': c.metrics(true[:-n_val].ravel(), pred[:-n_val].ravel()),
        'val': c.metrics(true[-n_val:].ravel(), pred[-n_val:].ravel()),
        'sigma_val': float(np.std(true[-n_val:] - pred[-n_val:])),
    }


def forecast(model, strategy, train_scaled, obs_scaled, n_future):
    window = list(train_scaled[-WINDOW:].ravel())
    if strategy == 'oneshot':
        return model.predict(np.array(window).reshape(1, WINDOW, 1), verbose=0)[0, :n_future]
    preds = []
    for i in range(n_future):
        p = float(model.predict(np.array(window[-WINDOW:]).reshape(1, WINDOW, 1), verbose=0)[0, 0])
        preds.append(p)
        # un pas: la finestra avança amb el valor observat; autoregressiu: amb la predicció
        use_obs = strategy == 'onestep' and not np.isnan(obs_scaled[i])
        window.append(obs_scaled[i] if use_obs else p)
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--retrain', action='store_true', help="Ignorar els models guardats i tornar a entrenar")
    args = parser.parse_args()

    #########################################################################
    # 1. CARREGAR
    #########################################################################
    print("\n=== 1. Carregar dades ===")
    for path, step in [(c.TIMESERIES_CSV, 's10_1_data_preparation.py'), (c.BEST_MODEL_JSON, 's10_2_data_analysis.py')]:
        if not os.path.exists(path):
            raise SystemExit(f"Executa primer {step}")
    meta = c.load_json(c.METADATA_JSON)
    best = c.load_json(c.BEST_MODEL_JSON)
    freq, spy = meta['freq'], meta['steps_per_year']

    train_ts = c.load_series(c.TIMESERIES_CSV).asfreq(freq)[:c.HISTORY_END]
    future_index = pd.date_range(c.FORECAST_START, c.FORECAST_END, freq=freq)
    n_future = len(future_index)
    observed = (c.load_series(c.OBSERVED_TS_CSV).reindex(future_index)
                if os.path.exists(c.OBSERVED_TS_CSV) else pd.Series(np.nan, index=future_index))
    has_obs = observed.notna().any()
    print(f"Entrenament: {len(train_ts)} mostres ({train_ts.index[0].date()} -> {train_ts.index[-1].date()})")
    print(f"A predir: {n_future} intervals; observats disponibles: {observed.notna().sum()}")

    forecasts, intervals, rows = {}, {}, []
    descriptions = {'parametric': f"Paramètric ({best['model']})"}

    #########################################################################
    # MODEL A: PARAMÈTRIC (tendència + cicles)
    #########################################################################
    print(f"\n=== Model A: {c.MODEL_NAMES[best['model']]} ===")
    model_a = c.ParametricModel(best['model'], best['n_freq']).fit(train_ts)
    print(model_a.describe(spy))
    pred_a = model_a.forecast(n_future)
    pred_a.index = future_index
    ci_a = model_a.confidence_interval(n_future)
    forecasts['parametric'], intervals['parametric'] = pred_a, ci_a
    print(f"  σ residus: {model_a.sigma:.3f} m; amplada mitjana IC95%: ±{ci_a.mean():.2f} m")
    val_2024 = best['comparison'][best['model']].get('val_RMSE_2024', np.nan)
    rows.append({'model': 'parametric', 'train': c.metrics(train_ts, model_a.fitted),
                 'val': {'RMSE': val_2024, 'R2': np.nan}, 'sigma': model_a.sigma})

    #########################################################################
    # MODELS B: XARXES RECURRENTS
    #########################################################################
    print("\n=== Models B: xarxes recurrents ===")
    model_dir = c.part_dir(os.path.join(PART, 'models'))
    scaler_file = os.path.join(model_dir, 'scaler.pkl')
    # Normalització ajustada només amb l'entrenament (el mateix scaler per a tots els models)
    scaler = MinMaxScaler().fit(train_ts.values.reshape(-1, 1))
    joblib.dump(scaler, scaler_file)
    train_scaled = scaler.transform(train_ts.values.reshape(-1, 1))
    obs_scaled = scaler.transform(observed.values.reshape(-1, 1)).ravel()

    trained, histories, fits = {}, {}, {}
    for name, (cell, layers, horizon) in ARCHITECTURES.items():
        trained[name], histories[name] = train_or_load(name, cell, layers, horizon, train_scaled, model_dir, args.retrain)
        fits[name] = in_sample(trained[name], scaler, train_scaled, horizon)

    for variant, (arch, strategy, desc) in VARIANTS.items():
        pred = forecast(trained[arch], strategy, train_scaled, obs_scaled, n_future)
        forecasts[variant] = pd.Series(scaler.inverse_transform(pred.reshape(-1, 1)).ravel(), index=future_index)
        # Interval residual com al model paramètric, amb σ dels residus de validació
        sigma = fits[arch]['sigma_val']
        horizon_steps = np.ones(n_future) if strategy == 'onestep' else np.arange(1, n_future + 1)
        intervals[variant] = c.Z_95 * sigma * np.sqrt(1 + horizon_steps / len(train_ts))
        descriptions[variant] = desc
        rows.append({'model': variant, 'train': fits[arch]['train'], 'val': fits[arch]['val'], 'sigma': sigma})

    #########################################################################
    # AVALUACIÓ: TRAIN vs TEST I OVERFITTING
    #########################################################################
    print("\n=== Avaluació gener-juny 2025 ===")
    table = pd.DataFrame({'observed_m': observed})
    for name, pred in forecasts.items():
        table[f'{name}_m'] = pred
        table[f'{name}_ci95_m'] = intervals[name]
    table.index.name = 'date'
    table.to_csv(os.path.join(c.part_dir(PART), 'forecast_2025.csv'))
    print(table[['observed_m'] + [f'{n}_m' for n in forecasts]].round(2))

    records = []
    for row in rows:
        name = row['model']
        test = c.metrics(observed, forecasts[name]) if has_obs else {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
        inside = (observed >= forecasts[name] - intervals[name]) & (observed <= forecasts[name] + intervals[name])
        gap = row['train']['R2'] - test['R2']
        records.append({
            'model': name, 'description': descriptions[name],
            'train_RMSE': row['train']['RMSE'], 'train_R2': row['train']['R2'],
            'val_RMSE': row['val']['RMSE'], 'val_R2': row['val']['R2'],
            'test_RMSE': test['RMSE'], 'test_MAE': test['MAE'], 'test_R2': test['R2'],
            'R2_gap_train_test': gap, 'overfitting_flag': bool(gap > OVERFIT_GAP),
            'sigma_m': row['sigma'], 'ci95_mean_m': float(intervals[name].mean()),
            'PICP_95': float(inside[observed.notna()].mean()) if has_obs else np.nan,
        })
    results = pd.DataFrame(records).set_index('model')
    results.to_csv(os.path.join(c.part_dir(PART), 'forecast_metrics.csv'))

    cols = ['train_RMSE', 'train_R2', 'val_RMSE', 'test_RMSE', 'test_MAE', 'test_R2', 'R2_gap_train_test', 'PICP_95']
    print(f"\n{results[cols].round(3)}")
    flagged = results.index[results['overfitting_flag']].tolist()
    print(f"\nCriteri S11 (R²_train - R²_test > {OVERFIT_GAP}): {', '.join(flagged) if flagged else 'cap model'} marcat")
    if has_obs:
        print("  Nota: R²_test es calcula amb 6 punts de poca variància; el RMSE/MAE són més fiables.")
        forecast_models = results.drop(index='lstm_onestep')   # el model d'un pas veu observacions de 2025
        winner = forecast_models['test_RMSE'].idxmin()
        print(f"Millor predicció real (sense un pas): {descriptions[winner]}, RMSE {forecast_models.loc[winner, 'test_RMSE']:.2f} m")

    best_dl = results.drop(index=['parametric', 'lstm_onestep'])['val_RMSE'].idxmin()
    c.save_json({'parametric_model': best['model'], 'best_dl_by_validation': best_dl,
                 'metrics': results.round(4).to_dict(orient='index'),
                 'config': {'window': WINDOW, 'units': UNITS, 'dropout_oneshot': DROPOUT, 'batch_size': BATCH_SIZE,
                            'epochs_max': EPOCHS, 'patience': PATIENCE, 'seed': SEED,
                            'backend': keras.backend.backend()}},
                os.path.join(c.part_dir(PART), 'summary.json'))

    #########################################################################
    # GRÀFIQUES
    #########################################################################
    trained_now = {n: h for n, h in histories.items() if h is not None}
    if trained_now:
        fig, axes = plt.subplots(1, len(trained_now), figsize=(4.5 * len(trained_now), 3.5), squeeze=False)
        for ax, (name, h) in zip(axes[0], trained_now.items()):
            ax.plot(h['loss'], label='Train')
            ax.plot(h['val_loss'], label='Validació')
            ax.axvline(int(np.argmin(h['val_loss'])), c='k', ls='--', alpha=0.5)
            ax.set(title=name, xlabel='Època', ylabel='MSE (normalitzat)', yscale='log')
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        c.save_fig(fig, PART, '01_dl_loss_curves.png')

    reconstruction = pd.Series(model_a.fitted, index=train_ts.index)
    for name, start, variants in [('02_forecast_full.png', None, ['parametric', best_dl]),
                                  ('03_forecast_zoom.png', '2023-01-01', list(forecasts))]:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(train_ts[start:].index, train_ts[start:].values, 'o-', ms=3, alpha=0.5, c='C0', label='Històric 2017-2024')
        ax.plot(reconstruction[start:].index, reconstruction[start:].values, c='C0', lw=1.5, alpha=0.8,
                label='Reconstrucció paramètrica')
        for i, v in enumerate(variants):
            color = 'C3' if v == 'parametric' else f'C{(i % 6) + 4}' if start else 'C1'
            ax.plot(future_index, forecasts[v].values, 'o-', c=color, lw=2, label=descriptions[v])
            if v in ('parametric', best_dl):
                ax.fill_between(future_index, forecasts[v] - intervals[v], forecasts[v] + intervals[v],
                                color=color, alpha=0.15, label=f'IC 95% {descriptions[v]}')
        if has_obs:
            ax.plot(observed.index, observed.values, 'D', ms=7, c='k', label='Observat 2025')
        ax.axvline(pd.Timestamp(c.FORECAST_START), c='grey', ls=':')
        ax.set(title='Predicció de la línia de costa gener-juny 2025', xlabel='Data', ylabel='Distància (m)')
        ax.legend(loc='best', fontsize=7); ax.grid(alpha=0.3)
        c.save_fig(fig, PART, name)

    if has_obs:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(results))
        ax.bar(x - 0.2, results['train_RMSE'], 0.4, label='Train (in-sample)')
        ax.bar(x + 0.2, results['test_RMSE'], 0.4, label='Test gener-juny 2025')
        ax.set_xticks(x, results.index, rotation=20)
        ax.set(title='RMSE train vs test', ylabel='RMSE (m)')
        ax.legend(); ax.grid(alpha=0.3, axis='y')
        c.save_fig(fig, PART, '04_train_vs_test_rmse.png')


if __name__ == '__main__':
    main()
