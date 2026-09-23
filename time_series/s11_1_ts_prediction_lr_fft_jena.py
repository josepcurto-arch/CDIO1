import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

"""
Weather data provided by the Max Planck Institute for Biogeochemistry, Jena, Germany, 
under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
"""

data_file = "data/jena_daily_temp_2009_2017.csv"

#########################################################################
# 1. LOAD DATA 
#########################################################################

print(f"\n=== Load data ===")

df = pd.read_csv(data_file)

# Convert 'Date Time' to datetime format and set as index
df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
ts = df.set_index("Date Time")["T (degC)"]

# Check daily frequency
print(f"Frequency: {pd.infer_freq(ts.index)}")

# Train/test split (2009–2016 / 2017)
train_ts = ts["2009":"2016"]
test_ts = ts["2017"]

#########################################################################
# 2. MODEL: Trend (regression) + Cyclical (FFT)
#########################################################################

print(f"\n=== Fit model to data ===\n")

# Fit linear regression to capture the trend
n = len(train_ts) # Number of data points.Ex: 2922
t = np.arange(n).reshape(-1, 1) # 
# [[   0]
#  [   1]
#  [   2]
#  ...
#  [2921]]

model = LinearRegression()
model.fit(t, train_ts)
trend = model.predict(t)

print(f"Intercept (b0): {model.intercept_:.3f} °C")
print(f"Slope (b1):  {model.coef_[0]:.6f} °C/day ({model.coef_[0]*365:.3f} °C/year)")

# Remove trend to analyze cycles
train_ts_detrended = train_ts - trend

# Apply FFT
fft_values = fft(train_ts_detrended)
fft_freq = fftfreq(n)

# Extract magnitude
magnitude = np.abs(fft_values)

# Keep top N frequencies
N = 2
positive_magnitudes = magnitude[fft_freq >= 0]
positive_frequencies = fft_freq[fft_freq >= 0]
sorted_idx = np.argsort(positive_magnitudes)[::-1] 
top_indices = sorted_idx[:N]  
top_threshold = positive_magnitudes[top_indices[-1]]  
significant_freq = fft_freq[magnitude >= top_threshold]
significant_values = fft_values[magnitude >= top_threshold]
print(f"Significant frequencies: {significant_freq}")

fft_filtered = fft_values.copy()
fft_filtered[magnitude < top_threshold] = 0

# Reconstruct smoothed series
cyclical = np.real(ifft(fft_filtered))

# Combine trend + cycles
reconstructed = trend + cyclical

plt.figure(figsize=(12, 4))
plt.plot(train_ts.index, train_ts.values, linewidth=0.5, alpha=0.5, c = 'C0', label="Original")
plt.plot(train_ts.index, trend, linewidth=1, alpha=1, c = 'C3', label="Trend")
plt.plot(train_ts.index, cyclical, linewidth=1, alpha=1, c = 'C2', label="Cyclical")
plt.plot(train_ts.index, reconstructed, linewidth=2, alpha=1, c = 'C0', label="Reconstructed")
plt.title('Temperature over time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#########################################################################
# 3. MODEL FIT EVALUATION (in-sample: 2009-2016) 
#########################################################################

print(f"\n=== Model fit evaluation ===")

# Reconstructed vs original
mae = np.mean(np.abs(train_ts - reconstructed))
rmse = np.sqrt(mean_squared_error(train_ts, reconstructed))
r2 = r2_score(train_ts, reconstructed)

print("\nTraining metrics (reconstructed vs original):")
print(f"  MAE:  {mae:.3f} °C")
print(f"  RMSE: {rmse:.3f} °C")
print(f"  R²:   {r2:.3f}")

# Residuals analysis
residuals = train_ts - reconstructed
mean_res = np.mean(residuals)    
std_res = np.std(residuals)         
min_res = np.min(residuals)
max_res = np.max(residuals)

print("\nResiduals")
print(f"  Mean:  {mean_res:.3f} °C")
print(f"  Std:   {std_res:.3f} °C")
print(f"  Min: {min_res:.3f} °C")
print(f"  Max:  {max_res:.3f} °C")

plt.figure(figsize=(12, 6))
plt.plot(train_ts.index, residuals, marker='o', linestyle='None')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Distribution
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.8)
plt.title('Residuals distribution')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.1f}°C')
plt.legend()
plt.tight_layout()
plt.show()

#########################################################################
# 4. PREDICTION
#########################################################################

print(f"\n=== Predict future values ===")

# Number of future points to predict
n_future = 365  # 1 year
t_future = np.arange(n, n + n_future) # [2922 2923 ...]

# 1. Predict future trend (linear regression)
t_prediction = t_future.reshape(-1, 1)
trend_prediction = model.predict(t_prediction)

# 2. Reconstruct the cyclical pattern for future time points
cyclical_prediction = np.zeros(n_future)
for freq, value in zip(significant_freq, significant_values):
    amplitude = np.abs(value) / n
    phase = np.angle(value)
    cyclical_prediction += amplitude * np.cos(2 * np.pi * freq * t_future.flatten() + phase)

# 3. Combine trend + cycles for final prediction
final_prediction = trend_prediction.flatten() + cyclical_prediction

# Future indixes as dates
last_date = train_ts.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future, freq='D')
# Build pandas series
prediction_ts = pd.Series(final_prediction, index=future_dates)

plt.figure(figsize=(12, 4))
plt.plot(train_ts.index, train_ts.values, linewidth=0.5, alpha=0.5, c = 'C0', label="Original")
plt.plot(train_ts.index, reconstructed, linewidth=2, alpha=1, c = 'C0', label="Reconstructed")
plt.plot(prediction_ts.index, prediction_ts.values, linewidth=2, alpha=1, c = 'C3', label='Prediction')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.title('Daily temperature: Historical Data and Prediction', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#########################################################################
# 5.EVALUATION  (Predicted vs Real observed values for 2017)
#########################################################################

print(f"\n=== Evaluation Metrics ===\n")

# Check aligment of prediction and observations by dates
print(f"Test data loaded: {len(test_ts)} daily points from {test_ts.index[0].date()} to {test_ts.index[-1].date()}")
print(f"Predicted data: {len(prediction_ts)} daily points from {prediction_ts.index[0].date()} to {prediction_ts.index[-1].date()}")

# Evaluation metrics
mae = np.mean(np.abs(test_ts - prediction_ts))
rmse = np.sqrt(mean_squared_error(test_ts, prediction_ts))
r2 = r2_score(test_ts, prediction_ts)

print(f"\nEvaluation Metrics (Prediction vs Observed for 2017)")
print(f"  MAE:  {mae:.3f} °C")
print(f"  RMSE: {rmse:.3f} °C")
print(f"  R²:   {r2:.3f}")

# Visualization
plt.figure(figsize=(12, 4))
plt.plot(train_ts.index, train_ts.values, linewidth=0.5, alpha=0.5, c = 'C0', label="Original")
plt.plot(train_ts.index, reconstructed, linewidth=2, alpha=1, c = 'C0', label="Reconstructed")
plt.plot(test_ts.index, test_ts.values, linewidth=0.5, alpha=0.5, c = 'C3', label='Observed')
plt.plot(prediction_ts.index, prediction_ts.values, linewidth=2, alpha=1, c = 'C3', label='Prediction')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.title('Daily temperature: Historical Data and Prediction', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#########################################################################
# 6. CONFIDENCE INTERVALS
#########################################################################

print(f"\n=== Confidence Intervals ===")

# Use residual standard deviation
sigma_hat = std_res

# Confidence level (95%)
z_score = 1.96

# Calculate prediction intervals (growing with horizon)
prediction_intervals = []
for h in range(1, n_future + 1):
    # Uncertainty grows with forecast horizon
    uncertainty = z_score * sigma_hat * np.sqrt(1 + h / n)
    prediction_intervals.append(uncertainty)

prediction_intervals = np.array(prediction_intervals)

# Upper and lower bounds
upper_bound = final_prediction + prediction_intervals
lower_bound = final_prediction - prediction_intervals

print(f"\nPrediction Intervals (95% confidence)")
print(f"  σ̂ (residual std): {sigma_hat:.3f} °C")
print(f"  Average interval width: {2 * prediction_intervals.mean():.3f} °C")
print(f"  Interval width range: [{2 * prediction_intervals.min():.3f}, {2 * prediction_intervals.max():.3f}] °C")

# Visualization 
plt.figure(figsize=(12, 4))
plt.plot(train_ts.index, train_ts.values, linewidth=0.5, alpha=0.5, c = 'C0', label="Original")
plt.plot(train_ts.index, reconstructed, linewidth=2, alpha=1, c = 'C0', label="Reconstructed")
plt.plot(test_ts.index, test_ts.values, linewidth=0.5, alpha=0.5, c = 'C3', label='Observed')
plt.plot(prediction_ts.index, prediction_ts.values, linewidth=2, alpha=1, c = 'C3', label='Prediction')
plt.fill_between(future_dates, lower_bound, upper_bound, alpha=0.3, color='red', label='95% Confidence Interval')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.title('Daily temperature: Historical Data and Prediction with 95% CI', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Monthly prediction summary
print("\nDaily predictions for 2017:")
print("=" * 70)
for i, (date, pred, lower, upper) in enumerate(zip(future_dates, final_prediction, lower_bound, upper_bound)):
    print(f"{date.strftime('%Y-%m-%d')}: {pred:7.2f} °C  [{lower:7.2f}, {upper:7.2f}]  (±{prediction_intervals[i]:5.2f} °C)") 


# PICP (Prediction Interval Coverage Probability)
# Fraction of observations inside the confidence interval
inside_interval = (test_ts.values >= lower_bound) & (test_ts.values <= upper_bound)
picp = np.mean(inside_interval)

print(f"PICP : {picp:.3f} ({picp*100:.1f}% coverage)")

    # NOTE:
    # Evaluation metrics depend on the temporal resolution of the observations.
    # Resampling/averaging the observed series smooths variability and typically
    # reduces RMSE/MAE while increasing interval coverage (PICP). Therefore,
    # metrics computed at different sampling frequencies are not directly comparable.