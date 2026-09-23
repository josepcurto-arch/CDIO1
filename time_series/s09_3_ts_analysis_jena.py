"""
Weather data provided by the Max Planck Institute for Biogeochemistry, Jena, Germany, 
under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.fft import fft, ifft, fftfreq


print("=" * 70)
print("TIME SERIES ANALYSIS PIPELINE - JENA CLIMATE DATASET")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[1] LOADING DATA...\n")

file = "data/jena_climate_2009_2016.csv.zip"
df = pd.read_csv(file)

print(f"Data loaded from {file}\n")
print(df.info())
print(f"\nFirst rows:\n{df.head(3)}")
print(f"\nLast rows:\n{df.tail(3)}")

# ============================================================================
# 2. PREPARING DATA
# ============================================================================

print("\n" + "=" * 70)
print("[2] CLEANING AND PREPARING")
print("=" * 70)

# 2.1 Datetime as index
print("\n[2.1] Setting datetime as index...\n")
print(f"Index type before: {type(df.index)}")

df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
df.set_index('Date Time', inplace=True)
df.sort_index(inplace=True)

print(f"Index type after: {type(df.index)}")
print(f"\nFirst rows:\n{df.head(3)}")
print(f"\nLast rows:\n{df.tail(3)}")

# 2.2 Trim to analysis period (drop timestamp outside 2009-2016)
print("\n[2.2]Trimming to 2009-2016 analysis period...")
df = df[(df.index.year >= 2009) & (df.index.year <= 2016)]
print(f"Date range: {df.index.min()} to {df.index.max()}")

# 2.3 Focus on Temperature
print("\n[2.3] Extracting temperature series...")
ts = df['T (degC)'].copy()
print (ts.info())
print(f"\nFirst rows:\n{ts.head(3)}")
print(f"\nLast rows:\n{ts.tail(3)}")

# 2.4 Check for NANs
print("\n[2.4] Checking for missing values...")
total_nans = ts.isna().sum()
print(f"Missing values: {total_nans}")
# If there are NANs, handle them
if total_nans > 0:
    print(f"Filling missing values with forward fill...")
    df.ffill(inplace=True) 

# 2.5 Outliers detection (IQR method for Temperature)
print("\n[2.5] Detecting outliers (Temperature)...")
Q1 = ts.quantile(0.25)
Q3 = ts.quantile(0.75)
IQR = Q3 - Q1
outliers = ((ts < (Q1 - 1.5 * IQR)) | (ts > (Q3 + 1.5 * IQR))).sum()
print(f"Outliers detected: {outliers} ({outliers/len(df)*100:.2f}%)")
print(f"Temperature range: [{ts.min():.1f}, {ts.max():.1f}] °C")

# 2.6 Resampling (from 10-min to daily)
print("\n[2.6] Resampling from 10-min to daily frequency...")
ts_daily = ts.resample('D').mean()

# Two NaN appear after resampling: days without data in the original series!!!
print(f"Total missing values after resample: {ts_daily.isna().sum()}")
print(f"Dates with NaN: {ts_daily.index[ts_daily.isna()]}")
# Apply interpolation
ts_daily = ts_daily.interpolate(method='linear')

plt.figure(figsize=(12, 4))
plt.plot(ts.index, ts.values, linewidth=0.5, alpha=0.7, c='C1', label = 'original (10-min frequency)')
plt.plot(ts_daily.index, ts_daily.values, linewidth=0.8, c='C0', label = 'resampled (daily frequency)')
plt.title('Temperature over time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

ts = ts_daily.copy()

# Save the cleaned and resampled series for future use
output_file = "data/jena_climate_2009_2016_daily_temperature.csv"
ts.to_csv(output_file)

# ============================================================================
# 3. EXPLORATORY VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("[3] EXPLORATORY VISUALIZATION")
print("=" * 70)

plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts.values, linewidth=0.8, alpha=0.8)
plt.title('Daily temperature over time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.show()

# Distribution
plt.figure(figsize=(12, 6))
plt.hist(ts.values, bins=50, edgecolor='black', alpha=0.8)
plt.title('Temperature distribution')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.axvline(ts.mean(), color='red', linestyle='--', label=f'Mean: {ts.mean():.1f}°C')
plt.legend()
plt.show()

print("\nExploratory plots generated")

# ============================================================================
# 4. STATISTICAL ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("[4] STATISTICAL ANALYSIS")
print("=" * 70)

# 4.1 Descriptive Statistics
print("\n[4.1] Descriptive Statistics:")
print(f"{ts.describe()}\n")

print(f"  Mean:     {ts.mean():.2f} °C")
print(f"  Median:   {ts.median():.2f} °C")
print(f"  Std Dev:  {ts.std():.2f} °C")
print(f"  Min:    {ts.min():.2f} °C")
print(f"  Max:     {ts.max():.2f} °C")
print(f"  Range:   {ts.max() - ts.min():.2f} °C")

# 4.2 Box Plots
print("\n[4.2] Generating box plots...")

# Box plot by month
# list of 12 arrays/series, each one with the values of all years for that month
month_data = [ts[ts.index.month == m].values for m in range(1, 13)]
month_labels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]
plt.figure(figsize=(12, 6))
plt.boxplot(month_data, tick_labels=month_labels)
plt.title("Temperature distribution by month")
plt.ylabel("Temperature (°C)")
plt.xlabel("Month")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# ============================================================================
# 5. TREND ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("[5] TREND ANALYSIS")
print("=" * 70)

# 5.1 Moving Averages
print("\n[5.1] Calculating moving averages...")
ma_7 = ts.rolling(window=7).mean()
ma_30 = ts.rolling(window=30).mean()
ma_365 = ts.rolling(window=365).mean()

plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts.values, label='Daily', alpha=0.5, linewidth=0.5)
plt.plot(ma_7.index, ma_7.values, label='7-day MA', linewidth=1.5)
plt.plot(ma_30.index, ma_30.values, label='30-day MA', linewidth=1.5)
plt.plot(ma_365.index, ma_365.values, label='365-day MA', linewidth=2)
plt.title('Moving Averages')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Moving averages: 7-day, 30-day, 365-day")

# 5.2 Linear Regression (Trend Line)
print("\n[5.2] Fitting linear regression for trend...")
X = np.arange(len(ts)).reshape(-1, 1)
y = ts.values

model = LinearRegression()
model.fit(X, y)

# Predictions (trend)
trend = model.predict(X)

intercept = model.intercept_
slope = model.coef_[0]
rmse = np.sqrt(mean_squared_error(y, trend))
r2 = model.score(X, y)

print(f"Intercept (b0): {intercept:.3f} °C")
print(f"Slope (b1):  {slope:.6f} °C/day ({slope*365:.3f} °C/year)")
print(f"MAE:   {np.mean(np.abs(y - trend)):.3f} °C")
print(f"RMSE:  {rmse:.3f} °C")
print(f"R² score: {r2:.3f}")

# Plot data and regression line
plt.figure(figsize=(12, 4))
plt.plot(ts.index, y, color='C0', linewidth=0.8, alpha=0.8, label="Observed data")
plt.plot(ts.index, trend, color='C3', label=f"Trend (slope={slope:.3f})")
plt.title(f'Linear Trend (R²={r2:.3f})')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================================
# 6. FREQUENCY ANALYSIS (FFT)
# ============================================================================

print("\n" + "=" * 70)
print("[6] FREQUENCY ANALYSIS (FFT)")
print("=" * 70)

# Remove trend to analyze cycles
ts_detrended = ts - trend 

plt.figure(figsize=(12, 4))
plt.plot(ts.index, ts.values, linewidth=0.5, alpha=0.8, c = 'C0', label="original")
plt.plot(ts.index, trend, linewidth=1, alpha=1, c = 'C3', label="trend")
plt.plot(ts.index, ts_detrended, linewidth=1, alpha=0.8, c = 'C2', label="detrended")
plt.title('Temperature over time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Apply FFT
n = len(ts_detrended)
fft_values = fft(ts_detrended) 
fft_freq = fftfreq(n) 

# fftfreq(n) returns a vector of length n with values in the range [-0.5, 0.5), arranged as:
#  [0, 1/n, 2/n, ..., 0.5, -0.5, ..., -1/n]
# The first half contains positive frequencies (0 up to ~0.5), 
# and the second half the negative ones (~-0.5 up to 0) 

# fft_freq is in cycles-per-sample; since each sample = 1 day,
# frequencies are cycles/day and 1/freq gives the period in days.
# If the sampling interval were e.g. 1 hour, periods would be in hours

# Extract magnitude and phase
magnitude = np.abs(fft_values)
# phase = np.angle(fft_values)

# Plot the frequency spectrum and a zoomed view around the main peak
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

axes[0].plot(fft_freq, magnitude, color='tab:blue')
axes[0].set_title('Frequency spectrum')
axes[0].set_xlabel('Frequency (cycles/day)')
axes[0].set_ylabel('Magnitude')
axes[0].grid(True, alpha=0.3)

# Zoom around the dominant frequency (excluding the zero-frequency component)
positive_mask = fft_freq >= 0
positive_freq = fft_freq[positive_mask]
positive_mag = magnitude[positive_mask]
main_idx = np.argmax(positive_mag[1:]) + 1
peak_freq = positive_freq[main_idx]
peak_mag = positive_mag[main_idx]
zoom_band = 0.025
axes[1].plot(positive_freq, positive_mag, color='tab:orange')
axes[1].axvline(peak_freq, color='red', linestyle='--', label=f'Peak: {peak_freq:.4f} cycles/day')
axes[1].set_xlim(max(0, peak_freq - zoom_band), peak_freq + zoom_band)
axes[1].set_title('Zoom around the main peak')
axes[1].set_xlabel('Frequency (cycles/day)')
axes[1].set_ylabel('Magnitude')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

# Show top frequencies
positive_freq_magnitudes = magnitude[fft_freq >= 0]
positive_freq_frequencies = fft_freq[fft_freq >= 0]
sorted_idx = np.argsort(positive_freq_magnitudes)[::-1] # [::-1] in descending order
print(f"\nTop 5 detected periods:")
for idx in sorted_idx[:5]:
    freq = positive_freq_frequencies[idx]
    magn = positive_freq_magnitudes[idx]
    print(f"  Period: {1/freq:>7.2f} days, Magnitude: {magn:>8.2f}")

# Filter frequencies by magnitude: Keep top N frequencies
N = 2
top_indices = sorted_idx[:N]  # indices of top N positive frequencies
top_threshold = positive_freq_magnitudes[top_indices[-1]]  # magnitude of the Nth
fft_filtered = fft_values.copy()
fft_filtered[magnitude < top_threshold] = 0

# Reconstruct smoothed series
cyclical = np.real(ifft(fft_filtered))

plt.figure(figsize=(12, 4))
plt.plot(ts.index, ts.values, linewidth=0.5, alpha=0.5, c = 'C0', label="original")
plt.plot(ts.index, trend, linewidth=1, alpha=1, c = 'C3', label="trend")
plt.plot(ts.index, ts_detrended, linewidth=0.5, alpha=0.5, c = 'C2', label="detrended")
plt.plot(ts.index, cyclical, linewidth=2, alpha=1, c = 'C2', label="cyclical")
plt.title('Temperature over time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================================
# 7. TIME SERIES DECOMPOSSITION
# ============================================================================

print("\n" + "=" * 70)
print("[7] DECOMPOSSITION")
print("=" * 70)

reconstructed = trend + cyclical
residuals = ts - reconstructed

fig, axes = plt.subplots(4, 1, figsize=(12, 8))

axes[0].plot(ts.index, ts.values, color = 'C0', label="original") # blue
axes[0].set_title('Original series')
axes[0].set_ylabel('Temperature (°C)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(ts.index, trend, color = 'C3', label="trend") # red
axes[1].set_title('Trend component')
axes[1].set_ylabel('Temperature (°C)')
axes[1].grid(True, alpha=0.3)   

axes[2].plot(ts.index, cyclical, color = 'C2', label="cyclical") # green
axes[2].set_title('Seasonal component')
axes[2].set_ylabel('Temperature (°C)')
axes[2].grid(True, alpha=0.3)

axes[3].scatter(ts.index, residuals, s=5, c='C1', label="residuals") # orange
axes[3].set_title('Residuals')
axes[3].set_ylabel('Temperature (°C)')
axes[3].grid(True, alpha=0.3)   

plt.tight_layout()
plt.show()

print("\nDecomposition: trend, cyclical, residuals extracted and plotted.")

# ============================================================================
# 8. RESIDUALS ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("[8] RESIDUALS ANALYSIS")
print("=" * 70)

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
plt.plot(ts.index, residuals, marker='o', linestyle='None')
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

# ============================================================================

print("\n" + "=" * 70)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)

