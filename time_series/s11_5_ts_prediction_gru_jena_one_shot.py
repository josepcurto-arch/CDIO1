import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # desactiva mensajes de oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # silencia logs de TF (0=all, 3=only errors)
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras import Input
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense

"""
Weather data provided by the Max Planck Institute for Biogeochemistry, Jena, Germany, 
under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
"""

data_file = "data/jena_daily_temp_2009_2017.csv"
model_dir = "models"

### SLIDING WINDOW CONFIGURATION
WINDOW  = 365  # days
HORIZON = 365  # days
STRIDE  =   1  # days

# MODEL CONFIGURATION
GRU_UNITS  = 50
EPOCHS     = 20
BATCH_SIZE = 32

""" 
1. Data Collection
   - Load Jena Climate dataset
   - Train/Test split (2009–2016 / 2017)

2. Data Preparation
   - Normalize data (fit on train only!)
   - Create sliding windows

3. Model Development
   - Define GRU architecture
   - Train the model

4. Model Evaluation
   - Predict temperatures for 2017
   - Inverse-transform predictions
   - Calculate MAE and RMSE
   - Plot actual vs. predicted temperatures
"""

#########################################################################
# FUNCTIONS
#########################################################################
 
### Create sliding windows
def create_sequences(data, window, horizon, stride=1):

   X, y = [], []
   for i in range(0, len(data) - window - horizon + 1, stride):
      X.append(data[i:i + window])
      y.append(data[i + window:i + window + horizon])
   
   return np.array(X), np.array(y)

#########################################################################
# 1. LOAD DATA 
#########################################################################

print(f"\n=== Load data ===\n")

df = pd.read_csv(data_file)
print(df.head(3))

# Convert 'Date Time' to datetime format and set as index
df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
ts = df.set_index("Date Time")["T (degC)"]

# Check daily frequency
# print(f"Frequency: {pd.infer_freq(ts.index)}")

### Train/Test Split (2009–2016 / 2017)

train_ts = ts["2009":"2016"]
test_ts = ts["2017"]

# plt.figure(figsize=(12, 4))
# plt.plot(train_ts.index, train_ts.values, linewidth=0.5, alpha=0.5, c='C0', label='Training data')
# plt.plot(test_ts.index, test_ts.values, linewidth=0.5, alpha=0.5, c='C2', label='Test data')
# plt.xlabel('Date')
# plt.ylabel('Temperature (°C)')
# plt.title('Daily Temperature: Historical Data')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

#########################################################################
# 2. 3. AVOID RETRAINING IF MODEL ALREADY EXISTS
#########################################################################

os.makedirs(model_dir, exist_ok=True)
model_file = (
   f"{model_dir}/"
   f"gru_w{WINDOW}_h{HORIZON}_s{STRIDE}_"
   f"u{GRU_UNITS}_e{EPOCHS}_b{BATCH_SIZE}.keras"
)
scaler_file = model_file.replace(".keras", ".scaler")

if os.path.exists(model_file) and os.path.exists(scaler_file):
   print(f"\n=== Loading existing model ===\n")

   print(f"Loading model: {model_file}")
   model = load_model(model_file)
   model.summary()
  
   scaler = joblib.load(scaler_file)
   train_scaled = scaler.transform(train_ts.values.reshape(-1, 1))

else:

   #########################################################################
   # 2. PREPARE DATA 
   #########################################################################

   print(f"\n=== Prepare data ===\n")

   ### Normalization
   scaler = MinMaxScaler()

   # Fit the scaler only on the training period 
   train_scaled = scaler.fit_transform(train_ts.values.reshape(-1, 1))
   print(f"After normalization: {train_scaled[0:3]}\n") # numpy.ndarray

   ### Create sliding windows
   X_train, y_train = create_sequences(train_scaled, WINDOW, HORIZON)
   print(f"X_train: {X_train.shape}") #  (2193, 365, 1)
   print(f"y_train: {y_train.shape}") #  (2193, 365, 1)

   # create_sequences generates y_train with shape (samples, HORIZON, 1) 
   # Dense(HORIZON) produces output shape (samples, HORIZON)
   y_train = y_train.squeeze(-1) # remove last dimension
   print(f"y_train: {y_train.shape}") # (2193, 1)

   #########################################################################
   # 3. MODEL DEVELOPMENT
   #########################################################################

   print(f"\n=== Training new model ===\n")

   model = Sequential([
        Input(shape=(WINDOW, 1)),
        GRU(GRU_UNITS),
        Dense(HORIZON)
        ])

   model.compile(optimizer='adam', 
                 loss='mse', 
                 metrics=['mae', 'r2_score']
                 )

   model.summary()

   # Train the model
   history = model.fit(
      X_train,
      y_train,
      epochs=EPOCHS,
      batch_size=BATCH_SIZE,
      validation_split=0.2
   )    

   # Training metrics
   print(history.history.keys())
   
   # Metrics from the last training epoch
   final_rmse = np.sqrt(history.history['loss'][-1])
   final_mae  = history.history['mae'][-1]
   final_r2   = history.history['r2_score'][-1]

   print("\nTraining metrics:")
   print(f"  MAE  = {final_mae:.3f} ")
   print(f"  RMSE = {final_rmse:.3f} ")
   print(f"  R²   = {final_r2:.3f}")  

   # Metrics from the last validation epoch
   val_rmse = np.sqrt(history.history['val_loss'][-1])
   val_mae  = history.history['val_mae'][-1]
   val_r2   = history.history['val_r2_score'][-1]

   print("\nValidation metrics:")
   print(f"  MAE  = {val_mae:.3f} ")
   print(f"  RMSE = {val_rmse:.3f} ")
   print(f"  R²   = {val_r2:.3f}")  

   # Loss curve (train vs. validation)
   # Both curves decreasing → model is learning well
   # Train loss ↓ but val loss ↑ → overfitting
   # Both curves plateau early → underfitting, or model has converged
   plt.plot(history.history['loss'], label='Train Loss')
   plt.plot(history.history['val_loss'], label='Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss (MSE)')
   plt.title('Training vs. Validation Loss')
   plt.legend()
   plt.show()   

   # Save model
   print(f"\nSaving model: {model_file}")
   model.save(model_file)

   # Save the scaler
   joblib.dump(scaler, scaler_file)

#########################################################################
# 4. MODEL EVALUATION
#########################################################################

print(f"\n=== Model Evaluation ===\n")

# Use the last WINDOW days from the training period (end of 2016)
window_data = train_scaled[-WINDOW:].copy()

# Reshape to GRU input format: (samples, timesteps, features)
X_input = window_data.reshape(1, WINDOW, 1)

# Predict all 365 days of 2017 in a single forward pass
y_pred_scaled = model.predict(X_input, verbose=0)
print(f"Prediction shape: {y_pred_scaled.shape}")

# Convert predictions back to degrees Celsius
y_pred_scaled = np.array(y_pred_scaled).reshape(-1,1)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Ground-truth temperatures for 2017
y_true = test_ts.values

print(f"y_pred: {y_pred.shape}")
print(f"y_true: {y_true.shape}")

# Compute error metrics
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print ("\nTest metrics:")
print(f"  MAE  = {mae:.3f} °C")
print(f"  RMSE = {rmse:.3f} °C")
print(f"  R²   = {r2:.3f}")  

# Dates corresponding to the predictions
prediction_dates = test_ts.index
prediction_ts = pd.Series(y_pred.flatten(), index=prediction_dates)
observed_ts = pd.Series(y_true.flatten(), index=prediction_dates)

plt.figure(figsize=(12, 4))
plt.plot(train_ts.index, train_ts.values, linewidth=0.5, alpha=0.5, c='C0', label='Training Data')
plt.plot(test_ts.index, test_ts.values, linewidth=0.5, c='C2', label='Observed 2017')
plt.plot(prediction_ts.index, prediction_ts.values, linewidth=1, c='C3', label='GRU Prediction')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Daily Temperature: Historical Data and GRU Forecast (One-Shot)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(test_ts.index, test_ts.values, linewidth=1, c='C2', label='Observed 2017')
plt.plot(prediction_ts.index, prediction_ts.values, linewidth=1, c='C3', label='GRU Prediction')
plt.title("Daily Temperature Forecast (2017)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
