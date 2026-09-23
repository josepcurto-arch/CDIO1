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
from keras.layers import LSTM, Dense

"""
Weather data provided by the Max Planck Institute for Biogeochemistry, Jena, Germany, 
under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
"""

data_file = "data/jena_daily_temp_2009_2017.csv"
model_dir = "models"
output_dir = "output"

# SLIDING WINDOW CONFIGURATION
WINDOW  = 90  # days
HORIZON =  1  # days
STRIDE  =  1  # days

# MODEL CONFIGURATION
LSTM_UNITS = 50
EPOCHS     = 20
BATCH_SIZE = 32

os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

model_name =f"lstm_u{LSTM_UNITS}_e{EPOCHS}_b{BATCH_SIZE}_w{WINDOW}_h{HORIZON}_s{STRIDE}"

""" 
1. Data Collection
   - Load Jena Climate dataset
   - Train/Test split (2009–2016 / 2017)

2. Data Preparation
   - Normalize data (fit on train only!)
   - Create sliding windows

3. Model Development
   - Define LSTM architecture
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

# Train/Test Split (2009–2016 / 2017)
train_ts = ts["2009":"2016"]
test_ts = ts["2017"]

plt.figure(figsize=(12, 4))
plt.plot(train_ts.index, train_ts.values, linewidth=0.5, alpha=0.5, c='C0', label='Training data')
plt.plot(test_ts.index, test_ts.values, linewidth=0.5, alpha=0.5, c='C2', label='Test data')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Daily Temperature: Historical Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#########################################################################
# 2-3. AVOID RETRAINING IF MODEL ALREADY EXISTS
#########################################################################

model_file = f"{model_dir}/{model_name}.keras"
scaler_file = f"{model_dir}/{model_name}.scaler"

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
   print(f"Before normalization: {train_ts.values[0:3]}\n") 
   train_scaled = scaler.fit_transform(train_ts.values.reshape(-1, 1))
   print(f"After normalization: {train_scaled[0:3]}\n") 

   ### Create sliding windows
   X_train, y_train = create_sequences(train_scaled, WINDOW, HORIZON)
   print(f"X_train: {X_train.shape}") #  (2832, 90, 1)
   print(f"y_train: {y_train.shape}") #  (2832, 1, 1)

   # create_sequences generates y_train with shape (samples, HORIZON, 1) 
   # Dense(HORIZON) produces output shape (samples, HORIZON)
   y_train = y_train.squeeze(-1) # remove last dimension
   print(f"y_train: {y_train.shape}") # (2832, 1)

   #########################################################################
   # 3. MODEL DEVELOPMENT
   #########################################################################

   print(f"\n=== Training new model ===\n")

   # Build LSTM model
   model = Sequential([
      Input(shape=(WINDOW, 1)),
      LSTM(LSTM_UNITS),
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

   print("\nTraining metrics (normalized values!):")
   print(f"  MAE  = {final_mae:.3f} ")
   print(f"  RMSE = {final_rmse:.3f} ")
   print(f"  R²   = {final_r2:.3f}")  

   # Metrics from the last validation epoch
   val_rmse = np.sqrt(history.history['val_loss'][-1])
   val_mae  = history.history['val_mae'][-1]
   val_r2   = history.history['val_r2_score'][-1]

   print("\nValidation metrics (normalized values!):")
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
   plt.savefig(f"{output_dir}/{model_name}_loss_curve.png", dpi=300)
   plt.show()

   # Addtional metrics: MAE is useful here (°C, easier to interpret)
   # plt.plot(history.history['mae'], label='Train MAE')
   # plt.plot(history.history['val_mae'], label='Validation MAE')
   # plt.xlabel('Epoch')
   # plt.ylabel('MAE (°C)')
   # plt.title('Mean Absolute Error over Epochs')
   # plt.legend()
   # plt.show()

   # Save model
   print(f"\nSaving model: {model_file}")
   model.save(model_file)

   # Save the scaler
   joblib.dump(scaler, scaler_file)

#########################################################################
# 4. MODEL EVALUATION
#########################################################################

# ONE-STEP PREDICTION

print(f"\n=== One-Step prediction evaluation ===\n")

# Use the same scaler to transform the test period (2017)
test_scaled = scaler.transform(test_ts.values.reshape(-1, 1))

# Add the last WINDOW days of training data
eval_data = np.concatenate([train_scaled[-WINDOW:], test_scaled])

# Create test sequences
X_test, y_test = create_sequences(eval_data, WINDOW, HORIZON)

# Generate predictions
y_pred_scaled = model.predict(X_test, verbose=0)

# Convert predictions back to degrees Celsius
y_pred = scaler.inverse_transform(y_pred_scaled)

# Ground-truth temperatures for 2017
y_true = test_ts.values

# Compute error metrics
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print ("\nTest metrics:")
print(f"  MAE  = {mae:.3f} °C")
print(f"  RMSE = {rmse:.3f} °C")
print(f"  R²   = {r2:.3f}")  

# Dates corresponding to the predictions
prediction_dates = prediction_dates = test_ts.index
prediction_ts = pd.Series(y_pred.flatten(), index=prediction_dates)
observed_ts = pd.Series(y_true.flatten(), index=prediction_dates)

plt.figure(figsize=(12, 4))
plt.plot(train_ts.index, train_ts.values, linewidth=0.5, alpha=0.5, c='C0', label='Training Data')
plt.plot(test_ts.index, test_ts.values, linewidth=0.5, c='C2', label='Observed 2017')
plt.plot(prediction_ts.index, prediction_ts.values, linewidth=1, c='C3', label='LSTM Prediction')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Daily Temperature: Historical Data and LSTM Forecast (one-step ahead)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/{model_name}_forecast.png", dpi=300)
plt.show()

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(test_ts.index, test_ts.values, linewidth=1, c='C2', label='Observed 2017')
plt.plot(prediction_ts.index, prediction_ts.values, linewidth=1, c='C3', label='LSTM Prediction')
plt.title("Daily Temperature Forecast (2017)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/{model_name}_forecast_zoom.png", dpi=300)
plt.show()

