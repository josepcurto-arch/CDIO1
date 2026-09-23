import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# --- Regression models ---
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# 1. Load the dataset
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data           # features (e.g., population, income, etc.)
y = california_housing.target         # target: median house value

# 2. Explore data
print("=== Dataset information ===")
print(california_housing.frame.head())
print(california_housing.DESCR)

california_housing.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)
plt.show()

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create the model
model = LinearRegression()
# model= DecisionTreeRegressor(max_depth=5, random_state=42)
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model = GradientBoostingRegressor(random_state=42)
# model = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000, random_state=42)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predict on the test set
y_pred = model.predict(X_test)
print(f"Predictions: {y_pred}")

# 7. Evaluate the model
print("=== Evaluation information  ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"R2:  {r2_score(y_test, y_pred):.3f}")
