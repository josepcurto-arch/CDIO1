# Step 0 - Install packages (if not installed yet)
#   - pandas
#   - matplotlib
#   - scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# --- Classification models ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Step 1 - Load dataset
iris = pd.read_csv("data/iris.csv")

# Step 2 - Explore data
print("=== Dataset information  ===")
print(iris.info())
print(iris.head())
iris['Species'] = iris['Species'].astype('category')
iris['Species'] = iris['Species'].cat.codes
print(iris.head())
print(iris.describe())

scatter_matrix(iris, c=iris['Species'], marker='o', s=10, alpha=.8)
plt.show()

# Step 3 — Organize data into training and testing sets
X = iris.iloc[:, :-1].values
y = iris['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

# Step 4 — Create the model
clf = KNeighborsClassifier(3)
# clf = DecisionTreeClassifier(max_depth=5)
# clf = GaussianNB()
# clf = SVC(kernel="linear", C=0.025)
# clf = SVC(gamma=2, C=1)
# clf = MLPClassifier(alpha=1, max_iter=1000)

# Step 5 - Train the model
clf.fit(X_train, y_train)

# Step 6 - Make predictions
y_pred = clf.predict(X_test)
print(f"Predictions: {y_pred}")

# Step 7 — Evaluate the model
print("=== Evaluation information  ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Confusion matrix: \n{confusion_matrix(y_test, y_pred)}")
