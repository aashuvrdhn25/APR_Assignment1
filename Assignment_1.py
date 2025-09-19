# Linear Regression on Diabetes Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("diabetesData.csv")  

# Step 2: Explore dataset
print("Dataset shape:", df.shape)
print(df.head())

# Step 3: Features (X) and Target (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Step 6: Predictions
y_pred = lin_reg.predict(X_test)

# Step 7: Evaluation
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 8: Plot Actual vs Predicted
plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression - Actual vs Predicted")
plt.show()