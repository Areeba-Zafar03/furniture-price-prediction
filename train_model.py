import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load and Prepare Data
df = pd.read_csv("furniture_dataset.csv")
X = df.drop(columns=["Price (USD)"])
y = df["Price (USD)"]

# Step 2: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Step 5: Save the Model
with open("linear_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model training complete. Saved as linear_model.pkl")
