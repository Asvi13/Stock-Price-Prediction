import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# -----------------------------------
# 1. Load the dataset
# -----------------------------------

data = pd.read_csv("dataset/stock_data.csv")

print("Dataset loaded successfully!")
print("Total rows:", len(data))


# -----------------------------------
# 2. Select one stock
# -----------------------------------

stock_name = "AAL"

data = data[data["Name"] == stock_name].copy()

print("Stock selected:", stock_name)
print("Number of records:", len(data))


# -----------------------------------
# 3. Sort data by date
# -----------------------------------

data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date")


# -----------------------------------
# 4. Create next-day closing price
# -----------------------------------

data["Next_Close"] = data["close"].shift(-1)

# Remove the last row because it has no next-day value
data = data.dropna()


# -----------------------------------
# 5. Select input features
# -----------------------------------

features = ["open", "high", "low", "close", "volume"]

X = data[features]
y = data["Next_Close"]


# -----------------------------------
# 6. Split the data
# -----------------------------------

# 80% for training
# 20% for testing

split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# -----------------------------------
# 7. Create the ML model
# -----------------------------------

model = LinearRegression()


# -----------------------------------
# 8. Train the model
# -----------------------------------

model.fit(X_train, y_train)

print("Model training completed!")


# -----------------------------------
# 9. Make predictions
# -----------------------------------

predictions = model.predict(X_test)


# -----------------------------------
# 10. Evaluate the model
# -----------------------------------

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("------------------------")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


# -----------------------------------
# 11. Save the trained model
# -----------------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")

print("\nModel saved successfully!")
print("Location: model/model.pkl")