# Created by: Ms.Aye Theingi Thwin
# import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# Build First Model
from sklearn.ensemble import RandomForestRegressor


# Load sales data

sales = pd.read_csv("sales.csv")
sales["sale_date"] = pd.to_datetime(
    sales["sale_date"]
)

sales["revenue"] = (
    sales["quantity"] * sales["unit_price"]
)

print(sales.head())

"""
Aggregate daily sales
"""

# Total daily demand

daily_sales = (
    sales
    .groupby("sale_date")
    .agg(
        demand=("quantity", "sum"),
        revenue=("revenue", "sum")
    )
    .reset_index()
)   

# check the result
print(daily_sales.head())
print(daily_sales.shape)


# Visualize demand

plt.figure(figsize=(14, 6))

plt.plot(
    daily_sales["sale_date"],
    daily_sales["demand"]   
)

plt.title("Daily Retail Demand")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.show()


"""
Create Time Series Features
"""

daily_sales["day_of_week"] =(
    daily_sales["sale_date"].dt.dayofweek
)

daily_sales["month"] =(
    daily_sales["sale_date"].dt.month
)

daily_sales["day_of_month"] =(
    daily_sales["sale_date"].dt.day
)

# Lag - yesterday demand

daily_sales["lag_1"] = (
    daily_sales["demand"].shift(1)
) 

# Lag - demand 7 days ago

daily_sales["lag_7"] = (
    daily_sales["demand"].shift(7)
)

# Lag - demand 30 days ago

daily_sales["lag_30"] = (
    daily_sales["demand"].shift(30)
)


"""
Rolling averages
"""

# 7-day Average

daily_sales["rolling_7"] = (
    daily_sales["demand"]
    .rolling(7)
    .mean()
)

# 30-day Average

daily_sales["rolling_30"] = (
    daily_sales["demand"]
    .rolling(30)
    .mean()
)


"""
Remove Missing Values
"""

# lag_30 means missing values not exist for the first 30 days

daily_sales = daily_sales.dropna() 

print(
    daily_sales.isnull().sum()
)


"""
Define our features
"""

# Define features
 
features = [
    "day_of_week",
    "month",
    "day_of_month",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_7",
    "rolling_30"
]

# Target

target = "demand"

x = daily_sales[features]
y = daily_sales[target]


# Use the first 80% for training

split_index = int(
    len(daily_sales) * 0.80
) 

x_train = x.iloc[:split_index]
x_test = x.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Check the result

print("Training rows:", len(x_train))
print("Testing rows:", len(x_test))


"""
Build first model
"""

# Create Model

model = RandomForestRegressor(
    n_estimators = 200,
    random_state = 42,
    n_jobs = -1
)

# Train Model

model.fit(
    x_train,
    y_train
)


"""
Generate predictions
"""

predictions = model.predict(
    x_test
)

# Create Comparison Table

results = pd.DataFrame ({
    "Actual":  y_test.values,
    "Predicted": predictions
})

print(results.head(20))


"""
Evaluate the model
"""

# MAE

mae = mean_absolute_error(
    y_test,
    predictions
)

print("MAE:", round(mae, 2))

# RMSE

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

print("RMSE:", round(rmse, 2))

# MAPE

mape = (
    np.mean(
        np.abs(
            (y_test - predictions)
            / y_test
        )
    )
    * 100
)

print(
    "MAPE:",
    round(mape, 2),
    "%"
) 


"""
Plot Actual vs. Predicted Demand
"""

plt.figure(figsize=(14, 6))

plt.plot(
    y_test.values,
    label = "Actual"
)

plt.plot(
    predictions,
    label = "Predicted"
)

plt.title(
    "Actual vs. Predicted Retail Demand"
)

plt.xlabel("Test Period")
plt.ylabel("Demand")
plt.legend()
plt.tight_layout()
plt.show()


"""
Another business Features
"""

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending = False
)

print(importance)

# Visualization

plt.figure(figsize=(10, 6))

plt.barh(
    importance["feature"],
    importance["importance"]
)

plt.title("Forecast Model Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


"""
Create simple 30-day Forecast
"""

# Use recent available demand infomation

latest_features = daily_sales[
    features
].tail(30)

# Predict

future_prediction = model.predict(
    latest_features
)

forecast_30_days = pd.DataFrame({
    "forecast_day": range(1, 31),
    "forecast_demand": future_prediction
})

print(forecast_30_days)
