import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# LOAD DATA
# =====================================

train = pd.read_csv("train.csv")
holidays = pd.read_csv("holidays_events.csv")

# =====================================
# DATE COLUMN
# =====================================

train["date"] = pd.to_datetime(train["date"])
holidays["date"] = pd.to_datetime(holidays["date"])

# =====================================
# HOLIDAY PROCESSING
# =====================================

holidays["is_holiday"] = 1

holiday_df = holidays[
    ["date", "is_holiday"]
].drop_duplicates()

# =====================================
# MERGE HOLIDAYS
# =====================================

df = train.merge(
    holiday_df,
    on="date",
    how="left"
)

df["is_holiday"] = (
    df["is_holiday"]
    .fillna(0)
)

# =====================================
# TIME FEATURES
# =====================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.dayofweek
df["weekofyear"] = (
    df["date"]
    .dt.isocalendar()
    .week.astype(int)
)

df["quarter"] = df["date"].dt.quarter

df["is_weekend"] = (
    df["weekday"] >= 5
).astype(int)

# =====================================
# AGGREGATE SALES
# =====================================

daily = (
    df.groupby("date")
    ["sales"]
    .sum()
    .reset_index()
)

daily = daily.merge(
    holiday_df,
    on="date",
    how="left"
)

daily["is_holiday"] = (
    daily["is_holiday"]
    .fillna(0)
)

# =====================================
# DATE FEATURES AGAIN
# =====================================

daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["day"] = daily["date"].dt.day
daily["weekday"] = daily["date"].dt.dayofweek
daily["weekofyear"] = (
    daily["date"]
    .dt.isocalendar()
    .week.astype(int)
)

daily["quarter"] = daily["date"].dt.quarter

# =====================================
# LAG FEATURES
# =====================================

daily["lag_1"] = daily["sales"].shift(1)
daily["lag_7"] = daily["sales"].shift(7)
daily["lag_14"] = daily["sales"].shift(14)
daily["lag_30"] = daily["sales"].shift(30)

# =====================================
# ROLLING FEATURES
# =====================================

daily["rolling_mean_7"] = (
    daily["sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

daily["rolling_mean_30"] = (
    daily["sales"]
    .shift(1)
    .rolling(30)
    .mean()
)

daily["rolling_std_7"] = (
    daily["sales"]
    .shift(1)
    .rolling(7)
    .std()
)

# =====================================
# REMOVE NULLS
# =====================================

daily = daily.dropna()

# =====================================
# FEATURES
# =====================================

FEATURES = [
    "year",
    "month",
    "day",
    "weekday",
    "weekofyear",
    "quarter",
    "is_holiday",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_7"
]

TARGET = "sales"

X = daily[FEATURES]
y = daily[TARGET]

# =====================================
# TRAIN TEST SPLIT
# =====================================

split = int(len(daily) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =====================================
# XGBOOST MODEL
# =====================================

model = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# =====================================
# PREDICTIONS
# =====================================

preds = model.predict(X_test)

# =====================================
# METRICS
# =====================================

mae = mean_absolute_error(
    y_test,
    preds
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        preds
    )
)

r2 = r2_score(
    y_test,
    preds
)

mape = (
    np.mean(
        np.abs(
            (y_test - preds)
            / y_test
        )
    )
    * 100
)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)
print("MAPE:", mape)
Future ml 01 folder name