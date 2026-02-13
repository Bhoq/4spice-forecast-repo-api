# app.py
from __future__ import annotations

from datetime import date
from typing import List, Optional, Dict, Any, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, confloat

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


app = FastAPI(title="4Spice Forecast API", version="1.0")


# -----------------------------
# Models
# -----------------------------
class SalesRow(BaseModel):
    # Make Month a real date so FastAPI validates input (prevents "string" crashing pandas)
    Month: date = Field(..., description="Month start date in YYYY-MM-DD format (e.g., 2024-01-01)")
    Store: str
    Region: str
    Category: str

    Promo_Flag: conint(ge=0, le=1) = 0
    Discount_Pct: confloat(ge=0, le=1) = 0.0
    Marketing_Spend_USD: confloat(ge=0) = 0.0

    Avg_Unit_Price_USD: confloat(ge=0) = 0.0
    Units_Sold: conint(ge=0) = 0
    Gross_Revenue_USD: confloat(ge=0) = 0.0
    Returns_Pct: confloat(ge=0, le=1) = 0.0
    Net_Sales_USD: confloat(ge=0) = 0.0

    Holiday_Flag: conint(ge=0, le=1) = 0


class ForecastRequest(BaseModel):
    rows: List[SalesRow] = Field(..., min_length=3)
    forecast_year: int = Field(2027, ge=2000, le=2100)
    group_by: Optional[List[str]] = Field(
        default=None,
        description='Optional grouping columns. Examples: ["Region"], ["Category"], ["Store"], or null for TOTAL.'
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "forecast_year": 2027,
                "group_by": ["Region"],
                "rows": [
                    {
                        "Month": "2024-01-01",
                        "Store": "4Spice Superstore",
                        "Region": "Northeast",
                        "Category": "Spices",
                        "Promo_Flag": 1,
                        "Discount_Pct": 0.10,
                        "Marketing_Spend_USD": 1200,
                        "Avg_Unit_Price_USD": 4.5,
                        "Units_Sold": 1000,
                        "Gross_Revenue_USD": 4500,
                        "Returns_Pct": 0.02,
                        "Net_Sales_USD": 4410,
                        "Holiday_Flag": 0
                    },
                    {
                        "Month": "2024-02-01",
                        "Store": "4Spice Superstore",
                        "Region": "Northeast",
                        "Category": "Spices",
                        "Promo_Flag": 0,
                        "Discount_Pct": 0.05,
                        "Marketing_Spend_USD": 900,
                        "Avg_Unit_Price_USD": 4.6,
                        "Units_Sold": 950,
                        "Gross_Revenue_USD": 4370,
                        "Returns_Pct": 0.02,
                        "Net_Sales_USD": 4283,
                        "Holiday_Flag": 0
                    },
                    {
                        "Month": "2024-03-01",
                        "Store": "4Spice Superstore",
                        "Region": "Northeast",
                        "Category": "Spices",
                        "Promo_Flag": 1,
                        "Discount_Pct": 0.12,
                        "Marketing_Spend_USD": 1500,
                        "Avg_Unit_Price_USD": 4.55,
                        "Units_Sold": 1100,
                        "Gross_Revenue_USD": 5005,
                        "Returns_Pct": 0.02,
                        "Net_Sales_USD": 4905,
                        "Holiday_Flag": 0
                    }
                ]
            }
        }
    }


# -----------------------------
# Helpers
# -----------------------------
ALLOWED_GROUP_COLS = {"Store", "Region", "Category"}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Month_dt"] = pd.to_datetime(df["Month"])
    df["year"] = df["Month_dt"].dt.year
    df["month_num"] = df["Month_dt"].dt.month

    # seasonality encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    # time index (months since start)
    df = df.sort_values("Month_dt")
    df["t"] = (
        (df["Month_dt"].dt.year - df["Month_dt"].dt.year.min()) * 12
        + (df["Month_dt"].dt.month - df["Month_dt"].dt.month.min())
    )
    return df


def normalize_month_start(dt: pd.Timestamp) -> pd.Timestamp:
    # ensure month start
    return pd.Timestamp(year=dt.year, month=dt.month, day=1)


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "4Spice Forecast API is running"}


# -----------------------------
# Endpoint
# -----------------------------
@app.post("/forecast")
def forecast(req: ForecastRequest) -> Dict[str, Any]:
    # Validate group_by early (prevents weird key errors)
    group_cols = req.group_by or []
    invalid = [c for c in group_cols if c not in ALLOWED_GROUP_COLS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid group_by columns: {invalid}. Allowed: {sorted(ALLOWED_GROUP_COLS)}"
        )

    # Convert rows -> DataFrame
    df = pd.DataFrame([r.model_dump() for r in req.rows])

    # Ensure Month is at month start
    df["Month"] = pd.to_datetime(df["Month"]).apply(normalize_month_start)

    agg_cols = group_cols + ["Month"]

    # Monthly aggregation (sum sales/marketing, mean discount, max flags)
    monthly = (
        df.groupby(agg_cols, as_index=False)
        .agg(
            Net_Sales_USD=("Net_Sales_USD", "sum"),
            Marketing_Spend_USD=("Marketing_Spend_USD", "sum"),
            Promo_Flag=("Promo_Flag", "max"),
            Discount_Pct=("Discount_Pct", "mean"),
            Holiday_Flag=("Holiday_Flag", "max"),
        )
    )

    monthly = add_time_features(monthly)

    feature_cols = [
        "t",
        "month_sin",
        "month_cos",
        "Promo_Flag",
        "Discount_Pct",
        "Marketing_Spend_USD",
        "Holiday_Flag",
    ]
    target_col = "Net_Sales_USD"

    # guardrail: need enough data to train a model
    if len(monthly) < 3:
        raise HTTPException(
            status_code=400,
            detail="Not enough monthly data after aggregation. Provide at least 3 distinct months per group."
        )

    results: List[Dict[str, Any]] = []
    metrics: List[Dict[str, Any]] = []

    # Train separately per group (or overall)
    if group_cols:
        grouped_iter = monthly.groupby(group_cols)
    else:
        grouped_iter = [(("TOTAL",), monthly)]

    rng = np.random.default_rng(42)  # reproducible promo simulation

    for key, gdf in grouped_iter:
        gdf = gdf.sort_values("Month_dt").reset_index(drop=True)

        if len(gdf) < 3:
            # Skip tiny groups instead of crashing
            continue

        X = gdf[feature_cols].values
        y = gdf[target_col].values

        # Split: last 6 months as test if enough history
        if len(gdf) >= 18:
            split = len(gdf) - 6
        else:
            split = max(len(gdf) - 3, 1)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Metrics
        mae = None
        r2 = None
        if len(y_test) >= 1:
            y_pred = model.predict(X_test)
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else None

        # 12 months of forecast for requested year
        start_month = pd.Timestamp(f"{req.forecast_year}-01-01")
        future_months = pd.date_range(start_month, periods=12, freq="MS")

        # baseline future inputs from recent history
        promo_rate = float(gdf["Promo_Flag"].tail(12).mean()) if len(gdf) >= 12 else float(gdf["Promo_Flag"].mean())
        disc_avg = float(gdf["Discount_Pct"].tail(12).mean()) if len(gdf) >= 12 else float(gdf["Discount_Pct"].mean())
        mkt_avg = float(gdf["Marketing_Spend_USD"].tail(12).mean()) if len(gdf) >= 12 else float(gdf["Marketing_Spend_USD"].mean())

        t_last = int(gdf["t"].max())

        future = pd.DataFrame({"Month": future_months})
        future["Month_dt"] = future["Month"]
        future["month_num"] = future["Month_dt"].dt.month
        future["month_sin"] = np.sin(2 * np.pi * future["month_num"] / 12)
        future["month_cos"] = np.cos(2 * np.pi * future["month_num"] / 12)
        future["t"] = np.arange(t_last + 1, t_last + 13)

        # baseline inputs
        future["Promo_Flag"] = (rng.random(12) < promo_rate).astype(int)
        future["Discount_Pct"] = disc_avg
        future["Marketing_Spend_USD"] = mkt_avg
        future["Holiday_Flag"] = future["month_num"].isin([11, 12]).astype(int)

        y_future = model.predict(future[feature_cols].values)

        group_label: Union[str, Dict[str, str]]
        if not group_cols:
            group_label = "TOTAL"
        else:
            # key can be a scalar if single group col
            key_tuple = key if isinstance(key, tuple) else (key,)
            group_label = {col: str(val) for col, val in zip(group_cols, key_tuple)}

        results.append(
            {
                "group": group_label,
                "forecast": [
                    {"Month": str(m.date()), "Predicted_Net_Sales_USD": float(max(0, p))}
                    for m, p in zip(future_months, y_future)
                ],
            }
        )

        metrics.append({"group": group_label, "MAE": mae, "R2": r2})

    if not results:
        raise HTTPException(
            status_code=400,
            detail="No groups had enough data to train. Provide more months per group (>=3)."
        )

    return {"metrics": metrics, "results": results}
