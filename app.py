# app.py
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


app = FastAPI(title="4Spice Forecast API", version="1.0")


class SalesRow(BaseModel):
    Month: str  # "YYYY-MM-01"
    Store: str
    Region: str
    Category: str
    Promo_Flag: int
    Discount_Pct: float
    Marketing_Spend_USD: float
    Avg_Unit_Price_USD: float
    Units_Sold: int
    Gross_Revenue_USD: float
    Returns_Pct: float
    Net_Sales_USD: float
    Holiday_Flag: int


class ForecastRequest(BaseModel):
    rows: List[SalesRow]
    forecast_year: int = 2027
    group_by: Optional[List[str]] = None  # e.g. ["Region"] or ["Category"] or None


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


@app.post("/forecast")
def forecast(req: ForecastRequest) -> Dict[str, Any]:
    # Load rows -> DataFrame
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    df["Month"] = pd.to_datetime(df["Month"]).dt.to_period("M").dt.to_timestamp()

    group_cols = req.group_by or []
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

    results: List[Dict[str, Any]] = []
    metrics: List[Dict[str, Any]] = []

    # Train separately per group (or overall)
    if group_cols:
        groups = monthly.groupby(group_cols)
    else:
        groups = [(("TOTAL",), monthly)]

    for key, gdf in groups:
        gdf = gdf.sort_values("Month_dt").reset_index(drop=True)

        X = gdf[feature_cols].values
        y = gdf[target_col].values

        # Simple split: last 6 months as test if enough history
        if len(gdf) >= 18:
            split = len(gdf) - 6
        else:
            split = max(len(gdf) - 3, 1)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else None

        # Build 12 months of forecast for requested year
        start_month = pd.Timestamp(f"{req.forecast_year}-01-01")
        future_months = pd.date_range(start_month, periods=12, freq="MS")

        # Baseline future inputs from recent history
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

        # baseline inputs (you can later allow users to pass these)
        rng = np.random.default_rng(42)  # reproducible
        future["Promo_Flag"] = (rng.random(12) < promo_rate).astype(int)
        future["Discount_Pct"] = disc_avg
        future["Marketing_Spend_USD"] = mkt_avg
        future["Holiday_Flag"] = future["month_num"].isin([11, 12]).astype(int)

        y_future = model.predict(future[feature_cols].values)

        group_label = "TOTAL" if not group_cols else dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))

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

    return {"metrics": metrics, "results": results}
