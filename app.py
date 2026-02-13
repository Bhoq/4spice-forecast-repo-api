from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import date
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

app = FastAPI(title="4Spice Forecast API", version="1.0")


# =========================
# Models
# =========================

class SalesRow(BaseModel):
    Month: date
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
    group_by: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
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
                        "Discount_Pct": 0.1,
                        "Marketing_Spend_USD": 1200,
                        "Avg_Unit_Price_USD": 4.5,
                        "Units_Sold": 1000,
                        "Gross_Revenue_USD": 4500,
                        "Returns_Pct": 0.02,
                        "Net_Sales_USD": 4410,
                        "Holiday_Flag": 0
                    }
                ]
            }
        }


# =========================
# Helper
# =========================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Month_dt"] = pd.to_datetime(df["Month"])
    df["month_num"] = df["Month_dt"].dt.month

    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    df = df.sort_values("Month_dt")
    df["t"] = np.arange(len(df))

    return df


# =========================
# Endpoint
# =========================

@app.post("/forecast")
def forecast(req: ForecastRequest) -> Dict[str, Any]:

    df = pd.DataFrame([r.model_dump() for r in req.rows])

    group_cols = req.group_by or []
    agg_cols = group_cols + ["Month"]

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
        "Holiday_Flag"
    ]

    results = []
    metrics = []

    if group_cols:
        groups = monthly.groupby(group_cols)
    else:
        groups = [(("TOTAL",), monthly)]

    for key, gdf in groups:
        gdf = gdf.sort_values("Month_dt")

        X = gdf[feature_cols].values
        y = gdf["Net_Sales_USD"].values

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        mae = float(mean_absolute_error(y, y_pred))
        r2 = float(r2_score(y, y_pred)) if len(y) > 1 else None

        results.append({
            "group": key,
            "forecast": [
                {
                    "Month": str(m),
                    "Predicted_Net_Sales_USD": float(p)
                }
                for m, p in zip(gdf["Month"], y_pred)
            ]
        })

        metrics.append({
            "group": key,
            "MAE": mae,
            "R2": r2
        })

    return {"metrics": metrics, "results": results}
