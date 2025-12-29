import pandas as pd
import numpy as np
from pathlib import Path

import statsmodels.formula.api as smf
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_FILE = Path("data/processed/mmr_long.parquet")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    df = pd.read_parquet(DATA_FILE).copy()

    # Keep observed outcomes for modeling
    df = df.dropna(subset=["mmr"]).copy()
    df["mmr_log1p"] = np.log1p(df["mmr"])

    # -----------------------------
    # A) Panel model (fixed effects)
    # -----------------------------
    # Use only a reasonable year range if the earliest years are extremely sparse (optional)
    # df = df[df["year"] >= 1990].copy()

    panel = df.dropna(subset=["mmr_log1p"]).copy()

    # Country fixed effects via categorical term
    # Note: this can be heavy but is fine for World Bank country-year scale.
    fe_model = smf.ols("mmr_log1p ~ year + C(Q('Country Code'))", data=panel).fit()

    # Save panel summary to text
    with open(OUT_DIR / "panel_model_summary.txt", "w") as f:
        f.write(fe_model.summary().as_text())

    # -----------------------------
    # B) Forecasting with lag features
    # -----------------------------
    df = df.sort_values(["Country Code", "year"]).copy()
    df["mmr_lag1"] = df.groupby("Country Code")["mmr"].shift(1)
    df["mmr_lag2"] = df.groupby("Country Code")["mmr"].shift(2)
    df["mmr_roll3"] = df.groupby("Country Code")["mmr"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["mmr_delta1"] = df["mmr"] - df["mmr_lag1"]

    # Target = next year's mmr (supervised next-step prediction)
    df["target_next_mmr"] = df.groupby("Country Code")["mmr"].shift(-1)

    ml = df.dropna(subset=["target_next_mmr"]).copy()  # keep rows with target
    # Time-aware split
    max_year = int(ml["year"].max())
    cutoff = max_year - 5  # last 5 years as test window (adjust as needed)
    train = ml[ml["year"] <= cutoff].copy()
    test = ml[ml["year"] > cutoff].copy()

    features = ["mmr_lag1", "mmr_lag2", "mmr_roll3", "mmr_delta1", "year"]


    X_train, y_train = train[features], train["target_next_mmr"]
    X_test, y_test = test[features], test["target_next_mmr"]

    # Baseline ML model: Ridge regression
    model = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                           ("ridge", Ridge(alpha=0.1, random_state=0))
                           ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    metrics = {
        "cutoff_year": cutoff,
        "train_rows": len(train),
        "test_rows": len(test),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(rmse(y_test, pred)),
        "test_year_min": int(test["year"].min()),
        "test_year_max": int(test["year"].max()),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUT_DIR / "forecast_metrics.csv", index=False)

    # Save predictions for inspection
    preds = test.loc[:, ["Country Name", "Country Code", "year", "mmr", "target_next_mmr"]].copy()
    preds["pred_next_mmr"] = pred
    preds.to_csv(OUT_DIR / "forecast_predictions.csv", index=False)

    print("Model outputs written:")
    print(" - data/processed/panel_model_summary.txt")
    print(" - data/processed/forecast_metrics.csv")
    print(" - data/processed/forecast_predictions.csv")
    print("Forecast metrics:", metrics)


if __name__ == "__main__":
    main()