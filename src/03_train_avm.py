from pathlib import Path
import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "apartments_features.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "avm_predictions.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

NUMERIC_FEATURES = [
    "square_meters",
    "rooms",
    "floor",
    "floor_count",
    "building_age",
    "is_build_year_missing",
    "dist_pkin_km",
    "angle_from_center_rad",
    "dist_nearest_metro_km",
    "is_walking_distance_metro",
    "poi_count",
    "school_distance_km",
    "clinic_distance_km",
    "post_office_distance_km",
    "kindergarten_distance_km",
    "restaurant_distance_km",
    "college_distance_km",
    "pharmacy_distance_km",
    "sqm_per_room",
    "relative_floor",
    "is_ground_floor",
    "is_top_floor",
    "amenities_score",
    "month_idx",
    "is_bk2_period",
]

CATEGORICAL_FEATURES = [
    "type",
    "ownership",
    "building_material",
    "condition",
]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ape = np.abs((y_true - y_pred) / y_true)
    return {
        "MAE_PLN": mean_absolute_error(y_true, y_pred),
        "MAPE_pct": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "MdAPE_pct": np.median(ape) * 100,
        "R2": r2_score(y_true, y_pred),
        "Hit_Rate_10pct": np.mean(ape <= 0.10) * 100,
        "Hit_Rate_15pct": np.mean(ape <= 0.15) * 100,
    }


def train_and_evaluate() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file {INPUT_PATH}. Run 02_features.py first.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{INPUT_PATH.as_posix()}')").df()

    # Out-of-Time (OOT) split: Train 2023_08 - 2024_05 | Test OOT: 2024_06
    train_df = df[df["snapshot_month"] < "2024_06"].copy()
    test_df = df[df["snapshot_month"] == "2024_06"].copy()

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train_log = np.log(train_df["price_per_sqm"].values)

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Baseline: Hedonic Ridge Regression
    ridge_pipeline = Pipeline(
        steps=[
            ("prep", ColumnTransformer(
                transformers=[
                    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC_FEATURES),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
                ]
            )),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    ridge_pipeline.fit(X_train, y_train_log)
    ridge_pred_sqm = np.exp(ridge_pipeline.predict(X_test))
    ridge_pred_total = ridge_pred_sqm * test_df["square_meters"].values

    # Primary AVM Model: XGBoost Regressor
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names_out = preprocessor.get_feature_names_out()

    xgb_model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_proc, y_train_log)

    xgb_pred_sqm = np.exp(xgb_model.predict(X_test_proc))
    xgb_pred_total = xgb_pred_sqm * test_df["square_meters"].values

    y_test_total = test_df["price"].values
    metrics_ridge = evaluate_predictions(y_test_total, ridge_pred_total)
    metrics_xgb = evaluate_predictions(y_test_total, xgb_pred_total)

    print("=== OUT-OF-TIME VALIDATION RESULTS (TEST SET: 2024_06, N = 6,598) ===")
    comparison_df = pd.DataFrame(
        [metrics_ridge, metrics_xgb],
        index=["Baseline (Hedonic Ridge)", "AVM Model (XGBoost)"],
    )
    print(comparison_df.round(2).to_string())

    joblib.dump({"preprocessor": preprocessor, "model": xgb_model}, MODELS_DIR / "avm_xgb_bundle.joblib")

    test_df["avm_pred_price_sqm"] = xgb_pred_sqm
    test_df["avm_pred_price"] = xgb_pred_total
    test_df["avm_pct_error"] = (test_df["avm_pred_price"] - test_df["price"]) / test_df["price"]
    test_df.to_parquet(PREDICTIONS_PATH, index=False)

    # Plot 1: Actual vs Predicted (OOT 2024_06)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_total / 1e6, xgb_pred_total / 1e6, alpha=0.25, s=12, color="#1f77b4")
    max_val = max(y_test_total.max(), xgb_pred_total.max()) / 1e6
    plt.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect Valuation (y = x)")
    plt.xlabel("Listing Price (M PLN)")
    plt.ylabel("AVM XGBoost Valuation (M PLN)")
    plt.title(f"AVM Out-of-Time Performance (2024_06) | MAPE: {metrics_xgb['MAPE_pct']:.2f}% | R²: {metrics_xgb['R2']:.3f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_avm_actual_vs_predicted.png", dpi=150)
    plt.close()

    # Plot 2: SHAP Summary
    sample_idx = np.random.RandomState(42).choice(len(X_test_proc), size=min(1500, len(X_test_proc)), replace=False)
    X_shap = pd.DataFrame(X_test_proc[sample_idx], columns=[c.replace("num__", "").replace("cat__", "") for c in feature_names_out])

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_shap, max_display=15, show=False)
    plt.title("Key Price Drivers per sqm in Warsaw (SHAP Values – OOT Set)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Hedonic Market Index & Short-Term Projection (H2 2024)
    df_with_dummies = pd.get_dummies(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], drop_first=True)
    df_with_dummies = df_with_dummies.fillna(df_with_dummies.median())
    month_dummies = pd.get_dummies(df["snapshot_month"], prefix="m", drop_first=True)
    X_hedonic = pd.concat([df_with_dummies.drop(columns=["month_idx", "is_bk2_period"]), month_dummies], axis=1)

    hedonic_time_model = Ridge(alpha=5.0)
    hedonic_time_model.fit(X_hedonic, np.log(df["price_per_sqm"].values))

    months_sorted = sorted(df["snapshot_month"].unique())
    index_values = [100.0]
    for m in months_sorted[1:]:
        coef = hedonic_time_model.coef_[list(X_hedonic.columns).index(f"m_{m}")]
        index_values.append(100.0 * np.exp(coef))

    post_bk2_monthly_growth = (index_values[-1] / index_values[5]) ** (1 / 5) - 1
    future_months = ["2024_07", "2024_08", "2024_09", "2024_10", "2024_11", "2024_12"]

    base_forecast = [index_values[-1] * ((1 + post_bk2_monthly_growth) ** i) for i in range(1, 7)]
    stagnation_forecast = [index_values[-1] * ((1 + 0.002) ** i) for i in range(1, 7)]
    stress_forecast = [index_values[-1] * ((1 - 0.012) ** i) for i in range(1, 7)]

    print(f"\n=== HEDONIC MARKET DYNAMICS (BASE 2023_08 = 100 PTS) ===")
    print(f"Index level in 2024_06: {index_values[-1]:.1f} pts (+{index_values[-1] - 100:.1f}% ceteris paribus)")
    print(f"Avg monthly growth post-BK2 (Jan-Jun 2024): {post_bk2_monthly_growth * 100:.2f}% m/m")
    print(f"2024_12 Projection (Baseline trend): {base_forecast[-1]:.1f} pts ({(base_forecast[-1]/index_values[-1]-1)*100:+.1f}% vs Jun 2024)")
    print(f"2024_12 Projection (Downturn scenario): {stress_forecast[-1]:.1f} pts ({(stress_forecast[-1]/index_values[-1]-1)*100:+.1f}% vs Jun 2024)")

    plt.figure(figsize=(10, 5))
    plt.plot(months_sorted, index_values, marker="o", lw=2.5, color="#1f77b4", label="Hedonic Price Index (Observed)")
    x_future = [months_sorted[-1]] + future_months
    plt.plot(x_future, [index_values[-1]] + base_forecast, "--o", color="#2ca02c", label="Baseline Projection (Post-BK2 Trend)")
    plt.plot(x_future, [index_values[-1]] + stagnation_forecast, "--s", color="#ff7f0e", label="Stabilization Scenario (+0.2% m/m)")
    plt.plot(x_future, [index_values[-1]] + stress_forecast, "--^", color="#d62728", label="Correction Scenario (-1.2% m/m)")
    plt.axvline("2023_12", color="gray", linestyle=":", label="End of BK2% Subsidy Program")
    plt.xticks(rotation=45)
    plt.ylabel("Constant-Quality Price Index (2023_08 = 100)")
    plt.title("Warsaw Residential Hedonic Price Index & H2 2024 Scenarios")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_hedonic_index_forecast.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    train_and_evaluate()