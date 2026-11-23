from pathlib import Path
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "avm_predictions.parquet"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
PORTFOLIO_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mortgage_stress_results.parquet"

FORCED_SALE_HAIRCUT = 0.20   # 20% forced-sale / foreclosure discount
WORKOUT_COST_RATE = 0.05     # 5% legal & recovery workout costs relative to EAD


def compute_secured_lgd(ead: np.ndarray, collateral_value: np.ndarray) -> np.ndarray:
    net_recovery = collateral_value * (1.0 - FORCED_SALE_HAIRCUT)
    raw_lgd = (ead - net_recovery) / ead + WORKOUT_COST_RATE
    return np.clip(raw_lgd, 0.0, 1.0)


def run_collateral_stress_test() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing file {PREDICTIONS_PATH}. Run 03_train_avm.py first.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{PREDICTIONS_PATH.as_posix()}')").df()

    rng = np.random.default_rng(seed=42)
    origination_ltv = np.clip(rng.normal(loc=0.72, scale=0.14, size=len(df)), 0.25, 0.95)
    df["ead_pln"] = df["price"] * origination_ltv

    scenarios = {
        "1. Baseline (AVM 2024_06)": 1.00,
        "2. Mild Shock (-10% prices)": 0.90,
        "3. Severe Shock (-20% prices)": 0.80,
    }

    summary_rows = []
    for name, multiplier in scenarios.items():
        val_col = df["avm_pred_price"].values * multiplier
        ltv = (df["ead_pln"].values / val_col) * 100.0
        lgd = compute_secured_lgd(df["ead_pln"].values, val_col) * 100.0
        ew_lgd = np.average(lgd, weights=df["ead_pln"].values)

        summary_rows.append(
            {
                "Scenario": name,
                "Mean_LTV_pct": np.mean(ltv),
                "Median_LTV_pct": np.median(ltv),
                "Share_LTV_above_80pct": np.mean(ltv > 80.0) * 100.0,
                "Share_Underwater_LTV_100pct": np.mean(ltv > 100.0) * 100.0,
                "Mean_LGD_pct": np.mean(lgd),
                "EAD_Weighted_LGD_pct": ew_lgd,
            }
        )

        key = name.split(".")[0].strip()
        df[f"ltv_s{key}"] = ltv
        df[f"lgd_s{key}"] = lgd

    summary_df = pd.DataFrame(summary_rows)
    print("=== MORTGAGE COLLATERAL STRESS-TEST (WARSAW PORTFOLIO, N = 6,598) ===")
    print(f"Assumptions: Forced-sale haircut = {FORCED_SALE_HAIRCUT:.0%}, Workout costs = {WORKOUT_COST_RATE:.0%} EAD\n")
    print(summary_df.round(2).to_string(index=False))

    df["ltv_bucket"] = pd.cut(
        df["ltv_s1"],
        bins=[0, 60, 80, 90, 200],
        labels=["<= 60% (Low)", "60% - 80% (Standard)", "80% - 90% (Elevated)", "> 90% (High)"],
    )

    bucket_df = (
        df.groupby("ltv_bucket", observed=False)
        .agg(
            Loan_Count=("id", "count"),
            Exposure_M_PLN=("ead_pln", lambda x: x.sum() / 1e6),
            LGD_Baseline_pct=("lgd_s1", "mean"),
            LGD_Shock_10_pct=("lgd_s2", "mean"),
            LGD_Shock_20_pct=("lgd_s3", "mean"),
        )
        .reset_index()
    )
    print("\n--- LGD SENSITIVITY BY CURRENT LTV BUCKET ---")
    print(bucket_df.round(2).to_string(index=False))

    df.to_parquet(PORTFOLIO_OUTPUT_PATH, index=False)

    # Plot 4: LTV Shift under Stress Scenarios
    plt.figure(figsize=(10, 5))
    bins = np.linspace(20, 140, 45)
    plt.hist(df["ltv_s1"], bins=bins, alpha=0.55, label="Baseline (AVM)", color="#1f77b4", density=True)
    plt.hist(df["ltv_s2"], bins=bins, alpha=0.45, label="Mild Shock (-10% Prices)", color="#ff7f0e", density=True)
    plt.hist(df["ltv_s3"], bins=bins, alpha=0.45, label="Severe Shock (-20% Prices)", color="#d62728", density=True)
    plt.axvline(80, color="black", linestyle="--", lw=1.2, label="KNF Rec. S Threshold (LTV = 80%)")
    plt.axvline(100, color="darkred", linestyle=":", lw=1.5, label="Negative Equity Threshold (LTV = 100%)")
    plt.xlabel("Current Loan-to-Value (LTV %)")
    plt.ylabel("Portfolio Density")
    plt.title("Impact of Property Price Shocks on Mortgage Portfolio LTV Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_ltv_stress_distribution.png", dpi=150)
    plt.close()

    # Plot 5: Non-linear LGD Sensitivity by LTV Bucket
    x = np.arange(len(bucket_df))
    width = 0.26

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width, bucket_df["LGD_Baseline_pct"], width, label="Baseline (AVM)", color="#1f77b4")
    plt.bar(x, bucket_df["LGD_Shock_10_pct"], width, label="Mild Shock (-10% Prices)", color="#ff7f0e")
    plt.bar(x + width, bucket_df["LGD_Shock_20_pct"], width, label="Severe Shock (-20% Prices)", color="#d62728")

    plt.xticks(x, bucket_df["ltv_bucket"])
    plt.ylabel("Mean Secured LGD (%)")
    plt.title("Non-Linear LGD Sensitivity to Collateral Depreciation by LTV Bucket")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_lgd_sensitivity_by_ltv.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_collateral_stress_test()