import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path("data/processed/mmr_long.parquet")
OUT_FIG = Path("reports/figs")
OUT_TAB = Path("data/processed")
OUT_FIG.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_parquet(DATA_FILE)

    # Keep only rows with actual measurements for plots/rankings
    df_obs = df.dropna(subset=["mmr"]).copy()

    # Latest year per country
    latest_year = int(df_obs["year"].max())
    latest = df_obs[df_obs["year"] == latest_year].copy()

    # Some countries may not have data in the max year; alternative: compute per-country last year
    per_country_last = (df_obs.sort_values(["Country Code", "year"])
                        .groupby("Country Code")
                        .tail(1))

    # Save tables
    latest_rank = (per_country_last.sort_values("mmr", ascending=False)
                   .loc[:, ["Country Name", "Country Code", "year", "mmr"]])
    latest_rank.to_csv(OUT_TAB / "latest_mmr_by_country.csv", index=False)

    # Coverage table
    coverage = (df.groupby("Country Code")
                  .agg(
                      country_name=("Country Name", "last"),
                      min_year=("year", "min"),
                      max_year=("year", "max"),
                      n_years=("year", "count"),
                      n_obs=("mmr", lambda s: s.notna().sum()),
                      pct_missing=("mmr", lambda s: s.isna().mean())
                  )
                  .reset_index())
    coverage.to_csv(OUT_TAB / "coverage_by_country.csv", index=False)

    # Plot 1: Global median trend over time
    global_median = (df_obs.groupby("year")["mmr"]
                        .median()
                        .reset_index(name="median_mmr"))

    plt.figure()
    plt.plot(global_median["year"], global_median["median_mmr"])
    plt.title("Global Median Maternal Mortality Ratio (MMR) Over Time")
    plt.xlabel("Year")
    plt.ylabel("Median MMR")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "global_median_mmr_trend.png", dpi=200)
    plt.close()

    # Plot 2: Latest-year distribution (hist)
    plt.figure()
    plt.hist(per_country_last["mmr"].dropna(), bins=30)
    plt.title(f"Distribution of Latest Available MMR by Country (<= {latest_year})")
    plt.xlabel("MMR")
    plt.ylabel("Number of countries")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "latest_mmr_distribution.png", dpi=200)
    plt.close()

    # Plot 3: Top 20 highest latest MMR
    top20 = latest_rank.head(20).iloc[::-1]  # reverse for horizontal chart
    plt.figure()
    plt.barh(top20["Country Name"], top20["mmr"])
    plt.title("Top 20 Highest Latest Available MMR")
    plt.xlabel("MMR")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "top20_latest_mmr.png", dpi=200)
    plt.close()

    print("EDA outputs written:")
    print(" - data/processed/latest_mmr_by_country.csv")
    print(" - data/processed/coverage_by_country.csv")
    print(" - reports/figs/global_median_mmr_trend.png")
    print(" - reports/figs/latest_mmr_distribution.png")
    print(" - reports/figs/top20_latest_mmr.png")


if __name__ == "__main__":
    main()
