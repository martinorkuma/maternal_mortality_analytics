import re
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_FILE = RAW_DIR / "API_SH.STA.MMRT_DS2.csv"
META_COUNTRY = RAW_DIR / "Metadata_Country_API_SH.STA.MMRT_DS2.csv"
META_INDICATOR = RAW_DIR / "Metadata_Indicator_API_SH.STA.MMRT_DS2.csv"

EXPECTED_ID_COLS = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]

def find_year_columns(cols):
    return [c for c in cols if re.fullmatch(r"\d{4}", str(c))]

def read_worldbank_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, skiprows=4)

def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def main():
    # ---- Load main ----
    df = strip_columns(read_worldbank_csv(MAIN_FILE))

    missing = [c for c in EXPECTED_ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in main file: {missing}. "
            f"Columns found: {list(df.columns)[:20]}..."
        )

    year_cols = find_year_columns(df.columns)
    if not year_cols:
        raise ValueError("No year columns detected. Check the raw CSV format.")

    # Save cleaned wide (useful for QA)
    df.to_csv(OUT_DIR / "mmr_wide_clean.csv", index=False)

    # ---- Wide -> Long ----
    long_df = df.melt(
        id_vars=EXPECTED_ID_COLS,
        value_vars=year_cols,
        var_name="year",
        value_name="mmr"
    )

    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["mmr"] = pd.to_numeric(long_df["mmr"], errors="coerce")
    long_df = long_df.dropna(subset=["year"]).sort_values(["Country Code", "year"])

    # Derived column (safe + readable)
    long_df["mmr_log1p"] = np.log1p(long_df["mmr"])

    # ---- Metadata joins (non-silent) ----
    try:
        meta_c = strip_columns(read_worldbank_csv(META_COUNTRY))
        if "Country Code" in meta_c.columns:
            long_df = long_df.merge(meta_c, on="Country Code", how="left", suffixes=("", "_countrymeta"))
        else:
            print("Warning: META_COUNTRY has no 'Country Code' column; skipped merge.")
    except Exception as e:
        print(f"Warning: failed to load/merge country metadata: {e}")

    try:
        meta_i = strip_columns(read_worldbank_csv(META_INDICATOR))
        # Drop junk columns like "Unnamed: 4" if present
        meta_i = meta_i.loc[:, ~meta_i.columns.str.match(r"^Unnamed")]
        
        if "INDICATOR_CODE" in meta_i.columns:
            long_df = long_df.merge(meta_i, left_on="Indicator Code", right_on="INDICATOR_CODE", how="left", suffixes=("", "_indmeta"))
            # Optional: remove redundant join key from metadata side
            long_df = long_df.drop(columns=["INDICATOR_CODE"])
        else:
            print("Warning: META_INDICATOR has no 'INDICATOR_CODE' column; skipped merge.")
    except Exception as e:
        print(f"Warning: failed to load/merge indicator metadata: {e}")

    # ---- Save ----
    out_parquet = OUT_DIR / "mmr_long.parquet"
    out_csv = OUT_DIR / "mmr_long.csv"
    long_df.to_parquet(out_parquet, index=False)
    long_df.to_csv(out_csv, index=False)

    print("Wrote:")
    print(f" - {out_parquet}  rows={len(long_df):,} cols={len(long_df.columns)}")
    print(f" - {out_csv}")

if __name__ == "__main__":
    main()
