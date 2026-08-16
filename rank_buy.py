import glob
import os
import pandas as pd
import numpy as np


# ============================================================
# BUY RANKING
# ============================================================

INPUT_DIR = "signals"
OUTPUT_FILE = os.path.join(INPUT_DIR, "daily_top20_buy_ranked.csv")


def get_latest_scan():
    files = glob.glob(os.path.join(INPUT_DIR, "scan_*.csv"))

    if not files:
        raise FileNotFoundError(
            "No scan_*.csv files found in the signals folder."
        )

    return max(files, key=os.path.getmtime)


def num(row, column, default=0.0):
    """Safely read a numeric column."""
    if column not in row.index:
        return default

    value = pd.to_numeric(row[column], errors="coerce")

    if pd.isna(value):
        return default

    return float(value)


def boolean(row, column):
    """Safely read a boolean column."""
    if column not in row.index:
        return False

    value = row[column]

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in (
        "true",
        "1",
        "yes",
        "y",
    )


def calculate_score(row):
    """
    100-point BUY strength score.

    IMPORTANT:
    This score ranks existing FINAL BUY signals.
    It does NOT change the scanner's BUY/SELL logic.
    """

    score = 0.0

    # --------------------------------------------------------
    # 1. RSI MOMENTUM — 15 points
    # --------------------------------------------------------
    rsi = num(row, "RSI")

    if 55 <= rsi < 60:
        score += 8
    elif 60 <= rsi < 65:
        score += 11
    elif 65 <= rsi < 70:
        score += 14
    elif 70 <= rsi < 75:
        score += 15
    elif 75 <= rsi < 78:
        score += 10
    elif rsi >= 50:
        score += 5

    # --------------------------------------------------------
    # 2. ADX TREND STRENGTH — 15 points
    # --------------------------------------------------------
    adx = num(row, "ADX")

    if adx >= 35:
        score += 15
    elif adx >= 30:
        score += 13
    elif adx >= 25:
        score += 11
    elif adx >= 20:
        score += 8
    elif adx >= 15:
        score += 4

    # --------------------------------------------------------
    # 3. VOLUME — 15 points
    # --------------------------------------------------------
    vol_ratio = num(row, "VolumeRatio")

    if vol_ratio >= 2.0:
        score += 15
    elif vol_ratio >= 1.5:
        score += 13
    elif vol_ratio >= 1.2:
        score += 10
    elif vol_ratio >= 1.0:
        score += 6

    # --------------------------------------------------------
    # 4. KAMA TREND STACK — 15 points
    # --------------------------------------------------------
    kama_fast = num(row, "KAMA_FAST", np.nan)
    kama_mid = num(row, "KAMA_MID", np.nan)
    kama_slow = num(row, "KAMA_SLOW", np.nan)

    if (
        not np.isnan(kama_fast)
        and not np.isnan(kama_mid)
        and not np.isnan(kama_slow)
    ):
        if kama_fast > kama_mid > kama_slow:
            score += 15
        elif kama_fast > kama_mid:
            score += 8

    elif boolean(row, "BUY_MA_OK"):
        score += 15

    # --------------------------------------------------------
    # 5. VSTOP REVERSAL — 15 points
    # --------------------------------------------------------
    price = num(row, "Price")
    vstop = num(row, "VStop", np.nan)

    if not np.isnan(vstop) and price > 0 and vstop > 0:
        vstop_distance = ((price - vstop) / price) * 100

        if vstop_distance >= 15:
            score += 15
        elif vstop_distance >= 10:
            score += 13
        elif vstop_distance >= 7:
            score += 11
        elif vstop_distance >= 4:
            score += 8
        elif vstop_distance > 0:
            score += 5
    elif boolean(row, "BUY_FLIP_OK"):
        score += 15

    # --------------------------------------------------------
    # 6. OBV — 10 points
    # --------------------------------------------------------
    if boolean(row, "BUY_OBV_OK") or boolean(row, "OBV_OK"):
        score += 10

    # --------------------------------------------------------
    # 7. RELATIVE STRENGTH — 10 points
    # --------------------------------------------------------
    if (
        boolean(row, "BUY_RS_OK")
        or boolean(row, "RelativeStrength_Filter_OK")
    ):
        score += 10

    return round(min(score, 100), 1)


def main():

    latest_file = get_latest_scan()

    print("=" * 75)
    print("BUY STRENGTH RANKING")
    print("=" * 75)
    print(f"Input: {latest_file}")

    df = pd.read_csv(latest_file)

    print(f"Rows scanned: {len(df)}")

    # --------------------------------------------------------
    # Identify FINAL BUY rows
    # --------------------------------------------------------

    buy_column = None

    for candidate in (
        "FINAL_BUY",
        "FinalBUY",
        "BUY_SIGNAL",
        "Signal",
    ):
        if candidate in df.columns:
            buy_column = candidate
            break

    if buy_column is None:
        raise KeyError(
            "Could not find a BUY signal column. "
            f"Available columns: {df.columns.tolist()}"
        )

    buy_mask = (
        df[buy_column]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "BUY", "1", "YES"])
    )

    buys = df[buy_mask].copy()

    if buys.empty:
        print()
        print("NO CONFIRMED BUY SIGNALS.")
        return

    # --------------------------------------------------------
    # Calculate ranking
    # --------------------------------------------------------

    buys["BuyScore"] = buys.apply(calculate_score, axis=1)

    # Sort strongest first
    buys = buys.sort_values(
        ["BuyScore", "ADX", "RSI"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    buys["Rank"] = range(1, len(buys) + 1)

    # --------------------------------------------------------
    # Save Top 20
    # --------------------------------------------------------

    top20 = buys.head(20).copy()

    preferred_columns = [
        "Rank",
        "Symbol",
        "Name",
        "Category",
        "Timeframe",
        "Date",
        "Price",
        "CurrentPrice",
        "PriceDriftPct",
        "StalePrice",
        "PctFrom52WHigh",
        "PullbackZone",
        "BuyScore",
        "RSI",
        "ADX",
        "VolumeRatio",
        "VolAboveAvg",
        "VStop",
        "Trend",
        "BUY_FLIP_OK",
        "BUY_MA_OK",
        "BUY_RSI_OK",
        "BUY_VOL_OK",
        "BUY_ADX_OK",
        "BUY_OBV_OK",
        "BUY_RS_OK",
    ]

    output_columns = [
        c for c in preferred_columns
        if c in top20.columns
    ]

    top20[output_columns].to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("CONFIRMED BUY RANKING")
    print("=" * 75)

    display_columns = [
        c for c in [
            "Rank",
            "Symbol",
            "Category",
            "Timeframe",
            "Price",
            "CurrentPrice",
            "PriceDriftPct",
            "StalePrice",
            "PctFrom52WHigh",
            "PullbackZone",
            "BuyScore",
            "RSI",
            "ADX",
            "VolumeRatio",
        ]
        if c in top20.columns
    ]

    print(
        top20[display_columns]
        .to_string(index=False)
    )

    print()
    print("=" * 75)
    print(f"Confirmed BUYs: {len(buys)}")
    print(f"Top 20 saved: {OUTPUT_FILE}")
    print("=" * 75)


if __name__ == "__main__":
    main()