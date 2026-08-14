"""
ml/predict.py
==============
Applies the model trained by ml/train_model.py to TODAY's BUY candidates,
adding a genuinely validated "MLWinProbability" column alongside the
existing hand-picked BuyScore — so you can compare the two and see
whether the heuristic and the data-driven model agree.

REQUIRES: ml/model.pkl (run ml/train_model.py first).

Reads:  signals/daily_top20_buy_ranked.csv
Writes: signals/daily_top20_buy_ml_ranked.csv  (same rows, sorted by
        MLWinProbability, with two new columns: MLWinProbability, BuyScore)

If ml/model.pkl doesn't exist yet, this exits quietly without error so it
can safely sit in the daily workflow before you've trained anything.

USAGE
-----
    python -m ml.predict
"""

import os

import joblib
import numpy as np
import pandas as pd

MODEL_FILE = os.path.join("ml", "model.pkl")
INPUT_FILE = os.path.join("signals", "daily_top20_buy_ranked.csv")
OUTPUT_FILE = os.path.join("signals", "daily_top20_buy_ml_ranked.csv")

# Mirrors backtest/engine.py's _create_event() buy_score formula exactly
# (percentage of these 7 checks that passed), using the live scan's saved
# column names, so the live buy_score feature matches training data.
LIVE_BUY_CHECK_COLUMNS = [
    "BUY_FlipOK",
    "BUY_MA_OK",
    "BUY_ADX_OK",
    "BUY_RSI_OK",
    "BUY_VolumeOK",
    "BUY_OBV_OK",
    "BUY_RS_OK",
]


def compute_live_buy_score(row) -> float:
    passed = sum(1 for c in LIVE_BUY_CHECK_COLUMNS if c in row and bool(row[c]))
    return round(100.0 * passed / len(LIVE_BUY_CHECK_COLUMNS), 1)


def main():
    if not os.path.exists(MODEL_FILE):
        print(f"{MODEL_FILE} not found — skipping ML scoring (run ml/train_model.py first).")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"{INPUT_FILE} not found — run rank_buy.py first.")
        return

    bundle = joblib.load(MODEL_FILE)
    model, features = bundle["model"], bundle["features"]

    df = pd.read_csv(INPUT_FILE)
    if df.empty:
        print("No BUY candidates today — nothing to score.")
        return

    df["rsi"] = pd.to_numeric(df.get("RSI"), errors="coerce")
    df["adx"] = pd.to_numeric(df.get("ADX"), errors="coerce")
    df["volume_ratio"] = pd.to_numeric(df.get("VolumeRatio"), errors="coerce")
    df["relative_strength"] = pd.to_numeric(df.get("RS_LINE"), errors="coerce")
    df["buy_score"] = df.apply(compute_live_buy_score, axis=1)

    missing = df[features].isna().any(axis=1)
    if missing.any():
        print(f"Skipping ML score for {missing.sum()} row(s) with missing feature data "
              f"(e.g. relative strength filter disabled).")

    df["MLWinProbability"] = np.nan
    scoreable = df.loc[~missing, features]
    if not scoreable.empty:
        df.loc[~missing, "MLWinProbability"] = np.round(
            model.predict_proba(scoreable)[:, 1] * 100, 1
        )

    df = df.sort_values("MLWinProbability", ascending=False, na_position="last")

    os.makedirs("signals", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} scored candidates to {OUTPUT_FILE}")
    cols_to_show = [c for c in ["Symbol", "Price", "BuyScore", "MLWinProbability"] if c in df.columns]
    print(df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
