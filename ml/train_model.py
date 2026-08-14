"""
ml/train_model.py
===================
Trains a real, statistically-fit model to predict whether a BUY signal is
likely to be profitable — replacing (or supplementing) rank_buy.py's
hand-picked point-value heuristic with something actually validated
against historical outcomes.

REQUIRES: signals/backtest_trades.csv (produced by run_backtest.py ->
backtest/simulator.py). Run a FULL backtest first — see README.md,
"Backtesting" section. A quick 50-symbol test run will not have enough
trades to train anything meaningful; aim for at least a few hundred
trades, ideally 1,000+.

WHAT IT DOES
------------
1. Loads every historical trade the backtester simulated.
2. Uses each trade's signal-time features (RSI, ADX, volume ratio,
   relative strength, the original rule-based buy_score) to predict
   whether that trade's net_return ended up positive.
3. Splits chronologically (not randomly) — trains on the earlier 80% of
   trades, validates on the most recent 20% — so the reported accuracy
   reflects genuine forward-looking performance, not lookahead bias from
   shuffling dates together.
4. Prints validation metrics (accuracy, ROC-AUC, precision/recall) and
   feature importances, so you can see whether the model is actually
   better than random and which signals matter most.
5. Saves the trained model to ml/model.pkl.

USAGE
-----
    python -m ml.train_model

Re-run this periodically (e.g. monthly, or after any change to the
scanner's rules) using a freshly regenerated backtest_trades.csv, so the
model keeps learning from up-to-date market behavior instead of going
stale.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

TRADES_FILE = os.path.join("signals", "backtest_trades.csv")
MODEL_FILE = os.path.join("ml", "model.pkl")

FEATURES = ["rsi", "adx", "volume_ratio", "relative_strength", "buy_score"]
TARGET = "net_return"

MIN_TRADES_RECOMMENDED = 300


def load_training_data() -> pd.DataFrame:
    if not os.path.exists(TRADES_FILE):
        raise SystemExit(
            f"\n{TRADES_FILE} not found.\n"
            "Run a FULL backtest first: python run_backtest.py (with DEV_MODE = False), "
            "then python -m backtest.simulator, before training a model.\n"
        )

    df = pd.read_csv(TRADES_FILE)
    df = df.dropna(subset=FEATURES + [TARGET])

    if len(df) < MIN_TRADES_RECOMMENDED:
        print(
            f"WARNING: only {len(df)} usable trades found (recommended: "
            f"{MIN_TRADES_RECOMMENDED}+). The model below may be unreliable "
            "— consider running a full-universe backtest for more history "
            "before trusting these results."
        )

    return df


def main():
    df = load_training_data()

    # Chronological split so we're validating on the future relative to
    # training data, not randomly-shuffled dates (which would leak
    # information and make the model look better than it really is).
    df = df.sort_values("signal_date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURES], (train_df[TARGET] > 0).astype(int)
    X_test, y_test = test_df[FEATURES], (test_df[TARGET] > 0).astype(int)

    print(f"Training on {len(X_train)} trades, validating on {len(X_test)} more recent trades.")
    print(f"Baseline win rate in training data : {y_train.mean()*100:.1f}%")
    print(f"Baseline win rate in test data      : {y_test.mean()*100:.1f}%\n")

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("  VALIDATION RESULTS (held-out, most recent 20% of trades)")
    print("=" * 60)
    print(f"Accuracy : {accuracy_score(y_test, preds)*100:.1f}%")
    if len(np.unique(y_test)) > 1:
        print(f"ROC-AUC  : {roc_auc_score(y_test, probs):.3f}  (0.5 = no better than a coin flip, 1.0 = perfect)")
    print()
    print(classification_report(y_test, preds, target_names=["Loss", "Win"]))

    print("Feature importances (what the model actually relies on):")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<20s} {imp:.3f}")

    os.makedirs("ml", exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_FILE)
    print(f"\nSaved model to {MODEL_FILE}")

    if len(np.unique(y_test)) > 1 and roc_auc_score(y_test, probs) < 0.55:
        print(
            "\nNOTE: ROC-AUC is close to 0.5, meaning this model is barely "
            "better than guessing on held-out data. That's a real, useful "
            "finding — it suggests these features alone don't strongly "
            "predict outcomes yet. Consider adding more history (a bigger "
            "backtest) or new features before relying on MLScore."
        )


if __name__ == "__main__":
    main()
