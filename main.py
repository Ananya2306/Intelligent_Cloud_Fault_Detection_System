"""
main.py — Full end-to-end ICFDS pipeline.

Usage:
    python main.py

Requires data/raw/cloud_data.csv with columns:
    time, average_usage, maximum_usage, failed
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing     import preprocess_google_data
from src.feature_engineering import create_features
from src.train_ml_models   import train_models
from src.train_lstm        import train_lstm
from src.predict           import predict
from src.evaluate          import evaluate
from src.visualize         import plot

RAW_PATH  = "data/raw/cloud_data.csv"
PROC_PATH = "data/processed/cleaned_data.csv"

def main():
    if not os.path.exists(RAW_PATH):
        print(f"[WARN] Raw dataset not found at {RAW_PATH}")
        print("       Download Google Cluster Trace v3 and place it there.")
        print("       Starting backend-only mode...")
        import uvicorn
        uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
        return

    # Step 1 — Preprocess
    print("\n" + "="*55)
    print("  STEP 1 — DATA PREPROCESSING")
    print("="*55)
    df = preprocess_google_data(RAW_PATH, PROC_PATH)

    # Step 2 — Feature engineering
    print("\n" + "="*55)
    print("  STEP 2 — FEATURE ENGINEERING")
    print("="*55)
    df = create_features(df)

    # Step 3 — Train ML models
    print("\n" + "="*55)
    print("  STEP 3 — TRAIN ML MODELS")
    print("="*55)
    rf, iso, X_test, y_test = train_models(df)

    # Step 4 — Train LSTM
    print("\n" + "="*55)
    print("  STEP 4 — TRAIN LSTM")
    print("="*55)
    train_lstm(df)

    # Step 5 — Run predictions
    print("\n" + "="*55)
    print("  STEP 5 — PREDICTION")
    print("="*55)
    df = predict(df)

    # Step 6 — Evaluate
    print("\n" + "="*55)
    print("  STEP 6 — EVALUATION")
    print("="*55)
    metrics = evaluate(df)

    # Step 7 — Visualize
    print("\n" + "="*55)
    print("  STEP 7 — VISUALIZATION")
    print("="*55)
    plot(df, metrics)

    print("\n✅  FULL PIPELINE COMPLETE")
    print("   Models saved  → models/")
    print("   Plots saved   → outputs/graphs/")
    print("   Run backend   → uvicorn backend.app:app --reload")

if __name__ == "__main__":
    main()
