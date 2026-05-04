"""
predict.py — Run inference with all trained models.
"""
import joblib
import numpy as np
import pandas as pd

FEATURE_COLS = ['cpu', 'memory', 'max_usage']


def _rule_layer(cpu: float, memory: float) -> tuple:
    if cpu > 0.85:
        return 1, True, "CPU > 85%"
    if cpu > 0.75 and memory > 0.75:
        return 1, True, "CPU+MEM > 75%"
    if memory > 0.90:
        return 1, True, "MEM > 90%"
    return None, False, ""


def predict(df: pd.DataFrame,
            rf_path : str = "models/random_forest.pkl",
            iso_path: str = "models/isolation_forest.pkl",
            lstm_path: str = "models/lstm_model.h5") -> pd.DataFrame:

    X = df[FEATURE_COLS].values

    # RF
    rf = joblib.load(rf_path)
    df['rf_pred']  = rf.predict(X)
    df['rf_proba'] = rf.predict_proba(X)[:, 1]

    # Isolation Forest
    iso = joblib.load(iso_path)
    df['iso_pred'] = (iso.predict(X) == -1).astype(int)

    # LSTM (optional)
    try:
        from tensorflow.keras.models import load_model
        lstm = load_model(lstm_path)
        X_seq = X.reshape((X.shape[0], 1, X.shape[1]))
        df['lstm_pred'] = (lstm.predict(X_seq, verbose=0) > 0.5).astype(int).flatten()
    except Exception:
        df['lstm_pred'] = df['rf_pred']   # fallback

    # Hybrid decision layer
    final, rules = [], []
    for _, row in df.iterrows():
        pred, hit, reason = _rule_layer(row['cpu'], row['memory'])
        if hit:
            final.append(pred)
            rules.append(reason)
        else:
            final.append(int(row['rf_pred']))
            rules.append("")

    df['final_pred']   = final
    df['rule_reason']  = rules

    print("[predict] Done. Sample:")
    print(df[['cpu', 'memory', 'max_usage', 'rf_pred', 'iso_pred', 'lstm_pred',
              'final_pred', 'rule_reason']].head(5))
    return df
