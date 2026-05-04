import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

FEATURE_COLS = ['cpu', 'memory', 'max_usage', 'cpu_mean', 'memory_mean',
                'cpu_std', 'memory_std', 'resource_pressure',
                'cpu_delta', 'memory_delta']

def train_models(df, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)

    # Use only columns that exist in the dataframe
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df['fault'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # ── Random Forest ────────────────────────────────────────────────────────
    print("[train] Fitting Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators   = 100,
        max_depth       = None,
        min_samples_leaf= 2,
        class_weight    = 'balanced',
        n_jobs          = -1,
        random_state    = 42,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    print("[RF] Classification Report:")
    print(classification_report(y_test, rf_preds, target_names=['Normal','Fault']))
    print(f"[RF] ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")
    joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))
    print(f"[RF] Saved → {model_dir}/random_forest.pkl")

    # ── Isolation Forest ─────────────────────────────────────────────────────
    print("[train] Fitting Isolation Forest ...")
    iso = IsolationForest(
        n_estimators = 100,
        contamination= 0.08,    # approx fault rate
        max_samples  = 'auto',
        random_state = 42,
        n_jobs       = -1,
    )
    iso.fit(X_train)
    iso_preds_raw = iso.predict(X_test)
    iso_preds     = (iso_preds_raw == -1).astype(int)
    print("[IsoF] Classification Report:")
    print(classification_report(y_test, iso_preds, target_names=['Normal','Fault']))
    joblib.dump(iso, os.path.join(model_dir, "isolation_forest.pkl"))
    print(f"[IsoF] Saved → {model_dir}/isolation_forest.pkl")

    return rf, iso, X_test, y_test
